import cupy as cp
import numpy as np
import torch
from torch.utils.data import DataLoader
from cellstitch_cuda.seg_batch import ImageDataset
import sys
from cupyx.scipy.ndimage import zoom


def downscale_mask(masks, pixel=None, z_res=None):
    if not pixel:
        pixel = 1
    if not z_res:
        z_res = 1

    masks = cp.asarray(masks)

    anisotropy = z_res / pixel
    zoom_factors = (1 / (pixel / 0.5), 1 / (anisotropy * (pixel / 0.5)), 1)
    order = 0  # 0 nearest neighbor, 1 bilinear, 2 quadratic, 3 bicubic

    masks = zoom(masks, zoom_factors, order=order).get()
    cp._default_memory_pool.free_all_blocks()

    return masks


def upscale_img(images, pixel=None, z_res=None):
    if not pixel:
        pixel = 1
    if not z_res:
        z_res = 1

    anisotropy = z_res / pixel
    zoom_factors = (
        pixel / 0.5,
        anisotropy * (pixel / 0.5),
        1,
    )  # Pre-zoom for InstanSeg (expected pixel size = 0.5)
    order = 1  # 0 nearest neighbor, 1 bilinear, 2 quadratic, 3 bicubic

    zoomed = []
    for ch in images:  # CiZk
        ch = zoom(cp.asarray(ch), zoom_factors, order=order).get()
        cp._default_memory_pool.free_all_blocks()
        zoomed.append(ch)

    cp._default_memory_pool.free_all_blocks()

    return np.stack(zoomed)


def histogram_correct(images, match: str = "first"):
    """Correct bleaching over a given axis

    This function is used to correct signal degradation that can occur over the Z axis.

    Adapted from napari-bleach-correct: https://github.com/marx-alex/napari-bleach-correct
        Authored by https://github.com/marx-alex
        Original algorithm by Kota Miura: Miura K. Bleach correction ImageJ plugin for compensating the photobleaching
        of time-lapse sequences. F1000Res. 2020 Dec 21;9:1494. https://doi.org/10.12688/f1000research.27171.1

    """
    # cache image dtype
    dtype = images.dtype

    assert (
        3 <= len(images.shape) <= 4
    ), f"Expected 3d or 4d image stack, instead got {len(images.shape)} dimensions"

    avail_match_methods = ["first", "neighbor"]
    assert (
        match in avail_match_methods
    ), f"'match' expected to be one of {avail_match_methods}, instead got {match}"

    images = images.transpose(1, 0, 2, 3)  # ZCYX --> CZYX

    corrected = []
    for ch in images:
        ch = _correct(cp.asarray(ch), match).get()
        cp._default_memory_pool.free_all_blocks()
        corrected.append(ch)

    images = np.stack(corrected, axis=1, dtype=dtype)  # ZCYX

    return images


def _correct(channel, match):

    # channel = cp.array(channel)
    k, m, n = channel.shape
    pixel_size = m * n

    # flatten the last dimensions and calculate normalized cdf
    channel = channel.reshape(k, -1)
    values, cdfs = [], []

    for i in range(k):

        if i > 0:
            if match == "first":
                match_ix = 0
            else:
                match_ix = i - 1

            val, ix, cnt = cp.unique(
                channel[i, ...].flatten(), return_inverse=True, return_counts=True
            )
            cdf = cp.cumsum(cnt) / pixel_size

            interpolated = cp.interp(cdf, cdfs[match_ix], values[match_ix])
            channel[i, ...] = interpolated[ix]

        if i == 0 or match == "neighbor":
            val, cnt = cp.unique(channel[i, ...].flatten(), return_counts=True)
            cdf = cp.cumsum(cnt) / pixel_size
            values.append(val)
            cdfs.append(cdf)

    channel = channel.reshape(k, m, n)

    return channel


def segment_single_slice_medium(
    d,
    model,
    tiles,
    batch_size,
):
    res = model.eval_medium_image(
        d,
        return_image_tensor=False,
        pixel_size=None,
        target="all_outputs",
        cleanup_fragments=True,
        tile_size=tiles,
        batch_size=batch_size,
        normalise=True,
    )
    return res[0]


def segment_single_slice_small(d, model):
    res = model.eval_small_image(
        d,
        return_image_tensor=False,
        pixel_size=None,
        target="all_outputs",
        cleanup_fragments=True,
        normalise=True,
    )
    return res[0]


def segment_batch_slice_small(d, model):
    result = []
    for batch in d:
        batch = batch.to(model.inference_device)
        target_segmentation = torch.tensor([1, 1])
        with torch.amp.autocast("cuda"):
            instanseg_kwargs = {"cleanup_fragments": True}
            instances = model.instanseg(
                batch, target_segmentation=target_segmentation, **instanseg_kwargs
            )
        res = instances.cpu()
        result.append(res.numpy().astype("uint32"))
    result = [
        k for b in result for k in b
    ]  # For each z plane found in each batch b in result, stack the z planes
    result = np.stack(result).transpose(
        1, 2, 3, 0
    )  # ZcYX --> cYXZ || kcij --> cijk (for iterator k)

    return result[1], result[0]


def segmentation(d, model, m: str = "nuclei_cells", xy: bool = False):

    mode = 1  # Base for 'nuclei_cells' and 'cells'
    if m == "nuclei":
        mode = 0

    nuclei_cells = False
    if xy and m == "nuclei_cells":
        nuclei_cells = True

    nslices = d.shape[-1]

    vram = torch.cuda.mem_get_info()[0] / 1024  # In KB
    vram_est = 0.1765 * np.prod(
        d.shape[0:3]
    )  # Magic number literally obtained by plotting in Excel

    tiles = 1024
    if vram < vram_est:
        small = False
        vram_est = (
            0.1765 * tiles**2 * d.shape[0]
        )  # Base VRAM estimate on batch size, multiplied by channels
        batch = int(vram / vram_est)
        if batch == 0:
            print(
                "Not enough VRAM available for 1024x1024 tiles. Decreasing to standard 512x512."
            )
            tiles = 512
            vram_est = 0.1765 * tiles**2 * d.shape[0]
            batch = int(vram / vram_est)
            if batch == 0:
                print("Not enough VRAM available for 512x512 tiles. Aborting.")
                sys.exit(1)
    else:
        small = True

    if small:  # For images that fit within VRAM in their entirety:
        batch = int(vram / vram_est)
        if batch > 3:  # If over 3 slices would fit into VRAM in their entirety
            d = d.transpose(3, 0, 1, 2)  # CYXZ --> ZCYX || Cijk --> kCij
            dataset = ImageDataset(
                d
            )  # Create a dataset that handles images similar to Instanseg
            dataloader = DataLoader(
                dataset, batch_size=batch - 2, shuffle=False, drop_last=False
            )  # Leverage torch batching through DataLoader
            empty_res, nuclei = segment_batch_slice_small(dataloader, model)
        else:  # Otherwise, go slice by slice
            empty_res = np.zeros_like(d[0])
            nuclei = empty_res.copy()
            for xyz in range(nslices):
                res_slice = segment_single_slice_small(d[:, :, :, xyz], model)
                empty_res[:, :, xyz] = res_slice[mode]
                if nuclei_cells:
                    nuclei[:, :, xyz] = res_slice[0]
    else:  # For larger images
        empty_res = np.zeros_like(d[0])
        nuclei = empty_res.copy()
        for xyz in range(nslices):
            res_slice = segment_single_slice_medium(
                d[:, :, :, xyz], model, tiles, batch
            )
            empty_res[:, :, xyz] = res_slice[mode]
            if nuclei_cells:
                nuclei[:, :, xyz] = res_slice[0]
    if nuclei_cells:
        return empty_res, nuclei
    return empty_res
