import cupy as cp
import numpy as np
import torch
from cupyx.scipy.ndimage import zoom


def crop_downscale_mask(masks, pad: int = 0, pixel=None, z_res=None):
    if not pixel:
        pixel = 1
    if not z_res:
        z_res = 1

    if pad != 0:
        masks = masks[:, pad:-pad, :]  # iZk
    masks = cp.asarray(masks)

    anisotropy = z_res / pixel
    zoom_factors = (1, 1 / anisotropy, 1)
    order = 0  # 0 nearest neighbor, 1 bilinear, 2 quadratic, 3 bicubic

    masks = zoom(masks, zoom_factors, order=order).get()
    cp._default_memory_pool.free_all_blocks()

    return masks


def upscale_pad_img(images, pixel=None, z_res=None):
    if not pixel:
        pixel = 1
    if not z_res:
        z_res = 1

    anisotropy = z_res / pixel
    zoom_factors = (1, anisotropy, 1)
    order = 1  # 0 nearest neighbor, 1 bilinear, 2 quadratic, 3 bicubic

    zoomed = []
    for ch in images:
        ch = zoom(cp.asarray(ch), zoom_factors, order=order).get()
        cp._default_memory_pool.free_all_blocks()
        zoomed.append(ch)

    images = np.stack(zoomed)
    cp._default_memory_pool.free_all_blocks()

    padding_width = 0

    if images.shape[-2] < 512:
        padding_width = (512 - images.shape[-2]) // 2
        images = np.pad(
            images,
            ((0, 0), (0, 0), (padding_width, padding_width), (0, 0)),
            constant_values=0,
        )

    return images, padding_width


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
    d, model, batch_size, pixel=None, m: str = "nuclei_cells"
):
    res, image_tensor = model.eval_medium_image(
        d,
        pixel,
        target="all_outputs",
        cleanup_fragments=True,
        tile_size=1024,
        batch_size=batch_size,
        normalise=False,  # We already have normalized our data beforehand
    )
    return res[0]


def segment_single_slice_small(d, model, pixel=None):
    res, image_tensor = model.eval_small_image(
        d,
        pixel,
        target="all_outputs",
        cleanup_fragments=True,
        normalise=False,  # We already have normalized our data beforehand
    )
    return res[0]


def segmentation(d, model, pixel=None, m: str = "nuclei_cells", xy: bool = False):
    empty_res = np.zeros_like(d[0])

    mode = 1  # Base for 'nuclei_cells' and 'cells'
    if m == "nuclei":
        mode = 0

    nuclei_cells = False
    if xy and m == "nuclei_cells":
        nuclei = empty_res.copy()
        nuclei_cells = True

    nslices = d.shape[-1]

    d = normalize_img(d)  # Pre-normalize data

    size = d.shape[0] * d.shape[1] * d.shape[2]
    if size < 21233664:  # For small images (9x1536x1536 as a base)
        for xyz in range(nslices):
            res_slice = segment_single_slice_small(d[:, :, :, xyz], model, pixel)
            empty_res[:, :, xyz] = res_slice[mode]
            if nuclei_cells:
                nuclei[:, :, xyz] = res_slice[0]
    else:  # For large images
        if d.shape[0] > 5:
            batch = torch.cuda.mem_get_info()[0] // 1024**3 // round(d.shape[0]/4)  # An approximation for 1024x1024
        else:
            batch = torch.cuda.mem_get_info()[0] // 1024**3  # Up to 5 channels would have (less than) ~1 GB VRAM usage
        for xyz in range(nslices):
            res_slice = segment_single_slice_medium(
                d[:, :, :, xyz], model, batch, pixel
            )

            empty_res[:, :, xyz] = res_slice[mode]
            if nuclei_cells:
                nuclei[:, :, xyz] = res_slice[0]
    if nuclei_cells:
        return empty_res, nuclei
    return empty_res


def normalize_img(img, percentile=0.1, epsilon: float = 1e-3):
    """"
    Normalize all planes of a given axis (in this case, the last axis). The image is split into channels to relieve GPU
    strain.

    Adapted from Instanseg (instanseg.utils.utils.percentile_normalize()).
    """
    im = []
    for c in range(img.shape[0]):
        im_temp = cp.asarray(img[c])
        p_min = cp.percentile(im_temp, percentile, axis=(0, 1), keepdims=True).get()
        cp._default_memory_pool.free_all_blocks()
        p_max = cp.percentile(im_temp, 100 - percentile, axis=(0, 1), keepdims=True).get()
        im_temp = im_temp.get()
        cp._default_memory_pool.free_all_blocks()
        im.append(((im_temp - p_min) / np.maximum(epsilon, p_max - p_min)).astype("float32"))

    return np.stack(im)
