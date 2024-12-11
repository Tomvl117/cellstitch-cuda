import numpy as np
import torch
import tifffile
from instanseg import InstanSeg
from cellstitch.pipeline import full_stitch
from scipy.ndimage import zoom
import os
from joblib import Parallel, delayed


def upscale_pad_img(images: np.array, pixel=None, z_res=None):
    if not pixel:
        pixel = 1
    if not z_res:
        z_res = 1

    anisotropy = z_res / pixel
    zoom_factors = (1, 1, anisotropy)

    images = images.transpose(3, 0, 1, 2)  # Cijk --> kCij

    args_list = [
        (
            plane,
            zoom_factors,
        )
        for plane in images
    ]

    images = Parallel(n_jobs=-1)(delayed(_upscale)(*args) for args in args_list)

    images = np.stack(images).transpose(1, 2, 3, 0)  # kCij --> Cijk

    if images.shape[-2] < 512:
        padding_width = (512 - images.shape[-2]) // 2
        images = np.pad(
            images,
            ((0, 0), (0, 0), (padding_width, padding_width), (0, 0)),
            constant_values=0
        )

    return images


def _upscale(plane, zoom_factors):
    plane = zoom(np.array(plane), zoom_factors, order=1)

    return plane


def histogram_correct(images: np.array, match: str = "first"):
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

    args_list = [
        (
            channel,
            match,
        )
        for channel in images
    ]

    images = Parallel(n_jobs=-1)(delayed(_correct)(*args) for args in args_list)

    images = np.array(images, dtype=dtype)

    images = images.transpose(1, 0, 2, 3)  # CZYX --> ZCYX

    return images


def _correct(channel, match):

    channel = np.array(channel)
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

            val, ix, cnt = np.unique(
                channel[i, ...].flatten(), return_inverse=True, return_counts=True
            )
            cdf = np.cumsum(cnt) / pixel_size

            interpolated = np.interp(cdf, cdfs[match_ix], values[match_ix])
            channel[i, ...] = interpolated[ix]

        if i == 0 or match == "neighbor":
            val, cnt = np.unique(channel[i, ...].flatten(), return_counts=True)
            cdf = np.cumsum(cnt) / pixel_size
            values.append(val)
            cdfs.append(cdf)

    channel = channel.reshape(k, m, n)

    return channel


def segment_single_slice_medium(d, model, batch_size, pixel=None):
    res, image_tensor = model.eval_medium_image(
        d,
        pixel,
        target="all_outputs",
        cleanup_fragments=True,
        tile_size=1024,
        batch_size=batch_size,
    )
    res = np.array(res[0], dtype='uint')

    # Initialize new label ID
    new_label_id = 0

    nuclear_cells = np.zeros_like(res[1])

    for label_id in np.unique(res[0]):
        if label_id != 0:
            # Find the coordinates of the current label in the nuclei layer
            coords = np.argwhere(res[0] == label_id)

            # Check if any of these coordinates are also labeled in the cell layer
            colocalized = False
            for coord in coords:
                if res[1][coord[0], coord[1]] != 0:
                    # If the nuclear label is colocalized with a cell label, save the cell label
                    cell_id = res[1][coord[0], coord[1]]
                    colocalized = True
                    break

            # If colocalized, assign a new label ID
            if colocalized:
                nuclear_cells[res[1] == cell_id] = new_label_id
                new_label_id += 1

    return nuclear_cells


def segment_single_slice_small(d, model, pixel=None):
    res, image_tensor = model.eval_small_image(
        d,
        pixel,
        target="all_outputs",
        cleanup_fragments=True,
    )
    res = np.array(res[0], dtype='uint')

    # Initialize new label ID
    new_label_id = 0

    nuclear_cells = np.zeros_like(res[1])

    for label_id in np.unique(res[0]):
        if label_id != 0:
            # Find the coordinates of the current label in the nuclei layer
            coords = np.argwhere(res[0] == label_id)

            # Check if any of these coordinates are also labeled in the cell layer
            colocalized = False
            for coord in coords:
                if res[1][coord[0], coord[1]] != 0:
                    # If the nuclear label is colocalized with a cell label, save the cell label
                    cell_id = res[1][coord[0], coord[1]]
                    colocalized = True
                    break

            # If colocalized, assign a new label ID
            if colocalized:
                nuclear_cells[res[1] == cell_id] = new_label_id
                new_label_id += 1

    return nuclear_cells


def iterative_segmentation(d, model, pixel=None):
    empty_res = np.zeros_like(d[0])
    nslices = d.shape[-1]
    if d.shape[1] < 1024 or d.shape[2] < 1024:  # For small images
        for xyz in range(nslices):
            res_slice = segment_single_slice_small(
                d[:, :, :, xyz], model, pixel
            )
            empty_res[:, :, xyz] = res_slice
    else:  # For large images
        batch = (torch.cuda.mem_get_info()[0] // 1024 ** 3 // 4)
        for xyz in range(nslices):
            res_slice = segment_single_slice_medium(
                d[:, :, :, xyz], model, batch, pixel
            )  # Count up from the previous z-slice
            empty_res[:, :, xyz] = res_slice
    return empty_res


file_path = r"E:\1_DATA\Rheenen\tvl_jr\3d stitching\raw.tif"  # Shape: 101, 15, 600, 700
out_path = os.path.split(file_path)[0]

# Read image file
with tifffile.TiffFile(file_path) as tif:
    img = tif.asarray()  # ZCYX
    metadata = tif.imagej_metadata or {}

# img = histogram_correct(img)

# Instanseg-based pipeline
x_resolution = 2.2
pixel_size = 1 / x_resolution
z_resolution = 3.5

model = InstanSeg("fluorescence_nuclei_and_cells")

# Segment over Z-axis
transposed_img = img.transpose(1, 2, 3, 0)  # ZCYX -> CYXZ
xy_masks = iterative_segmentation(transposed_img, model, pixel_size).transpose(2, 0, 1)  # YXZ -> ZYX
tifffile.imwrite(os.path.join(out_path, "xy_masks.tif"), xy_masks)

# Segment over X-axis
transposed_img = img.transpose(1, 2, 0, 3)  # ZCYX -> CYZX
transposed_img = upscale_pad_img(transposed_img, pixel_size, z_resolution)
yz_masks = iterative_segmentation(transposed_img, model, pixel_size).transpose(1, 0, 2)  # YZX -> ZYX
tifffile.imwrite(os.path.join(out_path, "yz_masks.tif"), yz_masks)

# Segment over Y-axis
transposed_img = img.transpose(1, 3, 0, 2)  # ZCYX -> CXZY
transposed_img = upscale_pad_img(transposed_img, pixel_size, z_resolution)
xz_masks = iterative_segmentation(transposed_img, model, pixel_size).transpose(1, 2, 0)  # XZY -> ZYX
tifffile.imwrite(os.path.join(out_path, "xz_masks.tif"), xz_masks)

# Memory cleanup
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache

print("Stitching...")

cellstitch_masks = full_stitch(xy_masks, yz_masks, xz_masks)

tifffile.imwrite(os.path.join(out_path, "cellstitch_masks.tif"), cellstitch_masks)
