import numpy as np
import torch
import tifffile
from instanseg import InstanSeg
from cellstitch.pipeline import full_stitch
from scipy.ndimage import zoom
import os
from joblib import Parallel, delayed


def crop_downscale_mask(masks: np.array, pad: int = 0, pixel=None, z_res=None):
    if not pixel:
        pixel = 1
    if not z_res:
        z_res = 1

    masks = masks.transpose(2, 0, 1)  # iZk --> kiZ

    if pad != 0:
        masks = masks[:, :, pad:-pad]

    anisotropy = z_res / pixel
    zoom_factors = (1, 1/anisotropy)
    order = 0  # 0 nearest neighbor, 1 bilinear, 2 quadratic, 3 bicubic

    args_list = [
        (
            plane,
            zoom_factors,
            order,
        )
        for plane in masks
    ]

    masks = Parallel(n_jobs=-1)(delayed(_scale)(*args) for args in args_list)

    masks = np.stack(masks).transpose(1, 2, 0)  # kiZ --> iZk

    return masks


def upscale_pad_img(images: np.array, pixel=None, z_res=None):
    if not pixel:
        pixel = 1
    if not z_res:
        z_res = 1

    anisotropy = z_res / pixel
    zoom_factors = (1, 1, anisotropy)

    order = 1  # 0 nearest neighbor, 1 bilinear, 2 quadratic, 3 bicubic

    images = images.transpose(3, 0, 1, 2)  # Cijk --> kCij

    args_list = [
        (
            plane,
            zoom_factors,
            order,
        )
        for plane in images
    ]

    images = Parallel(n_jobs=-1)(delayed(_scale)(*args) for args in args_list)

    images = np.stack(images).transpose(1, 2, 3, 0)  # kCij --> Cijk

    padding_width = 0

    if images.shape[-2] < 512:
        padding_width = (512 - images.shape[-2]) // 2
        images = np.pad(
            images,
            ((0, 0), (0, 0), (padding_width, padding_width), (0, 0)),
            constant_values=0
        )

    return images, padding_width


def _scale(plane, zoom_factors, order):
    plane = zoom(np.array(plane), zoom_factors, order=order)

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


def segment_single_slice_medium(d, model, batch_size, pixel=None, m="nuclei_cells"):
    res, image_tensor = model.eval_medium_image(
        d,
        pixel,
        target="all_outputs",
        cleanup_fragments=True,
        tile_size=1024,
        batch_size=batch_size,
    )

    if m == "nuclei":
        res = np.array(res[0][0], dtype='uint')
    elif m == "cells":
        res = np.array(res[0][1], dtype='uint')
    elif m == "nuclei_cells":
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
        res = nuclear_cells

    return res


def segment_single_slice_small(d, model, pixel=None, m="nuclei_cells"):
    res, image_tensor = model.eval_small_image(
        d,
        pixel,
        target="all_outputs",
        cleanup_fragments=True,
    )

    if m == "nuclei":
        res = np.array(res[0][0], dtype='uint')
    elif m == "cells":
        res = np.array(res[0][1], dtype='uint')
    elif m == "nuclei_cells":
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
        res = nuclear_cells

    return res


def iterative_segmentation(d, model, pixel=None, m="nuclei_cells"):
    empty_res = np.zeros_like(d[0])
    nslices = d.shape[-1]
    if d.shape[1] < 1024 or d.shape[2] < 1024:  # For small images
        for xyz in range(nslices):
            res_slice = segment_single_slice_small(
                d[:, :, :, xyz], model, pixel, m
            )
            empty_res[:, :, xyz] = res_slice
    else:  # For large images
        batch = (torch.cuda.mem_get_info()[0] // 1024 ** 3 // 4)
        for xyz in range(nslices):
            res_slice = segment_single_slice_medium(
                d[:, :, :, xyz], model, batch, pixel, m
            )  # Count up from the previous z-slice
            empty_res[:, :, xyz] = res_slice
    return empty_res


file_path = r"E:\1_DATA\Rheenen\tvl_jr\3d stitching\unmixed\unmixed.tif"
out_path = os.path.split(file_path)[0]

# Read image file
with tifffile.TiffFile(file_path) as tif:
    img = tif.asarray()  # ZCYX
    metadata = tif.imagej_metadata or {}

# Instanseg-based pipeline
x_resolution = 2.2
pixel_size = 1 / x_resolution
z_resolution = 3.5

mode = "nuclei"  # "nuclei" or "cells" or "nuclei_cells"
model = InstanSeg("fluorescence_nuclei_and_cells")

img = histogram_correct(img)

# Segment over Z-axis
transposed_img = img.transpose(1, 2, 3, 0)  # ZCYX -> CYXZ
yx_masks = iterative_segmentation(transposed_img, model, pixel_size, mode).transpose(2, 0, 1)  # YXZ -> ZYX
tifffile.imwrite(os.path.join(out_path, "yx_masks.tif"), yx_masks)

# Segment over X-axis
transposed_img = img.transpose(1, 2, 0, 3)  # ZCYX -> CYZX
transposed_img, padding = upscale_pad_img(transposed_img, pixel_size, z_resolution)  # Preprocess YZ planes
yz_masks = iterative_segmentation(transposed_img, model, pixel_size, mode)
yz_masks = crop_downscale_mask(yz_masks, padding, pixel_size, z_resolution).transpose(1, 0, 2)  # YZX -> ZYX
tifffile.imwrite(os.path.join(out_path, "yz_masks.tif"), yz_masks)

# Segment over Y-axis
transposed_img = img.transpose(1, 3, 0, 2)  # ZCYX -> CXZY
transposed_img, padding = upscale_pad_img(transposed_img, pixel_size, z_resolution)  # Preprocess XZ planes
xz_masks = iterative_segmentation(transposed_img, model, pixel_size, mode)
xz_masks = crop_downscale_mask(xz_masks, padding, pixel_size, z_resolution).transpose(1, 2, 0)  # XZY -> ZYX
tifffile.imwrite(os.path.join(out_path, "xz_masks.tif"), xz_masks)

# Memory cleanup
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache

print("Stitching...")

cellstitch_masks = full_stitch(yx_masks, yz_masks, xz_masks)

tifffile.imwrite(os.path.join(out_path, "cellstitch_masks.tif"), cellstitch_masks)
