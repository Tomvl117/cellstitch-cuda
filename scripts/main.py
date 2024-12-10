import numpy as np
import torch
import tifffile
from instanseg import InstanSeg
from cellstitch.pipeline import full_stitch
import os


def segment_single_slice_medium(d, model, batch_size, pixel):
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


def segment_single_slice_small(d, model, pixel):
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


def iterative_segmentation(d, model, pixel):
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
    img = tif.asarray().transpose(1, 0, 2, 3)  # ZCYX -> CZYX
    metadata = tif.imagej_metadata or {}

# Instanseg-based pipeline
x_resolution = 2.2
pixel_size = 1 / x_resolution
z_resolution = 3.5
anisotropy = int(z_resolution / pixel_size)

model = InstanSeg("fluorescence_nuclei_and_cells")

# Segment over Z-axis
transposed_img = img.transpose(0, 2, 3, 1)  # CZYX -> CYXZ
xy_masks = iterative_segmentation(transposed_img, model, pixel_size).transpose(2, 0, 1)  # YXZ -> ZYX

# Segment over X-axis
transposed_img = img.transpose(0, 2, 1, 3)  # CZYX -> CYZX
yz_masks = iterative_segmentation(transposed_img, model, pixel_size).transpose(1, 0, 2)  # YZX -> ZYX

# Segment over Y-axis
transposed_img = img.transpose(0, 3, 1, 2)  # CZYX -> CXZY
xz_masks = iterative_segmentation(transposed_img, model, pixel_size).transpose(1, 2, 0)  # XZY -> ZYX

# Memory cleanup
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache

tifffile.imwrite(os.path.join(out_path, "xy_masks.tif"), xy_masks)
tifffile.imwrite(os.path.join(out_path, "yz_masks.tif"), yz_masks)
tifffile.imwrite(os.path.join(out_path, "xz_masks.tif"), xz_masks)

cellstitch_masks = full_stitch(xy_masks, yz_masks, xz_masks)

tifffile.imwrite(os.path.join(out_path, "cellstitch_masks.tif"),
                 cellstitch_masks)
