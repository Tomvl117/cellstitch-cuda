import numpy as np
import torch
import tifffile
from instanseg import InstanSeg
from cellstitch.pipeline import full_stitch


def segment_single_slice_medium(data, model):
    res, image_tensor = model.eval_medium_image(
                    data,
                    pixel_size,
                    target="cells",
                    cleanup_fragments=True,
                    tile_size=1024,
                    batch_size=(torch.cuda.mem_get_info()[0] // 1024**3 // 5),
                )
    res = np.array(res)[0][0]
    return res


def segment_single_slice(data, model):
    res, image_tensor = model.eval_small_image(
                    data,
                    pixel_size,
                    target="cells",
                    cleanup_fragments=True,
                )
    res = np.array(res)[0][0]
    return res


def iterative_segmentation(data, empty_res, nslices, model):
    max_cell = 0
    if data.shape[1] > 1024 and data.shape[2] > 1024:  # Check if tiling is required
        for xyz in range(nslices):
            res_slice = segment_single_slice_medium(
                data[:, :, :, xyz], model
            )
            res_slice = res_slice + max_cell * (
                    res_slice != 0
            )  # Count up from the previous z-slice
            max_cell = np.max(res_slice)
            empty_res[:, :, xyz] = res_slice
    else:
        for xyz in range(nslices):
            res_slice = segment_single_slice(
                data[:, :, :, xyz], model
            )
            res_slice = res_slice + max_cell * (
                    res_slice != 0
            )  # Count up from the previous z-slice
            max_cell = np.max(res_slice)
            empty_res[:, :, xyz] = res_slice
    return empty_res


file_path = r"E:\1_DATA\Rheenen\tvl_jr\SP8\2024Nov4_SI_1mg_AIO-3D\raw-crop.tif"

# Read image file
with tifffile.TiffFile(file_path) as tif:
    img = tif.asarray().transpose(1, 0, 2, 3)  # CZYX = (15, 41, 1000, 998)
    img = img.astype(np.float32)
    metadata = tif.imagej_metadata or {}

# Instanseg-based pipeline
x_resolution = 2.199999
pixel_size = 1 / x_resolution

model = InstanSeg("fluorescence_nuclei_and_cells")


# Segment over Z-axis
num_slices = img.shape[1]
xy_masks = np.zeros_like(img[0, :, :, :]).astype(int).transpose(1, 2, 0)  # ZYX -> YXZ
transposed_img = img.transpose(0, 2, 3, 1)  # CZYX -> CYXZ
xy_masks = iterative_segmentation(transposed_img, xy_masks, num_slices, model).transpose(2, 0, 1)  # YXZ -> ZYX


# Segment over X-axis
num_slices = img.shape[3]
yz_masks = np.zeros_like(img[0, :, :, :]).astype(int).transpose(1, 0, 2)  # ZYX -> YZX
transposed_img = img.transpose(0, 2, 1, 3)  # CZYX -> CYZX
yz_masks = iterative_segmentation(transposed_img, yz_masks, num_slices, model).transpose(1, 0, 2)  # YZX -> ZYX


# Segment over Y-axis
num_slices = img.shape[2]
xz_masks = np.zeros_like(img[0, :, :, :]).astype(int).transpose(2, 0, 1)  # ZYX -> XZY
transposed_img = img.transpose(0, 3, 1, 2)  # CZYX -> CXZY
xz_masks = iterative_segmentation(transposed_img, xz_masks, num_slices, model).transpose(1, 2, 0)  # XZY -> ZYX


# Memory cleanup
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache

cellstitch_masks = full_stitch(xy_masks, yz_masks, xz_masks)

tifffile.imwrite("cellstitch_masks.tif", cellstitch_masks)
