import torch
import tifffile
import os
from instanseg import InstanSeg
from cellpose.utils import stitch3D
from cellstitch.pipeline import full_stitch
from cellstitch import preprocessing as pp


stitch_method = "iou"  # "iou" or "cellstitch"
file_path = r"E:\1_DATA\Rheenen\tvl_jr\3d stitching\unmixed2\unmixed.tif"
out_path = os.path.split(file_path)[0]

mode = "nuclei"  # Segmentation mode: "nuclei" or "cells" or "nuclei_cells"

# Read image file
with tifffile.TiffFile(file_path) as tif:
    img = tif.asarray()  # ZCYX
    metadata = tif.imagej_metadata or {}

# Instanseg-based pipeline
x_resolution = 2.2
pixel_size = 1 / x_resolution
z_resolution = 3.5

model = InstanSeg("fluorescence_nuclei_and_cells")

img = pp.histogram_correct(img)

# Segment over Z-axis
transposed_img = img.transpose(1, 2, 3, 0)  # ZCYX -> CYXZ
yx_masks = pp.segmentation(transposed_img, model, pixel_size, mode).transpose(2, 0, 1)  # YXZ -> ZYX
tifffile.imwrite(os.path.join(out_path, "yx_masks.tif"), yx_masks)

if stitch_method == "iou":

    # Memory cleanup
    del model, img, transposed_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

    print("Running IoU stitching...")

    iou_masks = stitch3D(yx_masks, stitch_threshold=0.25)
    tifffile.imwrite(os.path.join(out_path, "iou_masks.tif"), iou_masks)

elif stitch_method == "cellstitch":

    # Segment over X-axis
    transposed_img = img.transpose(1, 2, 0, 3)  # ZCYX -> CYZX
    transposed_img, padding = pp.upscale_pad_img(transposed_img, pixel_size, z_resolution)  # Preprocess YZ planes
    yz_masks = pp.segmentation(transposed_img, model, pixel_size, mode)
    yz_masks = pp.crop_downscale_mask(yz_masks, padding, pixel_size, z_resolution).transpose(1, 0, 2)  # YZX -> ZYX
    tifffile.imwrite(os.path.join(out_path, "yz_masks.tif"), yz_masks)

    # Segment over Y-axis
    transposed_img = img.transpose(1, 3, 0, 2)  # ZCYX -> CXZY
    transposed_img, padding = pp.upscale_pad_img(transposed_img, pixel_size, z_resolution)  # Preprocess XZ planes
    xz_masks = pp.segmentation(transposed_img, model, pixel_size, mode)
    xz_masks = pp.crop_downscale_mask(xz_masks, padding, pixel_size, z_resolution).transpose(1, 2, 0)  # XZY -> ZYX
    tifffile.imwrite(os.path.join(out_path, "xz_masks.tif"), xz_masks)

    # Memory cleanup
    del model, img, transposed_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

    print("Running CellStitch stitching...")

    cellstitch_masks = full_stitch(yx_masks, yz_masks, xz_masks)

    tifffile.imwrite(os.path.join(out_path, "cellstitch_masks.tif"), cellstitch_masks)

else:
    print("Incompatible stitching method.")
