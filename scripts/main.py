import torch
import tifffile
import os
import numpy as np
import cupy as cp
from instanseg import InstanSeg
from cellpose.utils import stitch3D
from cellstitch.pipeline import full_stitch
from cellstitch import preprocessing_cupy as ppc


# Set params
img = r"E:\1_DATA\Rheenen\tvl_jr\3d stitching\unmixed3\unmixed.tif"  # Or img: numpy.ndarray
stitch_method = "cellstitch"  # "iou" or "cellstitch"
mode = (
    "nuclei_cells"  # Instanseg segmentation mode: "nuclei" or "cells" or "nuclei_cells"
)
out_path = None  # Set to None to write to the input file location (if provided)
pixel_size = None  # microns per pixel
z_step = None  # microns per pixel
bleach_correct = True
verbose = True

# Check cuda
if cp.cuda.is_available():
    print("CUDA is available. Using device", cp.cuda.get_device_id())
else:
    print("CUDA is not available; using CPU.")

# Read image file
if os.path.isfile(img):
    with tifffile.TiffFile(img) as tif:
        img = tif.asarray()  # ZCYX
        metadata = tif.imagej_metadata or {}
        del tif
elif type(img) != np.ndarray:
    print("img must either be a path to an existing image, or a numpy ndarray.")
    quit()

# Check image dimensions
if img.ndim != 4:
    print("Expected a 4D image (ZCYX), while the img dimensions are ", img.ndim)
    quit()

# Set pixelsizes
if pixel_size is None:
    if 'Info' in metadata:
        info = metadata['Info'].split()
        pixel_size = 1 / float([s for s in info if "XResolution" in s][0].split('=')[-1])  # Oh my gosh
    else:
        print("Could not find the pixel_size in the metadata. The output might not be fully reliable.")
if z_step is None:
    if 'Info' in metadata:
        info = metadata['Info'].split()
        z_step = float([s for s in info if "spacing" in s][0].split('=')[-1])  # At least it's pretty fast
    else:
        print("Could not find the z_step in the metadata. The output might not be fully reliable.")

# Set up output path
if out_path is None and os.path.isfile(img):
    out_path = os.path.split(img)[0]
elif not os.path.exists(out_path):
    os.makedirs(out_path)

# Instanseg-based pipeline
model = InstanSeg("fluorescence_nuclei_and_cells")

# Correct bleaching over Z-axis
if bleach_correct:
    img = ppc.histogram_correct(img).transpose(1, 2, 3, 0)  # ZCYX -> CYXZ
    cp._default_memory_pool.free_all_blocks()
    if verbose:
        print("Finished bleach correction.")
else:
    img = img.transpose(1, 2, 3, 0)  # ZCYX -> CYXZ

# Segment over Z-axis
if verbose:
    print("Segmenting YX planes (Z-axis).")
yx_masks = ppc.segmentation(img, model, pixel_size, mode).transpose(
    2, 0, 1
)  # YXZ -> ZYX
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache
tifffile.imwrite(os.path.join(out_path, "yx_masks.tif"), yx_masks)

if stitch_method == "iou":

    # Memory cleanup
    del model, img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

    if verbose:
        print("Running IoU stitching...")

    iou_masks = stitch3D(yx_masks, stitch_threshold=0.25)
    tifffile.imwrite(os.path.join(out_path, "iou_masks.tif"), iou_masks)

elif stitch_method == "cellstitch":

    # Segment over X-axis
    if verbose:
        print("Segmenting YZ planes (X-axis).")
    transposed_img = img.transpose(0, 1, 3, 2)  # CYXZ -> CYZX
    transposed_img, padding = ppc.upscale_pad_img(
        transposed_img, pixel_size, z_step
    )  # Preprocess YZ planes
    cp._default_memory_pool.free_all_blocks()
    yz_masks = ppc.segmentation(transposed_img, model, pixel_size, mode)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
    yz_masks = ppc.crop_downscale_mask(
        yz_masks, padding, pixel_size, z_step
    ).transpose(
        1, 0, 2
    )  # YZX -> ZYX
    cp._default_memory_pool.free_all_blocks()
    tifffile.imwrite(os.path.join(out_path, "yz_masks.tif"), yz_masks)

    # Segment over Y-axis
    if verbose:
        print("Segmenting XZ planes (Y-axis).")
    transposed_img = img.transpose(0, 2, 3, 1)  # CYXZ -> CXZY
    transposed_img, padding = ppc.upscale_pad_img(
        transposed_img, pixel_size, z_step
    )  # Preprocess XZ planes
    cp._default_memory_pool.free_all_blocks()
    xz_masks = ppc.segmentation(transposed_img, model, pixel_size, mode)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
    xz_masks = ppc.crop_downscale_mask(
        xz_masks, padding, pixel_size, z_step
    ).transpose(
        1, 2, 0
    )  # XZY -> ZYX
    cp._default_memory_pool.free_all_blocks()
    tifffile.imwrite(os.path.join(out_path, "xz_masks.tif"), xz_masks)

    # Memory cleanup
    del model, img, transposed_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

    if verbose:
        print("Running CellStitch stitching...")

    cellstitch_masks = full_stitch(yx_masks, yz_masks, xz_masks, verbose=verbose)

    tifffile.imwrite(os.path.join(out_path, "cellstitch_masks.tif"), cellstitch_masks)

else:
    print("Incompatible stitching method. Supported options are \'iou\' and \'cellstitch\'.")
