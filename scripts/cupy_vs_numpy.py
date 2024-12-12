import torch
import tifffile
import os
import cupy as cp
from instanseg import InstanSeg
from cellstitch import preprocessing as pp
from cellstitch import preprocessing_cupy as ppc
import time

file_path = r"E:\1_DATA\Rheenen\tvl_jr\3d stitching\unmixed3\unmixed.tif"

mode = "nuclei_cells"  # Segmentation mode: "nuclei" or "cells" or "nuclei_cells"

# Read image file
with tifffile.TiffFile(file_path) as tif:
    img_raw = tif.asarray()  # ZCYX
    metadata = tif.imagej_metadata or {}
del tif

# Instanseg-based pipeline
x_resolution = 2.2
pixel_size = 1 / x_resolution
z_resolution = 3.5

# CuPy

model = InstanSeg("fluorescence_nuclei_and_cells")

time_start = time.time()

img = ppc.histogram_correct(img_raw).transpose(1, 2, 3, 0).get()  # ZCYX -> CYXZ
cp._default_memory_pool.free_all_blocks()

# Segment over Z-axis
yx_masks = ppc.segmentation(img, model, pixel_size, mode).transpose(2, 0, 1)  # YXZ -> ZYX
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache

# Segment over X-axis
transposed_img = img.transpose(0, 1, 3, 2)  # CYXZ -> CYZX
transposed_img, padding = ppc.upscale_pad_img(transposed_img, pixel_size, z_resolution)  # Preprocess YZ planes
cp._default_memory_pool.free_all_blocks()
yz_masks = ppc.segmentation(transposed_img, model, pixel_size, mode)
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache
yz_masks = ppc.crop_downscale_mask(yz_masks, padding, pixel_size, z_resolution).transpose(1, 0, 2).get()  # YZX -> ZYX
cp._default_memory_pool.free_all_blocks()

# Segment over Y-axis
transposed_img = img.transpose(0, 2, 3, 1)  # CYXZ -> CXZY
transposed_img, padding = ppc.upscale_pad_img(transposed_img, pixel_size, z_resolution)  # Preprocess XZ planes
cp._default_memory_pool.free_all_blocks()
xz_masks = ppc.segmentation(transposed_img, model, pixel_size, mode)
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache
xz_masks = ppc.crop_downscale_mask(xz_masks, padding, pixel_size, z_resolution).transpose(1, 2, 0).get()  # XZY -> ZYX
cp._default_memory_pool.free_all_blocks()

# Memory cleanup
del model, img, transposed_img
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache

print("CuPy preprocessing time:", time.time()-time_start)

# NumPy

model = InstanSeg("fluorescence_nuclei_and_cells")

time_start = time.time()

img = pp.histogram_correct(img_raw).transpose(1, 2, 3, 0)  # ZCYX -> CYXZ

# Segment over Z-axis
yx_masks = pp.segmentation(img, model, pixel_size, mode).transpose(2, 0, 1)  # YXZ -> ZYX
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache

# Segment over X-axis
transposed_img = img.transpose(0, 1, 3, 2)  # CYXZ -> CYZX
transposed_img, padding = pp.upscale_pad_img(transposed_img, pixel_size, z_resolution)  # Preprocess YZ planes
yz_masks = pp.segmentation(transposed_img, model, pixel_size, mode)
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache
yz_masks = pp.crop_downscale_mask(yz_masks, padding, pixel_size, z_resolution).transpose(1, 0, 2)  # YZX -> ZYX

# Segment over Y-axis
transposed_img = img.transpose(0, 2, 3, 1)  # CYXZ -> CXZY
transposed_img, padding = pp.upscale_pad_img(transposed_img, pixel_size, z_resolution)  # Preprocess XZ planes
xz_masks = pp.segmentation(transposed_img, model, pixel_size, mode)
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache
xz_masks = pp.crop_downscale_mask(xz_masks, padding, pixel_size, z_resolution).transpose(1, 2, 0)  # XZY -> ZYX

# Memory cleanup
del model, img, transposed_img
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache


print("NumPy preprocessing time:", time.time()-time_start)
