import torch
import tifffile
import os
import sys
import numpy as np
from instanseg import InstanSeg
from cellpose.metrics import _label_overlap
from cellpose.utils import stitch3D
from cellstitch_cuda import preprocessing_cupy as ppc
from cellstitch_cuda.postprocessing_cupy import fill_holes_and_remove_small_masks

from .alignment import *


def relabel_layer(masks, z, lbls):
    """
    Relabel the label in LBLS in layer Z of MASKS.
    """
    layer = masks[z]
    if z != 0:
        reference_layer = masks[z - 1]
    else:
        reference_layer = masks[z + 1]

    overlap = cp.asarray(_label_overlap(reference_layer.get(), layer.get()))

    for lbl in lbls:
        lbl0 = cp.argmax(overlap[:, lbl])
        layer[layer == lbl] = lbl0


def overseg_correction(masks):
    lbls = cp.unique(masks)[1:]

    # get a list of labels that need to be corrected
    layers_lbls = {}

    for lbl in lbls:
        existing_layers = cp.any(masks == lbl, axis=(1, 2))
        depth = cp.sum(existing_layers)

        if depth == 1:
            z = int(cp.where(existing_layers)[0][0])
            layers_lbls.setdefault(z, []).append(lbl)

    for z, lbls in layers_lbls.items():
        relabel_layer(masks, z, lbls)


def full_stitch(xy_masks_prior, yz_masks, xz_masks, verbose=False):
    """
    Stitch masks in-place (top -> bottom).
    """
    xy_masks = xy_masks_prior.copy()
    num_frame = xy_masks.shape[0]
    prev_index = 0

    while Frame(xy_masks[prev_index]).is_empty():
        prev_index += 1

    curr_index = prev_index + 1

    time_start = time.time()

    while curr_index < num_frame:
        cp._default_memory_pool.free_all_blocks()
        if Frame(xy_masks[curr_index]).is_empty():
            # if frame is empty, skip
            curr_index += 1
        else:
            if verbose:
                print(
                    "===Stitching frame %s with frame %s ...==="
                    % (curr_index, prev_index)
                )

            yz_not_stitched = cp.asarray(
                (yz_masks[prev_index] != 0)
                * (yz_masks[curr_index] != 0)
                * (yz_masks[prev_index] != yz_masks[curr_index])
            )
            xz_not_stitched = cp.asarray(
                (xz_masks[prev_index] != 0)
                * (xz_masks[curr_index] != 0)
                * (xz_masks[prev_index] != xz_masks[curr_index])
            )

            fp = FramePair(
                xy_masks[prev_index], xy_masks[curr_index], max_lbl=xy_masks.max()
            )
            fp.stitch(yz_not_stitched, xz_not_stitched, verbose=verbose)
            xy_masks[curr_index] = fp.frame1.mask.get()

            prev_index = curr_index
            curr_index += 1

    if verbose:
        print("Total time to stitch: ", time.time() - time_start)
    time_start = time.time()

    xy_masks = fill_holes_and_remove_small_masks(xy_masks)
    cp._default_memory_pool.free_all_blocks()

    if verbose:
        print("Time to fill holes and remove small masks: ", time.time() - time_start)
    time_start = time.time()

    overseg_correction(xy_masks)

    if verbose:
        print("Time to correct oversegmentation: ", time.time() - time_start)
    return xy_masks.get()


def cellstitch_cuda(
    img,
    output_masks: bool = False,
    output_path=None,
    stitch_method: str = "cellstitch",
    seg_mode: str = "nuclei_cells",
    pixel_size=None,
    z_step=None,
    bleach_correct: bool = True,
    verbose: bool = False,
):
    """ All-in-one function to segment and stitch 2D labels

    Full stitching pipeline, which does the following:
        1. Histogram-based signal degradation correction
        2. Segmentation over the Z axis using InstanSeg
        3. Stitching of 2D planes into 3D labels, by one of two methods:
            a. Cellpose's standard Intersect over Union (IoU) calculation
            b. CellStitch's orthogonal labeling, which leverages Optimal Transport to create robust masks.

    Args:
        img: Either a path pointing to an existing image, or a numpy.ndarray. Must be 4D (ZCYX).
        output_masks: True to write all masks to the output path, or False to only return the final stitched mask.
            Default False
        output_path: Set to None to write to the input file location (if provided). Ignored of output_masks is False.
            Default None
        stitch_method: "iou" for Cellpose IoU stitching, or "cellstitch" for CellStitch stitching.
            Default "cellstitch"
        seg_mode: Instanseg segmentation mode: "nuclei" to only return nuclear masks, "cells" to return all the cell
            masks (including those without nuclei), or "nuclei_cells", which returns only cells with detected nuclei.
            Default "nuclei_cells"
        pixel_size: XY pixel size in microns per pixel. When set to None, will be read from img metadata if possible.
            Default None
        z_step: Z pixel size (z step) in microns per step. When set to None, will be read from img metadata if possible.
            Default None
        bleach_correct: Whether histogram-based signal degradation correction should be applied to img.
            Default False
        verbose: Verbosity.
            Default False
    """

    # Check cuda
    if cp.cuda.is_available():
        print("CUDA is available. Using device", cp.cuda.get_device_id())
    else:
        print("CUDA is not available; using CPU.")

    # Initialize path
    path = ""

    # Read image file
    if os.path.isfile(img):
        path = str(img)
        with tifffile.TiffFile(path) as tif:
            img = tif.asarray()  # ZCYX
            metadata = tif.imagej_metadata or {}
    elif not isinstance(img, np.ndarray):
        print("img must either be a path to an existing image, or a numpy ndarray.")
        sys.exit(1)

    # Check image dimensions
    if img.ndim != 4:
        print("Expected a 4D image (ZCYX), while the img dimensions are ", img.ndim)
        sys.exit(1)

    # Set pixelsizes
    if pixel_size is None and "Info" in metadata:
        info = metadata["Info"].split()
        try:
            pixel_size = 1 / float(
                [s for s in info if "XResolution" in s][0].split("=")[-1]
            )  # Oh my gosh
        except Warning:
            print(
                "No XResolution found in image metadata. The output might not be fully reliable."
            )
    else:
        print(
            "No pixel_size provided. The output might not be fully reliable. If unexpected, check the image metadata."
        )
    if z_step is None and "Info" in metadata:
        info = metadata["Info"].split()
        try:
            z_step = float(
                [s for s in info if "spacing" in s][0].split("=")[-1]
            )  # At least it's pretty fast
        except Warning:
            print("No spacing (Z step) found in image metadata. The output might not be fully reliable.")
    else:
        print(
            "No z_step provided. The output might not be fully reliable. If unexpected, check the image metadata."
        )

    # Set up output path
    if output_masks:
        if output_path is None and os.path.isfile(path):
            output_path = os.path.split(path)[0]
        elif not os.path.exists(output_path):
            os.makedirs(output_path)

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
    yx_masks = ppc.segmentation(img, model, pixel_size, seg_mode).transpose(
        2, 0, 1
    )  # YXZ -> ZYX
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
    if output_masks:
        tifffile.imwrite(os.path.join(output_path, "yx_masks.tif"), yx_masks)

    if stitch_method == "iou":

        # Memory cleanup
        del model, img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache

        if verbose:
            print("Running IoU stitching...")

        iou_masks = stitch3D(yx_masks, stitch_threshold=0.25)

        if output_masks:
            tifffile.imwrite(os.path.join(output_path, "iou_masks.tif"), iou_masks)

        return iou_masks

    elif stitch_method == "cellstitch":

        # Segment over X-axis
        if verbose:
            print("Segmenting YZ planes (X-axis).")
        transposed_img = img.transpose(0, 1, 3, 2)  # CYXZ -> CYZX
        transposed_img, padding = ppc.upscale_pad_img(
            transposed_img, pixel_size, z_step
        )  # Preprocess YZ planes
        cp._default_memory_pool.free_all_blocks()
        yz_masks = ppc.segmentation(transposed_img, model, pixel_size, seg_mode)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
        yz_masks = ppc.crop_downscale_mask(
            yz_masks, padding, pixel_size, z_step
        ).transpose(
            1, 0, 2
        )  # YZX -> ZYX
        cp._default_memory_pool.free_all_blocks()
        if output_masks:
            tifffile.imwrite(os.path.join(output_path, "yz_masks.tif"), yz_masks)

        # Segment over Y-axis
        if verbose:
            print("Segmenting XZ planes (Y-axis).")
        transposed_img = img.transpose(0, 2, 3, 1)  # CYXZ -> CXZY
        transposed_img, padding = ppc.upscale_pad_img(
            transposed_img, pixel_size, z_step
        )  # Preprocess XZ planes
        cp._default_memory_pool.free_all_blocks()
        xz_masks = ppc.segmentation(transposed_img, model, pixel_size, seg_mode)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
        xz_masks = ppc.crop_downscale_mask(
            xz_masks, padding, pixel_size, z_step
        ).transpose(
            1, 2, 0
        )  # XZY -> ZYX
        cp._default_memory_pool.free_all_blocks()
        if output_masks:
            tifffile.imwrite(os.path.join(output_path, "xz_masks.tif"), xz_masks)

        # Memory cleanup
        del model, img, transposed_img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache

        if verbose:
            print("Running CellStitch stitching...")

        cellstitch_masks = full_stitch(yx_masks, yz_masks, xz_masks, verbose=verbose)

        if output_masks:
            tifffile.imwrite(
                os.path.join(output_path, "cellstitch_masks.tif"), cellstitch_masks
            )

        return cellstitch_masks

    else:
        print(
            "Incompatible stitching method. Supported options are \"iou\" and \"cellstitch\"."
        )
