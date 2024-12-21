import tifffile
import os
from instanseg import InstanSeg
from cellpose.metrics import _label_overlap
from cellpose.utils import stitch3D
from cellstitch_cuda.postprocessing_cupy import fill_holes_and_remove_small_masks, filter_nuclei_cells

from cellstitch_cuda.alignment import *
from cellstitch_cuda.preprocessing_cupy import *


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
    masks_cp = cp.asarray(masks)
    lbls = cp.unique(masks_cp)[1:]

    # get a list of labels that need to be corrected
    layers_lbls = {}

    for lbl in lbls:
        existing_layers = cp.any(masks_cp == lbl, axis=(1, 2))
        depth = cp.sum(existing_layers)

        if depth == 1:
            z = int(cp.where(existing_layers)[0][0])
            layers_lbls.setdefault(z, []).append(lbl)

    for z, lbls in layers_lbls.items():
        relabel_layer(masks_cp, z, lbls)
        cp._default_memory_pool.free_all_blocks()

    masks = masks_cp.get()
    cp._default_memory_pool.free_all_blocks()

    return masks


def full_stitch(xy_masks_prior, yz_masks, xz_masks, nuclei=None, filter: bool = True, n_jobs=-1, verbose=False):
    """Stitch masks in-place

    Stitches masks from top to bottom.

    Args:
        xy_masks_prior: numpy.ndarray with XY masks
        yz_masks: numpy.ndarray with YZ masks
        xz_masks: numpy.ndarray with XZ masks
        nuclei: numpy.ndarray with XY masks of nuclei
        filter: Use CellPose-based fill_holes_and_remove_small_masks() function. Default True
        n_jobs: Number of threads used. Set n_jobs to 1 for debugging parallel processing tasks. Default -1
        verbose: Verbosity. Default False
    """

    xy_masks = np.array(xy_masks_prior, dtype="uint32")
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

    if filter:
        cp._default_memory_pool.free_all_blocks()
        time_start = time.time()
        xy_masks = fill_holes_and_remove_small_masks(xy_masks, n_jobs=n_jobs)
        if verbose:
            print(
                "Time to fill holes and remove small masks: ", time.time() - time_start
            )

    cp._default_memory_pool.free_all_blocks()

    if not nuclei is None:
        time_start = time.time()
        xy_masks = filter_nuclei_cells(xy_masks, nuclei)
        cp._default_memory_pool.free_all_blocks()
        if verbose:
            print(
                "Time to filter cells with nuclei: ", time.time() - time_start
            )

    time_start = time.time()

    xy_masks = overseg_correction(xy_masks)

    if verbose:
        print("Time to correct oversegmentation: ", time.time() - time_start)

    return xy_masks


def cellstitch_cuda(
    img,
    output_masks: bool = False,
    output_path=None,
    stitch_method: str = "cellstitch",
    seg_mode: str = "nuclei_cells",
    normalise: bool = True,
    pixel_size=None,
    z_step=None,
    bleach_correct: bool = True,
    filtering: bool = True,
    n_jobs: int = -1,
    verbose: bool = False,
):
    """All-in-one function to segment and stitch 2D labels

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
        normalise: Whether to perform normalisation prior to InstanSeg-based segmentation. It is performed over the
            entire stack (per channel) to save time, but it can still take a while. Default True
        pixel_size: XY pixel size in microns per pixel. When set to None, will be read from img metadata if possible.
            Default None
        z_step: Z pixel size (z step) in microns per step. When set to None, will be read from img metadata if possible.
            Default None
        bleach_correct: Whether histogram-based signal degradation correction should be applied to img.
            Default True
        filtering: Whether the fill_holes_and_remove_small_masks function should be executed. With larger datasets, this
            has the tendency to massively slow down the postprocessing.
            Default True
        n_jobs: Set the number of threads to be used in parallel processing tasks. Use 1 for debugging. Generally, best
            left at the default value.
            Default -1
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
            tags = tif.pages[0].tags
    elif not isinstance(img, np.ndarray):
        print("img must either be a path to an existing image, or a numpy ndarray.")
        sys.exit(1)

    # Check image dimensions
    if img.ndim != 4:
        print("Expected a 4D image (ZCYX), while the img dimensions are ", img.ndim)
        sys.exit(1)

    # Set pixel sizes
    if pixel_size is None:
        try:
            pixel_size = 1 / (tags["XResolution"].value[0] / tags["XResolution"].value[1])
            if verbose:
                print("Pixel size:", pixel_size)
        except:
            print(
                "No XResolution found in image metadata. The output might not be fully reliable."
            )
    if z_step is None:
        try:
            img_descr = tags["ImageDescription"].value.split()
            z_step = float(
                [s for s in img_descr if "spacing" in s][0].split("=")[-1]
            )  # At least it's pretty fast
            if verbose:
                print("Z step:", z_step)
        except:
            try:
                img_descr = tags["IJMetadata"].value["Info"].split()
                z_step = float(
                    [s for s in img_descr if "spacing" in s][0].split("=")[-1]
                )  # It's even funnier the second time
                if verbose:
                    print("Z step:", z_step)
            except:
                print(
                    "No spacing (Z step) found in image metadata. The output might not be fully reliable."
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
        img = histogram_correct(img)
        cp._default_memory_pool.free_all_blocks()
        if verbose:
            print("Finished bleach correction.")

    img = img.transpose(1, 2, 3, 0)  # ZCYX -> CYXZ

    # Segment over Z-axis
    if verbose:
        print("Segmenting YX planes (Z-axis).")

    yx_masks = segmentation(img, model, seg_mode, normalise=normalise, xy=True)
    if seg_mode == "nuclei_cells":
        nuclei = yx_masks[1].transpose(
            2, 0, 1
        )  # YXZ -> ZYX
        yx_masks = yx_masks[0].transpose(
            2, 0, 1
        )  # YXZ -> ZYX

        if output_masks:
            tifffile.imwrite(os.path.join(output_path, "nuclei_masks.tif"), nuclei)
    else:
        yx_masks = yx_masks.transpose(
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

        if seg_mode == "nuclei_cells":
            iou_masks = filter_nuclei_cells(iou_masks, nuclei)

        if output_masks:
            tifffile.imwrite(os.path.join(output_path, "iou_masks.tif"), iou_masks)

        return iou_masks

    elif stitch_method == "cellstitch":

        # Segment over X-axis
        if verbose:
            print("Segmenting YZ planes (X-axis).")
        transposed_img = img.transpose(0, 1, 3, 2)  # CYXZ -> CYZX
        transposed_img, padding = upscale_pad_img(
            transposed_img, pixel_size, z_step
        )  # Preprocess YZ planes
        cp._default_memory_pool.free_all_blocks()
        yz_masks = segmentation(transposed_img, model, seg_mode, normalise=normalise)
        del transposed_img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
        yz_masks = crop_downscale_mask(
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
        transposed_img, padding = upscale_pad_img(
            transposed_img, pixel_size, z_step
        )  # Preprocess XZ planes
        cp._default_memory_pool.free_all_blocks()
        xz_masks = segmentation(transposed_img, model, seg_mode, normalise=normalise)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
        xz_masks = crop_downscale_mask(
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

        if seg_mode == "nuclei_cells":
            cellstitch_masks = full_stitch(
                yx_masks, yz_masks, xz_masks, nuclei, filter=filtering, verbose=verbose
            )
        else:
            cellstitch_masks = full_stitch(
                yx_masks, yz_masks, xz_masks, filter=filtering, n_jobs=n_jobs, verbose=verbose
            )

        if output_masks:
            tifffile.imwrite(
                os.path.join(output_path, "cellstitch_masks.tif"), cellstitch_masks
            )

        return cellstitch_masks

    else:
        print(
            'Incompatible stitching method. Supported options are "iou" and "cellstitch".'
        )
        sys.exit(1)
