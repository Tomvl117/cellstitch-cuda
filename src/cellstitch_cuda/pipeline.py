import tifffile
import os
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

from cellstitch_cuda.alignment import _label_overlap
from instanseg import InstanSeg
from cellstitch_cuda.postprocessing_cupy import (
    fill_holes_and_remove_small_masks,
    filter_nuclei_cells,
)
from cellstitch_cuda.interpolate import full_interpolate

from cellstitch_cuda.alignment import *
from cellstitch_cuda.preprocessing_cupy import *


def split_label(region, limit, max_lbl):

    image = region.image

    n = int(round(region.num_pixels / limit))

    a = int(round(image.shape[0]/n))
    b = image.shape[1]
    c = image.shape[2]

    distance = ndi.distance_transform_edt(image)
    max_coords = peak_local_max(distance, labels=image, footprint=np.ones((a, b, c)))
    local_maxima = np.zeros_like(image, dtype=bool)
    local_maxima[tuple(max_coords.T)] = True
    markers = ndi.label(local_maxima)[0]

    labels = watershed(-distance, markers, mask=image)

    mask = np.where(labels > 1, labels + max_lbl - 1, region.label) * image

    return mask


def relabel_layer(masks, z, lbls):
    """
    Relabel the label in LBLS in layer Z of MASKS.
    """
    layer = masks[z]
    if z != 0:
        reference_layer = masks[z - 1]
    else:
        reference_layer = masks[z + 1]

    overlap = _label_overlap(reference_layer, layer)

    for lbl in lbls:
        lbl0 = np.argmax(overlap[:, lbl])
        layer[layer == lbl] = lbl0


def correction(masks):
    """Correct over- and undersegmentation

    This function first attempts to stitch any masks that are only 1 plane thick, then goes over the labels again to
    ensure that masks are not too large.

    The assumption for undersegmentation correction is that, on average, the labels are of the correct size. For labels
    that have a volume > 3 standard deviations away from the mean, they will be split into appropriate sizes.
    """

    # get a list of labels that need to be corrected
    # layers_lbls = {}
    #
    # regions = regionprops(masks)
    #
    # for region in regions:
    #     if region.bbox[3] - region.bbox[0] == 1:
    #         layers_lbls.setdefault(region.bbox[0], []).append(region.label)
    #
    # for z, lbls in layers_lbls.items():
    #     relabel_layer(masks, z, lbls)
    #     cp._default_memory_pool.free_all_blocks()

    regions = regionprops(masks)

    area = []
    for region in regions:
        area.append(region.num_pixels)
    area = np.array(area)

    mean_area = np.mean(area)
    std_dev = np.std(area, ddof=1)

    size_limit = int(round(mean_area + 3 * std_dev))

    for region in regions:
        if region.num_pixels > size_limit:
            mask = split_label(region, int(round(mean_area)), masks.max())
            masks[region.slice] = np.where(masks[region.slice] == region.label, mask, masks[region.slice])

    return masks


def full_stitch(
    xy_masks_prior,
    yz_masks,
    xz_masks,
    nuclei=None,
    filter: bool = True,
    n_jobs=-1,
    verbose=False,
):
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

            cp._default_memory_pool.free_all_blocks()

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

            cp._default_memory_pool.free_all_blocks()

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
            print("Time to filter cells with nuclei: ", time.time() - time_start)

    time_start = time.time()

    xy_masks = correction(xy_masks)

    if verbose:
        print("Time to correct over- and undersegmentation: ", time.time() - time_start)

    return xy_masks


def cellstitch_cuda(
    img,
    output_masks: bool = False,
    output_path=None,
    seg_mode: str = "nuclei_cells",
    pixel_size=None,
    z_step=None,
    bleach_correct: bool = True,
    filtering: bool = True,
    interpolation: bool = False,
    n_jobs: int = -1,
    verbose: bool = False,
):
    """All-in-one function to segment and stitch 2D labels

    Full stitching pipeline, which does the following:
        1. Histogram-based signal degradation correction
        2. Segmentation over the Z axis using InstanSeg
        3. Stitching of 2D planes into 3D labels through CellStitch's orthogonal labeling, which leverages Optimal
            Transport to create robust masks.

    Args:
        img: Either a path pointing to an existing image, or a numpy.ndarray. Must be 4D (ZCYX).
        output_masks: True to write all masks to the output path, or False to only return the final stitched mask.
            Default False
        output_path: Set to None to write to the input file location (if provided). Ignored of `output_masks` is False.
            N.B.: If `output_masks` is True, while no path has been provided (e.g., by loading a numpy.ndarray
            directly), the output masks will be written to the folder where the script is run from.
            Default None
        seg_mode: Instanseg segmentation mode: "nuclei" to only return nuclear masks, "cells" to return all the cell
            masks (including those without nuclei), or "nuclei_cells", which returns only cells with detected nuclei.
            Default "nuclei_cells"
        pixel_size: XY pixel size in microns per pixel. When set to None, will be read from img metadata if possible.
            Default None
        z_step: Z pixel size (z step) in microns per step. When set to None, will be read from img metadata if possible.
            Default None
        bleach_correct: Whether histogram-based signal degradation correction should be applied to `img`.
            Default True
        filtering: Whether the optimized `fill_holes_and_remove_small_masks` function should be executed.
            Default True
        interpolation: If set to True, the function returns a tuple of the array of stitched masks and an array with
            interpolated volumetric masks. CellStitch provides an interpolation method to turn anisotropic masks into
            pseudo-isotropic masks. The algorithm, adapted from the original codebase, has been completely rewritten for
            efficient parallel processing. Outputs a separate mask in the output folder if `output_masks` = True.
            Default False
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
            pixel_size = 1 / (
                tags["XResolution"].value[0] / tags["XResolution"].value[1]
            )
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

    yx_masks = segmentation(img, model, seg_mode, xy=True)
    if seg_mode == "nuclei_cells":
        nuclei = yx_masks[1].transpose(2, 0, 1)  # YXZ -> ZYX
        yx_masks = yx_masks[0].transpose(2, 0, 1)  # YXZ -> ZYX

        if output_masks:
            tifffile.imwrite(os.path.join(output_path, "nuclei_masks.tif"), nuclei)
    else:
        yx_masks = yx_masks.transpose(2, 0, 1)  # YXZ -> ZYX

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

    if output_masks:
        tifffile.imwrite(os.path.join(output_path, "yx_masks.tif"), yx_masks)

    # Segment over X-axis
    if verbose:
        print("Segmenting YZ planes (X-axis).")
    transposed_img = img.transpose(0, 1, 3, 2)  # CYXZ -> CYZX
    transposed_img = upscale_img(
        transposed_img, pixel_size, z_step
    )  # Preprocess YZ planes
    cp._default_memory_pool.free_all_blocks()
    yz_masks = segmentation(transposed_img, model, seg_mode)
    del transposed_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
    yz_masks = downscale_mask(yz_masks, pixel_size, z_step).transpose(
        1, 0, 2
    )  # YZX -> ZYX
    cp._default_memory_pool.free_all_blocks()
    if output_masks:
        tifffile.imwrite(os.path.join(output_path, "yz_masks.tif"), yz_masks)

    # Segment over Y-axis
    if verbose:
        print("Segmenting XZ planes (Y-axis).")
    transposed_img = img.transpose(0, 2, 3, 1)  # CYXZ -> CXZY
    del img
    transposed_img = upscale_img(
        transposed_img, pixel_size, z_step
    )  # Preprocess XZ planes
    cp._default_memory_pool.free_all_blocks()
    xz_masks = segmentation(transposed_img, model, seg_mode)
    del transposed_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache
    xz_masks = downscale_mask(xz_masks, pixel_size, z_step).transpose(
        1, 2, 0
    )  # XZY -> ZYX
    cp._default_memory_pool.free_all_blocks()
    if output_masks:
        tifffile.imwrite(os.path.join(output_path, "xz_masks.tif"), xz_masks)

    # Memory cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

    if verbose:
        print("Running CellStitch stitching...")

    if seg_mode == "nuclei_cells":
        cellstitch_masks = full_stitch(
            yx_masks,
            yz_masks,
            xz_masks,
            nuclei,
            filter=filtering,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    else:
        cellstitch_masks = full_stitch(
            yx_masks,
            yz_masks,
            xz_masks,
            filter=filtering,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    cp._default_memory_pool.free_all_blocks()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

    if output_masks:
        tifffile.imwrite(
            os.path.join(output_path, "cellstitch_masks.tif"), cellstitch_masks
        )

    if interpolation:
        if not pixel_size or not z_step:
            print("Cannot determine anisotropy for interpolation. Defaulting to anisotropy = 2. Result might not be "
                  "accurate.")
            anisotropy = 2
        else:
            anisotropy = int(round(z_step/pixel_size))
        if anisotropy > 1:
            if verbose:
                time_start = time.time()
            cellstitch_masks_interp = full_interpolate(
                cellstitch_masks,
                anisotropy=anisotropy,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            if verbose:
                print("Total time to interpolate:", time.time() - time_start)
            if output_masks:
                tifffile.imwrite(
                    os.path.join(output_path, "cellstitch_masks_interpolated.tif"), cellstitch_masks_interp
                )
            return cellstitch_masks, cellstitch_masks_interp
        else:
            print("Image data is already isotropic. Skipping interpolation.")

    return cellstitch_masks
