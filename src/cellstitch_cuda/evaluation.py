import gc

from cellpose import metrics as cp_metrics
from cellpose import utils as cp_utils
from cupyx.scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError

from skimage.measure import marching_cubes, mesh_surface_area
from skimage.metrics import variation_of_information

from cellstitch_cuda.alignment import *


# --------------------
# Helper functions
# --------------------


def get_num_cells(masks):
    return len(cp.unique(masks)[1:])


def get_avg_vol(masks):
    total_vol = (masks != 0).sum()
    n = get_num_cells(masks)
    return total_vol / n


def sample_indices(masks, n=50):
    """Randomly sample instances from the list of input 3D masks"""
    indices = []
    for mask in masks:
        n = min(get_num_cells(mask), n)
        assert n > 0, "Empty masks"
        indices.append(cp.random.choice(cp.unique(mask)[1:], n))
    return cp.array(indices)


def match_lbls(source_mask, target_mask, sc_lbl):
    """
    Find matched label between source mask & target mask with largest IoU area
    Return the best-matched target label
    """
    coords = cp.nonzero(source_mask == sc_lbl)
    cand_lbls, counts = cp.unique(
        target_mask[coords], return_counts=True
    )  # Candidate labels

    if len(cand_lbls) == 0 or (len(cand_lbls) == 1 and cand_lbls[0] == 0):
        tg_lbl = -1
    else:
        # First check whether argmax matched lbl has IoU exceeding threshold
        if cand_lbls[0] == 0:  # Ignore label 0
            cand_lbls = cand_lbls[1:]
            counts = counts[1:]
        tg_lbl = cand_lbls[counts.argmax()]
    return tg_lbl


def _compute_convex_hull(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.

    # Reference:
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30

    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    """
    assert (
        cp.array(image.shape) <= cp.iinfo(cp.int16).max
    ).all(), (
        f"This function assumes your image is smaller than {2**15} in each dimension"
    )

    points = cp.argwhere(image).astype(cp.int16)
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = cp.indices(image.shape[1:], cp.int16)
    idx_2d = cp.moveaxis(idx_2d, 0, -1)

    idx_3d = cp.zeros((*image.shape[1:], image.ndim), cp.int16)
    idx_3d[:, :, 1:] = idx_2d

    mask = cp.zeros_like(image, dtype=bool)
    for z in range(len(image)):
        idx_3d[:, :, 0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1

    return mask


def _calc_surface_area(mask):
    verts, faces, _, _ = marching_cubes(mask)
    return mesh_surface_area(verts, faces)


def _subsample_mask(sc_mask, tg_mask):
    """
    Subsample the minimum cuboid surrounding the nonzero elements of the source & target binary masks
    """
    d, h, w = sc_mask.shape

    # Convert to integer if input is binary
    union = cp.logical_or(sc_mask, tg_mask).astype(cp.uint8)
    coords = cp.argwhere(union).astype(cp.uint16)

    dl, hl, wl = coords.min(0)
    dr, hr, wr = coords.max(0)

    # Maintain 5-pixel wide background outside mask volume
    bw = 5
    dl, hl, wl = max(0, dl - bw), max(0, hl - bw), max(0, wl - bw)
    dr, hr, wr = min(d, dr + bw + 1), min(h, hr + bw + 1), min(w, wr + bw + 1)
    return sc_mask[dl:dr, hl:hr, wl:wr], tg_mask[dl:dr, hl:hr, wl:wr]


# --------------------
# Evaluation metrics
# --------------------


def average_precision(masks_true, masks_pred, threshold):
    iou = cp_metrics._intersection_over_union(masks_true, masks_pred)
    lbls_true, lbls_pred = (
        cp.unique(masks_true)[1:,].tolist(),
        cp.unique(masks_pred)[1:,].tolist(),
    )

    tp = 0
    for lbl_true in lbls_true:
        for lbl_pred in lbls_pred:
            if iou[lbl_true][lbl_pred] >= threshold:
                tp += 2  # count the number of matched cells

    fp = len(lbls_pred) - tp / 2
    fn = len(lbls_true) - tp / 2

    ap = tp / (tp + fp + fn)

    return [ap, tp, fp, fn]  # return as list for easy convert to dataframe


def voi(masks_true, masks_pred):
    """
    Calculate Variance-of-information (VI) to measure the amount of over-/under-segmentation
    """
    h_over, h_under = variation_of_information(masks_true, masks_pred)
    return h_over, h_under


def avg_symmetric_surf_dist(mask, pred, mask_lbls, pred_lbls):
    """
    Calculate Average Symmetric Surface Distance for each paired mask label & prediction
     (Randomly sample 100 instances: presampled as `mask_lbls` & `pred_lbls`)
    """

    # Helper functions
    def _calc_assd(mask, pred):
        # Take outline
        mask_outline = cp_utils.masks_to_outlines(mask)
        pred_outline = cp_utils.masks_to_outlines(pred)

        # Calculate ASSD for each outline pixel
        m1 = cp.ones_like(mask, dtype=cp.uint8)
        m1[pred_outline] = 0
        m2 = cp.ones_like(pred, dtype=cp.uint8)
        m2[mask_outline] = 0

        dist1 = ndi.distance_transform_edt(m1)[cp.nonzero(mask_outline)]
        dist2 = ndi.distance_transform_edt(m2)[cp.nonzero(pred_outline)]
        assd = cp.concatenate([dist1, dist2]).mean()

        del m1, m2, mask_outline, pred_outline
        gc.collect()

        return assd

    assd = []
    for sc_lbl, tg_lbl in zip(mask_lbls, pred_lbls):
        if tg_lbl != -1:
            mask_bin, pred_bin = _subsample_mask(mask == sc_lbl, pred == tg_lbl)
            assd.append(_calc_assd(mask_bin, pred_bin))

    return cp.mean(assd) if len(assd) > 0 else 0


def compactness_convexity_ae(mask, pred, mask_lbls, pred_lbls, eps=1e-10):
    """
    Calculate Absolute Error of the following metrics between mapped (ground-truth, predicted) masks:
     - (1). Compactness
     - (2). Convexity
     (Randomly sample 100 instances: presampled as `mask_lbls` & `pred_lbls`)
    """
    comp_abs_errors = cp.zeros(len(mask_lbls))  # compactness errors
    conv_abs_errors = cp.zeros(len(mask_lbls))  # convexity errors

    for i, (mask_lbl, pred_lbl) in enumerate(zip(mask_lbls, pred_lbls)):
        if pred_lbl == -1:
            continue

        mask_bin, pred_bin = _subsample_mask(mask == mask_lbl, pred == pred_lbl)

        try:
            # Compactness
            vm, am = mask_bin.sum(), _calc_surface_area(mask_bin)
            vp, ap = pred_bin.sum(), _calc_surface_area(pred_bin)
            cm = 36 * cp.pi * vm**2 / (am**3 + eps)
            cp = 36 * cp.pi * vp**2 / (ap**3 + eps)
            comp_abs_errors[i] = cp.abs(cp - cm) / cm

            # Convexity
            mask_bin_ch = _compute_convex_hull(mask_bin)
            pred_bin_ch = _compute_convex_hull(pred_bin)

            am_ch = _calc_surface_area(mask_bin_ch)
            ap_ch = _calc_surface_area(pred_bin_ch)
            cm = am_ch / (am + eps)
            cp = ap_ch / (ap + eps)
            conv_abs_errors[i] = cp.abs(cp - cm) / cm

            del mask_bin_ch, pred_bin_ch
            gc.collect()

        except (RuntimeError, QhullError) as errs:
            print("Can't compute 3D convex hull / surface area given the mask, passed")
            pass

        del mask_bin, pred_bin
        gc.collect()

    return {"Compactness": comp_abs_errors.mean(), "Convexity": conv_abs_errors.mean()}
