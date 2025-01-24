import ot
import ot.plot
import numpy as np
import fill_voids
import time

from cellpose import utils as cp_utils
from joblib import Parallel, delayed
from skimage.measure import regionprops


# -------------------------------
# interpolation helper functions
# -------------------------------


def comp_match_plan(pc1, pc2, dist="sqeuclidean"):
    """Compute optimal matching plans between 2 sets of point clouds"""
    # compute cost matrix
    C = ot.dist(pc1, pc2, metric=dist).astype(np.float64)
    C /= C.max()

    # convert point clouds to uniform distributions
    n_pts1, n_pts2 = pc1.shape[0], pc2.shape[0]
    mu1, mu2 = np.ones(n_pts1) / n_pts1, np.ones(n_pts2) / n_pts2

    # compute transport plan
    plan = ot.emd(mu1, mu2, C)

    return plan


def interpolate(pc1, pc2, dist="sqeuclidean", anisotropy=2):
    """
    Calculate interpolated predictions

    Parameters
    ----------
    pc1 : np.ndarray
        Point cloud representing cell boundary in frame 1

    pc2 : np.ndarray
        Point cloud representing cell boundary in frame 2

    anisotropy : int
        Ratio of sampling rate between different axes

    Returns
    -------
    interp_pcs : list
        Smoothed boundary locations along interpolated layers
    """
    alphas = np.linspace(0, 1, anisotropy + 1)[1:-1]
    plan = comp_match_plan(pc1, pc2, dist=dist)
    normalized_plan = plan / plan.sum(
        axis=1, keepdims=1
    )  # normalize so that the row sum is 1

    interp_pcs = []

    for alpha in alphas:
        n_pts = pc1.shape[0]
        avg_pc = np.zeros((n_pts, 2), dtype=int)

        for i in range(n_pts):
            point = pc1[i]
            target_weights = normalized_plan[i]

            weighted_target = np.array(
                [np.sum(target_weights * pc2[:, 0]), np.sum(target_weights * pc2[:, 1])]
            )

            avg_pc[i, :] = point * (1 - alpha) + alpha * weighted_target

        interp_pcs.append(avg_pc)

    return interp_pcs


# -----------------------------------------
# Util functions for interp reconstruction
# -----------------------------------------


def get_lbls(mask):
    """Get unique labels from the predicted masks"""
    return np.unique(mask[mask != 0])


def min_size_filter(res, thld=100):
    """Filter out all masks with area below threshold"""
    assert len(res) == 4, "Invalid Cellpose 4-tuple result"
    preds = res[0]

    for i in range(len(preds)):
        lbls = get_lbls(preds[i])
        for lbl in lbls:
            msk = preds[i] == lbl
            if msk.sum() < thld:
                coords = np.nonzero(msk)
                preds[i][coords] = 0

    res_filtered = (preds, res[1], res[2], res[3])
    return res_filtered


def get_contours(masks):
    """Transfer solid mask predictions to non-overlapping contours w/ distinct integers"""
    masks_new = masks.copy()
    outlines = cp_utils.masks_to_outlines(masks)
    masks_new[~outlines] = 0

    return masks_new


def get_mask_perimeter(masks, lbl, is_contour=False):
    assert lbl in get_lbls(
        masks
    ), "Label {} doesn't in current mask predictions".format(lbl)

    if is_contour:
        p = (masks == lbl).sum()
    else:
        mask_lbl = (masks == lbl).astype(np.uint8)
        p = cp_utils.masks_to_outlines(mask_lbl).sum()

    return p


def calc_vols(pred):
    """
    Calculate volumes of each mask from predictions
    """
    lbls = get_lbls(pred)
    vols = [(pred == lbl).sum() for lbl in lbls]
    return vols


def calc_depth(masks):
    """
    Calculate z-layer depth of predictions
    """
    assert masks.ndim == 3, "Mask predictions must be 3D to calculate depth"

    lbls = get_lbls(masks)
    depths = np.vectorize(lambda lbl: np.diff(np.nonzero(masks == lbl)[0][[0, -1]])[0])(
        lbls
    )

    return depths


def mask_to_coord(mask):
    """Return (n, 2) coordinates from masks"""
    coord = np.asarray(np.nonzero(mask)).T
    return coord


def coord_to_mask(coord, size, lbl):
    """Convert from coordinates to original labeled masks"""
    mask = np.zeros(size)
    mask[tuple(coord.T)] = lbl
    return mask


def contour_to_mask(contour):
    lbl = get_lbls(contour)[0]
    """ Convert contour to solid masks with fill-in labels"""
    binary_contour = contour > 0
    binary_mask = fill_voids.fill(np.asarray(binary_contour))

    mask = np.zeros_like(binary_contour)
    mask[binary_mask] = lbl

    return mask


def connect(coord1, coord2, mask):
    """
    Modify the mask by connecting the two given coordinates.

    Parameters
    ----------
    coord1 : [x1, y1]
        Coordinate of the first pixel.

    coord2 : [x2, y2]
        Coordinate of the second pixel.

    mask : binary np.narray
        Binary mask of the boundary.
    """
    x1, y1 = coord1
    x2, y2 = coord2

    x_offset, y_offset = x2 - x1, y2 - y1

    # skip if the two coordinates are already connected
    if x_offset**2 + y_offset**2 <= 2:
        return

    diag_length = min(abs(x_offset), abs(y_offset))

    # initialize at coord1
    added_x, added_y = x1, y1

    # first, add diagonal pixels
    for i in range(1, diag_length + 1):
        added_x = x1 + i * np.sign(x_offset)
        added_y = y1 + i * np.sign(y_offset)

        mask[added_x, added_y] = 1

        # need to walk vertically
    if added_x == x2 and added_y != y2:
        offset = abs(added_y - y2)
        for i in range(1, offset + 1):
            mask[added_x, added_y + i * np.sign(y_offset)] = 1

            # or, now need to walk horizonally
    if added_y == y2 and added_x != x2:
        offset = abs(added_x - x2)
        for i in range(1, offset + 1):
            mask[added_x + i * np.sign(x_offset), added_y] = 1


def calc_angles(source_point, target_points, eps=1e-20):
    """
    Calculate angle (rad) between source point (source_point) & list of target points(target_points, dim=(n, 2))
    """
    source_points = np.tile(source_point, (target_points.shape[0], 1))
    diff = target_points - source_points
    angles = np.apply_along_axis(
        lambda x: np.arctan2(x[1], x[0] + eps), axis=1, arr=diff
    )

    return angles


def connect_boundary(coords, size, lbl=1):
    """
    Connect interpolation coordinates to generate close-loop mask contours

    Parameters
    ----------
    coords : ndarray, shape (ns,2)
        Boundary coordinates (might be disconnected).

    size: (n1, n2)
        Shape of the final mask.

    lbl: int
        Label of the original mask.

    Returns
    -------
    mask: ndarray, shape (n1, n2)
        Connected boundary mask.

    """
    # Sort boundary labels by angle to mask's mass center
    mass_center = np.round(coords.mean(0)).astype(np.int64)
    angles = calc_angles(mass_center, coords)
    sorted_coords = coords[angles.argsort()]

    mask = coord_to_mask(coords, size, lbl)

    for i, (x, y) in enumerate(sorted_coords[:-1]):
        next_x, next_y = sorted_coords[i + 1]
        connect((x, y), (next_x, next_y), mask)

    connect(tuple(sorted_coords[-1]), tuple(sorted_coords[0]), mask)

    return mask


# -----------------------------
# Core Interpolation functions
# -----------------------------


def process_region(label, label_slice, cell_mask, dist, anisotropy):

    coordinates = label_slice

    source_ct = cell_mask[0].astype(np.uint8)
    target_ct = cell_mask[1].astype(np.uint8)

    shape = source_ct.shape

    source_coord = mask_to_coord(source_ct)
    target_coord = mask_to_coord(target_ct)

    interp_coords = interpolate(
        source_coord, target_coord, dist=dist, anisotropy=anisotropy
    )

    interps = [
        fill_voids.fill(connect_boundary(interp, shape)) * label
        for interp in interp_coords
    ]

    return interps, list(coordinates)


def interp_layers(source_target, dist="sqeuclidean", anisotropy=2):
    """
    Interpolating adjacent z-layers using bounding boxes after contouring
    """

    interp_masks = np.zeros(
        (
            anisotropy + 1,  # num. interpolated layers
            source_target.shape[1],  # y
            source_target.shape[2],  # x
        ),
        dtype=source_target.dtype,
    )

    def _dilation(coords, lims):
        y, x = coords
        ymax, xmax = lims
        dy, dx = np.meshgrid(
            np.arange(y - 2, y + 3), np.arange(x - 2, x + 3), indexing="ij"
        )
        dy, dx = dy.flatten(), dx.flatten()
        mask = np.logical_and(
            np.logical_and(dy >= 0, dx >= 0), np.logical_and(dy < ymax, dx < xmax)
        )
        return dy[mask], dx[mask]

    source_mask = source_target[0]
    target_mask = source_target[1]

    shape = source_mask.shape
    source_contour = get_contours(source_mask)
    target_contour = get_contours(target_mask)

    # Boundary condition: if empty on source / target label
    # align the empty slice w/ mass centers to represent instance endings
    source_dummy = np.zeros_like(source_mask)
    target_dummy = np.zeros_like(target_mask)
    if not np.intersect1d(get_lbls(source_contour), get_lbls(target_contour)).size:
        if (source_contour.sum() == target_contour.sum() == 0) or (
            np.logical_and(source_mask, target_mask).sum() > 0
        ):
            return np.zeros(shape)
        get_mask_center = lambda x: (
            np.round(np.nonzero(x)[0].sum() / x.sum()).astype(np.uint16),
            np.round(np.nonzero(x)[1].sum() / x.sum()).astype(np.uint16),
        )
        for lbl in get_lbls(source_contour):
            yc, xc = _dilation(get_mask_center(source_mask == lbl), source_mask.shape)
            target_dummy[yc, xc] = lbl
        for lbl in get_lbls(target_contour):
            yc, xc = _dilation(get_mask_center(target_mask == lbl), target_mask.shape)
            source_dummy[yc, xc] = lbl
        source_contour += source_dummy
        target_contour += target_dummy

    # Determine which labels are intersecting
    joint_lbls = np.intersect1d(get_lbls(source_contour), get_lbls(target_contour))

    # Reassign source_target
    source_target = np.stack((source_contour, target_contour))

    # Filter source_target for labels that are found to intersect
    source_target = np.where(np.isin(source_target, joint_lbls), source_target, 0)

    # Use regionprops to set up bounding boxes based on the newly generated contours
    regions = regionprops(source_target)

    # Parallel process bounding boxes
    results = Parallel(n_jobs=-1)(
        delayed(process_region)(
            region.label,
            region.slice,
            region.image,
            dist,
            anisotropy,
        )
        for region in regions
    )

    masks, coordinates = zip(*results)

    # Write each interpolated bounding box into the empty interp_masks array
    for i, mask in enumerate(masks):
        z_min, z_max = 1, anisotropy
        y_min, y_max = coordinates[i][1].start, coordinates[i][1].stop
        x_min, x_max = coordinates[i][2].start, coordinates[i][2].stop

        interp_masks[z_min:z_max, y_min:y_max, x_min:x_max] = np.where(
            interp_masks[z_min:z_max, y_min:y_max, x_min:x_max] == 0,
            # interp_masks[z_min:z_max, y_min:y_max, x_min:x_max] < mask,  # Should behave similar to picking max value
            mask,
            interp_masks[z_min:z_max, y_min:y_max, x_min:x_max],
        )

    interp_masks[0] = source_mask
    interp_masks[-1] = target_mask

    return interp_masks


def full_interpolate(masks, anisotropy=2, dist="sqeuclidean", verbose=False):
    """
    Interpolating between all adjacent z-layers

    Parameters
    ----------
    masks : np.ndarray
        layers of 2D predictions
        (dim: (Depth, H, W))

    anisotropy : int
        Ratio of sampling rate between xy-axes & z-axis

    Returns
    -------|
    interp_masks : np.ndarray
        interpolated masks
        (dim: (Depth * anisotropy - (anisotropy-1), H, W))

    """
    if masks.max() < 256:
        masks = masks.astype("uint8")
    elif masks.max() < 65536:
        masks = masks.astype("uint16")

    dtype = masks.dtype

    interp_masks = np.zeros(
        (
            len(masks) + (len(masks) - 1) * (anisotropy - 1),
            masks.shape[1],
            masks.shape[2],
        ),
        dtype=dtype,
    )

    idx = 0
    for i in range(masks.shape[0] - 1):
        if verbose:
            time_start = time.time()
            print("Interpolating layer {} & {}...".format(i, i + 1))
        source_target = masks[i : i + 2].copy()
        interps = interp_layers(
            source_target, dist=dist, anisotropy=anisotropy
        )
        interp_masks[idx : idx + anisotropy + 1] = interps
        idx += anisotropy
        if verbose:
            print("Time to interpolate: ", time.time() - time_start)

    return interp_masks
