import ot
import cupyx
from skimage import color
import matplotlib.pyplot as plt
import numpy as np

from cellstitch_cuda.frame import *
from cellstitch_cuda.interpolate import get_mask_center_cupy
import time


class FramePair:
    def __init__(self, mask0, mask1, max_lbl=0):
        self.frame0 = Frame(mask0)
        self.frame1 = Frame(mask1)

        # store the max labels for stitching
        max_lbl_default = max(
            self.frame0.get_lbls().max(), self.frame1.get_lbls().max()
        )

        self.max_lbl = max(max_lbl, max_lbl_default)

    def display(self):
        """
        Display frame0 and frame1 next to each other, with consistent colorings.
        """

        num_lbls = len(cp.union1d(self.frame0.get_lbls(), self.frame1.get_lbls()))

        colors = cp.random.random((num_lbls, 3))

        frames = cp.array([self.frame0.mask, self.frame1.mask])
        rgb = color.label2rgb(frames, colors=colors, bg_label=0)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(rgb[0])
        axes[1].imshow(rgb[1])

        plt.tight_layout()
        plt.show()

    def get_plan(self, C):
        """
        Compute the transport plan between the two frames, given the cost matrix between the cells.
        """
        mask0 = cp.asarray(self.frame0.mask)
        mask1 = cp.asarray(self.frame1.mask)

        unique_labels0, counts0 = cp.unique(mask0, return_counts=True)
        unique_labels1, counts1 = cp.unique(mask1, return_counts=True)
        sizes0 = cp.asarray(counts0)
        sizes1 = cp.asarray(counts1)

        # convert to distribution to compute transport plan
        dist0 = (sizes0 / cp.sum(sizes0)).get()
        dist1 = (sizes1 / cp.sum(sizes1)).get()

        cp._default_memory_pool.free_all_blocks()

        # compute transportation plan
        plan = ot.emd(dist0, dist1, C)

        return plan

    def get_cost_matrix(self, overlap, lbls0, lbls1):
        """
        Return the cost matrix between cells in the two frame defined by IoU.
        """

        sizes0 = np.sum(overlap, axis=1)
        sizes1 = np.sum(overlap, axis=0)

        lbls0 = lbls0.get()
        lbls1 = lbls1.get()

        # Create a meshgrid for vectorized operations
        lbl0_indices, lbl1_indices = np.meshgrid(lbls0, lbls1, indexing="ij")

        overlap_sizes = overlap[lbl0_indices, lbl1_indices]
        cp._default_memory_pool.free_all_blocks()
        scaling_factors = overlap_sizes / (
            sizes0[lbl0_indices] + sizes1[lbl1_indices] - overlap_sizes + 1e-6
        )

        C = (1 - scaling_factors)

        return C

    def stitch(
        self, yz_not_stitched, xz_not_stitched, p_stitching_votes=0.75, radii_limit=4, verbose=False
    ):
        """Stitch frame1 using frame 0."""

        time_start = time.time()

        lbls0 = self.frame0.get_lbls()  # Get unique label IDs
        lbls1 = self.frame1.get_lbls()  # Get unique label IDs

        # get sizes
        overlap = _label_overlap(self.frame0.mask, self.frame1.mask)

        # compute matching
        C = self.get_cost_matrix(overlap, lbls0, lbls1)

        cp._default_memory_pool.free_all_blocks()

        plan = self.get_plan(C)

        cp._default_memory_pool.free_all_blocks()

        # get a soft matching from plan
        n, m = plan.shape
        soft_matching = cp.zeros((n, m))

        # Vectorized computation
        matched_indices = plan.argmax(axis=1)  # Use minimum cost instead of max plan probability
        soft_matching[cp.arange(n), matched_indices] = 1

        mask1 = cp.asarray(self.frame1.mask)

        stitched_mask1 = cp.zeros_like(mask1)
        for lbl1_index in range(1, m):
            # find the cell with the lowest cost (i.e. lowest scaled distance)
            matching_filter = soft_matching[:, lbl1_index]
            filtered_C = np.where(
                (matching_filter == 0).get(), np.inf, C[:, lbl1_index]
            )  # ignore the non-matched cells

            lbl0_index = np.argmin(
                filtered_C
            )  # this is the cell0 we will attempt to relabel cell1 with

            lbl0, lbl1 = int(lbls0[lbl0_index]), int(lbls1[lbl1_index])

            filter_1 = mask1 == lbl1

            # In case the minimum cost is 1 (if unchecked, causes stitching of labels that should not be connected)
            if lbl0 != 0 and filtered_C[lbl0_index] == 1:
                filter_0 = cp.asarray(self.frame0.mask == lbl0)

                # Check overlap first
                if not cp.any(filter_0 & filter_1):
                    # Get label centers
                    center_0 = get_mask_center_cupy(filter_0).astype(int)
                    center_1 = get_mask_center_cupy(filter_1).astype(int)

                    # Calculate distance between centers
                    dist = center_0 - center_1
                    dist = cp.linalg.norm(dist).astype(int).item()

                    area0 = cp.sum(filter_0)
                    area1 = cp.sum(filter_1)
                    radius0 = cp.sqrt(area0 / cp.pi).astype(int)
                    radius1 = cp.sqrt(area1 / cp.pi).astype(int)
                    cell_radius = max(radius0.item(), radius1.item())  # Largest radius

                    if dist > radii_limit * cell_radius:
                        # If the label is too far away, do not stitch
                        lbl0 = 0

            n_not_stitch_pixel = (
                yz_not_stitched[filter_1].sum() / 2
                + xz_not_stitched[filter_1].sum() / 2
            )
            stitch_cell = (
                n_not_stitch_pixel <= (1 - p_stitching_votes) * filter_1.sum()
            )

            if lbl0 != 0 and stitch_cell:  # only reassign if they overlap
                stitched_mask1[filter_1] = lbl0
            else:
                self.max_lbl += 1
                stitched_mask1[filter_1] = self.max_lbl  # create a new label

        if verbose:
            print("Time to stitch: ", time.time() - time_start)

        self.frame1 = Frame(stitched_mask1)


def _label_overlap(x, y):
    """Fast function to get pixel overlaps between masks in x and y.

    Args:
        x (np.ndarray, int): Where 0=NO masks; 1,2... are mask labels.
        y (np.ndarray, int): Where 0=NO masks; 1,2... are mask labels.

    Returns:
        overlap (np.ndarray, int): Matrix of pixel overlaps of size [x.max()+1, y.max()+1].

    Adapted from CellPose: https://github.com/MouseLand/cellpose
        https://doi.org/10.1038/s41592-020-01018-x: Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).
        Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106.
        Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
    """
    # put label arrays into standard form then flatten them
    x = x.ravel().get()
    y = y.ravel().get()
    xmax = int(x.max())
    ymax = int(y.max())

    overlap = np.zeros((1 + xmax, 1 + ymax), dtype=np.uint)

    np.add.at(overlap, (x, y), 1)

    return overlap
