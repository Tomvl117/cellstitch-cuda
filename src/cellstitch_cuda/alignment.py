import ot
import cupyx
from skimage import color
import matplotlib.pyplot as plt

from cellstitch_cuda.frame import *
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
        dist0 = sizes0 / cp.sum(sizes0)
        dist1 = sizes1 / cp.sum(sizes1)

        # compute transportation plan
        plan = ot.emd(dist0, dist1, C)

        return plan, mask1

    def get_cost_matrix(self, overlap, lbls0, lbls1):
        """
        Return the cost matrix between cells in the two frame defined by IoU.
        """

        sizes0 = cp.sum(overlap, axis=1)
        sizes1 = cp.sum(overlap, axis=0)

        # Create a meshgrid for vectorized operations
        lbl0_indices, lbl1_indices = cp.meshgrid(lbls0, lbls1, indexing="ij")

        overlap_sizes = overlap[lbl0_indices, lbl1_indices]
        scaling_factors = overlap_sizes / (
            sizes0[lbl0_indices] + sizes1[lbl1_indices] - overlap_sizes
        )

        C = 1 - scaling_factors

        return C

    def stitch(
        self, yz_not_stitched, xz_not_stitched, p_stitching_votes=0.75, verbose=False
    ):
        """Stitch frame1 using frame 0."""

        time_start = time.time()

        lbls0 = self.frame0.get_lbls()  # Get unique label IDs
        lbls1 = self.frame1.get_lbls()  # Get unique label IDs

        # get sizes
        overlap = _label_overlap_cupy(self.frame0.mask, self.frame1.mask)

        # compute matching
        C = self.get_cost_matrix(overlap, lbls0, lbls1)
        plan, mask1 = self.get_plan(C)

        # get a soft matching from plan
        n, m = plan.shape
        soft_matching = cp.zeros((n, m))

        # Vectorized computation
        matched_indices = plan.argmax(axis=1)
        soft_matching[cp.arange(n), matched_indices] = 1

        stitched_mask1 = cp.zeros(mask1.shape)
        for lbl1_index in range(1, m):
            # find the cell with the lowest cost (i.e. lowest scaled distance)
            matching_filter = soft_matching[:, lbl1_index]
            filtered_C = cp.where(
                matching_filter == 0, cp.Inf, C[:, lbl1_index]
            )  # ignore the non-matched cells

            lbl0_index = cp.argmin(
                filtered_C
            )  # this is the cell0 we will attempt to relabel cell1 with

            lbl0, lbl1 = int(lbls0[lbl0_index]), int(lbls1[lbl1_index])

            n_not_stitch_pixel = (
                yz_not_stitched[mask1 == lbl1].sum() / 2
                + xz_not_stitched[mask1 == lbl1].sum() / 2
            )
            stitch_cell = (
                n_not_stitch_pixel <= (1 - p_stitching_votes) * (mask1 == lbl1).sum()
            )

            if lbl0 != 0 and stitch_cell:  # only reassign if they overlap
                stitched_mask1[mask1 == lbl1] = lbl0
            else:
                self.max_lbl += 1
                stitched_mask1[mask1 == lbl1] = self.max_lbl  # create a new label

        if verbose:
            print("Time to stitch: ", time.time() - time_start)

        self.frame1 = Frame(stitched_mask1)


def _label_overlap_cupy(x, y):
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
    x = cp.asarray(x.ravel())
    y = cp.asarray(y.ravel())
    xmax = int(x.max())
    ymax = int(y.max())

    # preallocate a "contact map" matrix
    overlap = cp.zeros((1 + xmax, 1 + ymax), dtype=cp.uint)

    cupyx.scatter_add(overlap, (x, y), 1)

    return overlap
