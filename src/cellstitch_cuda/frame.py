import cupy as cp


class Frame:
    def __init__(self, mask):
        """A container to the mask with useful features."""
        self.mask = cp.asarray(mask, dtype=cp.uint32)

    def get_lbls(self):
        return cp.unique(self.mask)

    def is_empty(self):
        """
        return if the frame is empty.
        """
        return len(self.get_lbls()) == 1

    def get_locations(self):
        """
        returns the centroids of each cell in the frame.
        """
        lbls = self.get_lbls()[1:]
        locations = []
        # compute the average
        for lbl in lbls:
            coords = cp.asarray((self.mask == lbl)).T  # mask to coord
            locations.append(cp.average(coords, axis=0))
        return cp.array(locations)
