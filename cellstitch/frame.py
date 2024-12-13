import cupy as cp


class Frame:
    def __init__(self, mask):
        """A container to the mask with useful features."""
        self.mask = mask

    def get_lbls(self):
        return cp.unique(self.mask)

    def get_sizes(self):
        """
        Calculate sizes of each mask from frame.
        """
        sizes = [self.get_size(lbl) for lbl in self.get_lbls()]
        return cp.asarray(sizes)

    def is_empty(self):
        """
        return if the frame is empty.
        """
        return len(self.get_lbls()) == 1

    def get_size(self, lbl):
        """
        get the size of the given lbl from the frame.
        """
        return cp.sum((self.mask == int(lbl)))

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
