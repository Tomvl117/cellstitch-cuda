import numpy as np


class Frame:
    def __init__(self, mask):
        """A container to the mask with useful features."""
        self.mask = mask

    def get_lbls(self):
        return np.unique(self.mask)

    def get_sizes(self):
        """
        Calculate sizes of each mask from frame.
        """
        sizes = [(self.mask == lbl).sum() for lbl in self.get_lbls()]
        return np.array(sizes)

    def is_empty(self):
        """
        return if the frame is empty.
        """
        return len(self.get_lbls()) == 1

    def get_size(self, lbl):
        """
        get the size of the given lbl from the frame.
        """
        return (self.mask == lbl).sum()

    def get_locations(self):
        """
        returns the centroids of each cell in the frame.
        """
        lbls = self.get_lbls()[1:]
        locations = []
        # compute the average
        for lbl in lbls:
            coords = np.asarray((self.mask == lbl)).T  # mask to coord
            locations.append(np.average(coords, axis=0))
        return np.array(locations)
