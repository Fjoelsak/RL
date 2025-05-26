import numpy as np

# Coarse Coding (Tile Coding) Setup
class TileCoder:
    """
    A Tile Coder for discretizing continuous state spaces into sparse feature representations.

    Tile coding is a form of coarse coding used in reinforcement learning to convert
    continuous input spaces into binary feature vectors using multiple overlapping tilings.
    Each tiling partitions the input space into a grid of tiles with a small offset to
    improve generalization and representation.

    Attributes:
        low (np.ndarray): Lower bounds for each dimension of the input space.
        high (np.ndarray): Upper bounds for each dimension of the input space.
        tilings (int): Number of overlapping tilings to use.
        bins (list or np.ndarray): Number of tiles (bins) per dimension in each tiling.
        offsets (list of np.ndarray): Offset values for each tiling and each dimension.
        tile_width (np.ndarray): Width of each tile in each dimension.

    Methods:
        get_features(state):
            Returns a list of active tile indices for a given state, one for each tiling.
    """
    def __init__(self, low, high, tilings, bins):
        self.low = np.array(low)
        self.high = np.array(high)
        self.tilings = tilings
        self.bins = bins
        self.dimensions = len(low)
        self.offsets = [np.linspace(0, 1, tilings, endpoint=False) for _ in range(self.dimensions)]
        self.tile_width = (self.high - self.low) / (np.array(bins) - 1)

    def get_features(self, state):
        """
            Returns the active tile indices for a given continuous state across all tilings.

            For each tiling, the state is slightly shifted by a unique offset, then discretized
            into tile indices based on the tile widths. The result is a list of coordinates
            representing the active tile in each tiling.

            Args:
                state (array-like): A list or array representing the continuous state
                    (e.g., position and velocity in MountainCar).

            Returns:
                list of tuples: Each tuple contains the tiling index followed by the
                    tile indices for each dimension. For example:
                    [(0, i0, j0), (1, i1, j1), ..., (T-1, iT-1, jT-1)],
                    where T is the number of tilings.
            """
        features = []
        for tiling in range(self.tilings):
            indices = []
            for i in range(self.dimensions):
                # Apply offset
                offset = self.offsets[i][tiling] * self.tile_width[i]
                value = state[i] + offset
                index = int((value - self.low[i]) / self.tile_width[i])
                indices.append(index)
            features.append((tiling,) + tuple(indices))
        return features