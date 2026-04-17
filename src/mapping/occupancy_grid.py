"""Voxel-based occupancy grid mapping."""

import numpy as np
from typing import Optional

from ..utils.config import load_config


class OccupancyGrid:
    """3D occupancy grid for indoor mapping.

    Divides space into voxels and tracks occupancy probability
    using a log-odds representation. Free and occupied voxels are
    identified based on localization measurements and ray-casting.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize occupancy grid.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
        """
        if config is None:
            config = load_config("algorithm")

        map_config = config.get("mapping", {})
        self.voxel_size = map_config.get("voxel_size", 0.1)
        self.occupied_threshold = map_config.get("occupied_threshold", 0.7)
        self.free_threshold = map_config.get("free_threshold", 0.3)

        # Grid bounds (will be set when first point is added)
        self._origin = np.zeros(3)
        self._grid_shape = None
        self._log_odds = None

    def initialize(self, bounds_min: np.ndarray, bounds_max: np.ndarray):
        """Initialize the grid with spatial bounds.

        Args:
            bounds_min: (x, y, z) minimum bounds in meters.
            bounds_max: (x, y, z) maximum bounds in meters.
        """
        self._origin = bounds_min
        grid_size = np.ceil((bounds_max - bounds_min) / self.voxel_size).astype(int)
        self._grid_shape = tuple(grid_size)
        # Initialize with log-odds of 0 (50% prior probability)
        self._log_odds = np.zeros(self._grid_shape, dtype=np.float32)

    def update(self, position: np.ndarray, free_positions: Optional[np.ndarray] = None):
        """Update the occupancy grid with a new observation.

        Args:
            position: Observed occupied position (x, y, z).
            free_positions: (N, 3) array of positions along the ray
                           from anchor to position (marked as free space).
        """
        if self._log_odds is None:
            raise RuntimeError("Call initialize() before updating the grid.")

        # Mark observed position as occupied
        idx = self._to_index(position)
        if idx is not None:
            self._log_odds[idx] += 0.7  # Log-odds update for occupied

        # Mark ray-traced positions as free
        if free_positions is not None:
            for fp in free_positions:
                idx = self._to_index(fp)
                if idx is not None:
                    self._log_odds[idx] -= 0.3  # Log-odds update for free

    def get_occupied_voxels(self) -> np.ndarray:
        """Get positions of all occupied voxels.

        Returns:
            (N, 3) array of occupied voxel center positions.
        """
        if self._log_odds is None:
            return np.empty((0, 3))

        occupied = self._log_odds > np.log(self.occupied_threshold / (1 - self.occupied_threshold))
        indices = np.argwhere(occupied)
        positions = indices * self.voxel_size + self._origin + self.voxel_size / 2
        return positions

    def get_free_voxels(self) -> np.ndarray:
        """Get positions of all free voxels.

        Returns:
            (N, 3) array of free voxel center positions.
        """
        if self._log_odds is None:
            return np.empty((0, 3))

        free = self._log_odds < np.log(self.free_threshold / (1 - self.free_threshold))
        indices = np.argwhere(free)
        positions = indices * self.voxel_size + self._origin + self.voxel_size / 2
        return positions

    @property
    def probability_grid(self) -> np.ndarray:
        """Occupancy probability grid (0.0 = free, 1.0 = occupied)."""
        if self._log_odds is None:
            return np.empty((0, 0, 0))
        return 1.0 / (1.0 + np.exp(-self._log_odds))

    def _to_index(self, position: np.ndarray) -> Optional[tuple]:
        """Convert 3D position to grid index.

        Args:
            position: (x, y, z) position in meters.

        Returns:
            Tuple of (ix, iy, iz) indices, or None if out of bounds.
        """
        idx = np.floor((position - self._origin) / self.voxel_size).astype(int)
        if all(0 <= i < s for i, s in zip(idx, self._grid_shape)):
            return tuple(idx)
        return None