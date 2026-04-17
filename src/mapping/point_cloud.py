"""3D point cloud accumulation for indoor mapping."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PointCloudAccumulator:
    """Accumulates 3D position estimates into a point cloud for mapping.

    Collects localized positions over time and builds a 3D point cloud
    representation of the environment. Supports voxel downsampling
    and statistical outlier removal.
    """

    def __init__(self, voxel_size: float = 0.1):
        """Initialize point cloud accumulator.

        Args:
            voxel_size: Voxel size for downsampling in meters.
        """
        self.voxel_size = voxel_size
        self._points: list[np.ndarray] = []
        self._timestamps: list = []
        self._methods: list[str] = []

    def add_point(self, position: np.ndarray, method: str = "unknown"):
        """Add a localized position to the point cloud.

        Args:
            position: (x, y, z) coordinates in meters.
            method: Localization method used (e.g., 'trilateration', 'fingerprinting').
        """
        self._points.append(np.asarray(position, dtype=float))
        from datetime import datetime
        self._timestamps.append(datetime.now())
        self._methods.append(method)

    def add_points(self, positions: np.ndarray, method: str = "unknown"):
        """Add multiple positions at once.

        Args:
            positions: (N, 3) array of positions.
            method: Localization method.
        """
        for pos in positions:
            self.add_point(pos, method)

    @property
    def points(self) -> np.ndarray:
        """All accumulated points as (N, 3) array."""
        if not self._points:
            return np.empty((0, 3))
        return np.array(self._points)

    @property
    def num_points(self) -> int:
        """Number of accumulated points."""
        return len(self._points)

    def downsample(self) -> np.ndarray:
        """Voxel-downsample the point cloud.

        Returns:
            (M, 3) array of downsampled points where M <= N.
        """
        points = self.points
        if len(points) == 0:
            return points

        # Quantize to voxel grid
        voxel_indices = np.floor(points / self.voxel_size).astype(int)

        # Keep one point per voxel (first occurrence)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)

        return points[unique_idx]

    def remove_outliers(self, n_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        """Remove statistical outliers from the point cloud.

        Args:
            n_neighbors: Number of neighbors to consider.
            std_ratio: Points with average distance > mean + std_ratio * std
                       are removed.

        Returns:
            Filtered (N, 3) array.
        """
        points = self.points
        if len(points) < n_neighbors:
            return points

        # Compute mean distance to n_neighbors nearest points for each point
        from scipy.spatial import KDTree

        tree = KDTree(points)
        distances, _ = tree.query(points, k=n_neighbors + 1)  # +1 includes self
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self

        # Statistical outlier removal
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_ratio * global_std

        mask = mean_distances < threshold
        return points[mask]

    def save(self, filepath: Path):
        """Save point cloud to a binary file.

        Args:
            filepath: Output file path (.npy format).
        """
        points = self.points
        np.save(filepath, points)
        logger.info(f"Saved {len(points)} points to {filepath}")

    def save_ply(self, filepath: Path):
        """Save point cloud in PLY format (compatible with Open3D, MeshLab, etc.).

        Args:
            filepath: Output file path (.ply extension).
        """
        points = self.points
        with open(filepath, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        logger.info(f"Saved {len(points)} points to {filepath} (PLY format)")