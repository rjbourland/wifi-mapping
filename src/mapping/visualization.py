"""3D visualization for mapping results."""

import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..utils.config import load_config, get_anchors

logger = logging.getLogger(__name__)


class Visualizer:
    """3D visualization for localization results, point clouds, and maps.

    Uses matplotlib for basic 3D visualization. Open3D support is optional
    and provides interactive 3D viewing.
    """

    def __init__(self, room_dimensions: Optional[dict] = None):
        """Initialize visualizer.

        Args:
            room_dimensions: Dict with 'length_x', 'width_y', 'height_z' in meters.
                           If None, loads from anchors.yaml config.
        """
        if room_dimensions is None:
            try:
                room_dimensions = get_room_dimensions()
            except Exception:
                room_dimensions = {"length_x": 4.0, "width_y": 4.0, "height_z": 2.5}

        self.room = room_dimensions

    def plot_anchors(self, anchors: list, ax: Optional[plt.Axes] = None):
        """Plot anchor positions in 3D.

        Args:
            anchors: List of AnchorPosition dicts with 'position' and 'id'.
            ax: Existing matplotlib 3D axes. Creates new figure if None.
        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

        colors = {"ceiling": "red", "mid": "orange", "floor": "green"}
        for anchor in anchors:
            pos = np.array(anchor["position"]) if isinstance(anchor["position"], list) else anchor["position"]
            height = anchor.get("height", "mid")
            color = colors.get(height, "blue")
            ax.scatter(*pos, c=color, s=200, marker="^", zorder=5)
            ax.text(pos[0], pos[1], pos[2] + 0.1, anchor["id"], fontsize=9, ha="center")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Anchor Positions")

        return fig

    def plot_trajectory(
        self,
        positions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        anchors: Optional[list] = None,
        title: str = "Localized Trajectory",
    ):
        """Plot a localized trajectory in 3D.

        Args:
            positions: (N, 3) array of localized positions.
            ground_truth: Optional (N, 3) array of ground-truth positions.
            anchors: Optional list of anchor dicts to plot.
            title: Plot title.
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                "b.-", label="Estimated", alpha=0.7)

        if ground_truth is not None:
            ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                    "r-", label="Ground Truth", alpha=0.7)

        if anchors is not None:
            for anchor in anchors:
                pos = np.array(anchor["position"]) if isinstance(anchor["position"], list) else anchor["position"]
                ax.scatter(*pos, c="k", s=200, marker="^", zorder=5)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()

        return fig

    def plot_heatmap(
        self,
        positions: np.ndarray,
        z_slice: Optional[float] = None,
        resolution: float = 0.1,
        title: str = "Position Density Heatmap",
    ):
        """Plot a 2D heatmap of position density.

        Args:
            positions: (N, 3) array of positions.
            z_slice: If specified, only include positions near this z-height.
            resolution: Grid resolution in meters.
            title: Plot title.
        """
        if z_slice is not None:
            mask = np.abs(positions[:, 2] - z_slice) < resolution * 2
            xy = positions[mask, :2]
        else:
            xy = positions[:, :2]

        if len(xy) == 0:
            logger.warning("No positions to plot in heatmap")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap, xedges, yedges = np.histogram2d(
            xy[:, 0], xy[:, 1],
            bins=[
                np.arange(0, self.room["length_x"] + resolution, resolution),
                np.arange(0, self.room["width_y"] + resolution, resolution),
            ],
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin="lower", cmap="hot", aspect="auto")
        plt.colorbar(im, ax=ax, label="Count")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)

        return fig

    def show_open3d(self, points: np.ndarray):
        """Display point cloud using Open3D (if available).

        Args:
            points: (N, 3) array of 3D points.
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.error("Open3D not installed. Install with: pip install open3d")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd], window_name="WiFi Mapping Point Cloud")


def get_room_dimensions():
    """Load room dimensions from config."""
    return load_config("anchors").get("room", {"length_x": 4.0, "width_y": 4.0, "height_z": 2.5})