"""RSSI-based trilateration for 3D localization.

Implements log-distance path-loss model and least-squares trilateration
using multiple anchor positions.
"""

import logging
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..utils.config import load_config, get_anchors
from ..utils.data_formats import RSSISample, AnchorPosition, LocalizedPosition
from ..utils.math_utils import path_loss_distance
from datetime import datetime

logger = logging.getLogger(__name__)


class TrilaterationSolver:
    """Solves for 3D position using RSSI trilateration from multiple anchors.

    Uses the log-distance path-loss model to convert RSSI to distance,
    then solves the system of circle/sphere intersections using
    non-linear least squares.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize trilateration solver.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
                     If None, loads from default config.
        """
        if config is None:
            config = load_config("algorithm")

        pl_config = config.get("path_loss", {})
        self.rssi_d0 = pl_config.get("rssi_d0", -30.0)
        self.n = pl_config.get("n", 3.0)
        self.d0 = pl_config.get("d0", 1.0)
        self.sigma = pl_config.get("sigma", 4.0)

        tri_config = config.get("trilateration", {})
        self.solver_method = tri_config.get("solver", "lm")
        self.initial_guess = tri_config.get("initial_guess", "centroid")

        self._anchors: list[AnchorPosition] = []

    def set_anchors(self, anchors: list[AnchorPosition]):
        """Set anchor node positions.

        Args:
            anchors: List of AnchorPosition objects with known 3D positions.
        """
        self._anchors = anchors
        logger.info(f"Set {len(anchors)} anchor positions")

    def load_anchors_from_config(self):
        """Load anchor positions from the anchors.yaml config file."""
        config = load_config("anchors")
        self._anchors = []
        for anchor_id, info in config.get("anchors", {}).items():
            self._anchors.append(AnchorPosition(
                anchor_id=anchor_id,
                position=np.array(info["position"]),
                height=info.get("height", "mid"),
                hardware=info.get("hardware", "unknown"),
                ip=info.get("ip", ""),
                channel=info.get("channel", 6),
                bandwidth=info.get("bandwidth", 20),
            ))
        logger.info(f"Loaded {len(self._anchors)} anchors from config")

    def rssi_to_distance(self, rssi: float) -> float:
        """Convert RSSI to estimated distance using path-loss model.

        Args:
            rssi: Measured RSSI in dBm (negative).

        Returns:
            Estimated distance in meters.
        """
        return path_loss_distance(rssi, self.rssi_d0, self.n, self.d0)

    def localize(self, rssi_measurements: dict[str, float]) -> LocalizedPosition:
        """Estimate 3D position from RSSI measurements at multiple anchors.

        Args:
            rssi_measurements: Dict mapping anchor_id -> RSSI (dBm).

        Returns:
            LocalizedPosition with estimated (x, y, z) coordinates.
        """
        if not self._anchors:
            raise RuntimeError("No anchors set. Call set_anchors() or load_anchors_from_config() first.")

        # Match RSSI measurements to anchor positions
        anchor_positions = []
        distances = []
        for anchor in self._anchors:
            if anchor.anchor_id in rssi_measurements:
                anchor_positions.append(anchor.position)
                distances.append(self.rssi_to_distance(rssi_measurements[anchor.anchor_id]))

        if len(anchor_positions) < 3:
            raise ValueError(
                f"Need at least 3 anchors with RSSI measurements for 2D, "
                f"4 for 3D. Got {len(anchor_positions)}."
            )

        anchor_positions = np.array(anchor_positions)
        distances = np.array(distances)

        # Initial guess
        if self.initial_guess == "centroid":
            x0 = np.mean(anchor_positions, axis=0)
        else:
            x0 = np.zeros(3)

        # Pad to 3D if needed
        if anchor_positions.shape[1] < 3:
            anchor_positions = np.hstack([
                anchor_positions,
                np.zeros((len(anchor_positions), 3 - anchor_positions.shape[1]))
            ])
            x0 = np.append(x0, [0.0] * (3 - len(x0)))

        # Solve using least squares
        result = least_squares(
            self._residuals,
            x0,
            args=(anchor_positions, distances),
            method=self.solver_method,
        )

        if not result.success:
            logger.warning(f"Least-squares solver did not converge: {result.message}")

        return LocalizedPosition(
            timestamp=datetime.now(),
            position=result.x,
            method="trilateration",
            anchors_used=list(rssi_measurements.keys()),
        )

    @staticmethod
    def _residuals(
        x: np.ndarray, anchors: np.ndarray, distances: np.ndarray
    ) -> np.ndarray:
        """Residual function for least-squares trilateration.

        The residual for anchor i is:
            ||x - anchor_i|| - distance_i

        Args:
            x: Estimated position, shape (3,).
            anchors: Anchor positions, shape (N, 3).
            distances: Estimated distances, shape (N,).

        Returns:
            Residual array, shape (N,).
        """
        return np.linalg.norm(x - anchors, axis=1) - distances