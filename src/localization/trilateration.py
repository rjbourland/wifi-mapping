"""RSSI-based trilateration for 2D/3D localization.

Implements log-distance path-loss model and weighted least-squares
trilateration using multiple anchor positions.  Supports two input modes:

1. ``localize_from_scans()`` — takes ProcessedScan objects from the
   processing pipeline, matches BSSID → anchor, and uses stability-weighted
   least squares.
2. ``localize()`` — legacy interface accepting a simple dict of
   anchor_id → RSSI (dBm).
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..utils.data_formats import AnchorPosition, LocalizedPosition
from ..utils.math_utils import path_loss_distance

logger = logging.getLogger(__name__)

# Default path-loss parameters for localize_from_scans
DEFAULT_TX_POWER = -30.0   # dBm at 1 m reference distance
DEFAULT_PATH_LOSS_N = 2.0   # Exponent (2.0 = free-space / open indoor)

# Legacy defaults (matching algorithm.yaml)
LEGACY_PATH_LOSS_N = 3.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Estimated 2D position from trilateration."""

    x: float
    y: float
    estimated_error_meters: float
    timestamp: datetime
    ap_count: int


# ---------------------------------------------------------------------------
# Trilateration solver
# ---------------------------------------------------------------------------

class TrilaterationSolver:
    """Solves for position using RSSI trilateration from multiple anchors.

    Uses the log-distance path-loss model to convert RSSI to distance,
    then solves the system of circle intersections using non-linear
    weighted least squares.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize trilateration solver.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
                     If None, uses sensible defaults.
        """
        if config is not None:
            from ..utils.config import load_config
            if isinstance(config, str):
                config = load_config(config)

        pl_config = (config or {}).get("path_loss", {})
        tri_config = (config or {}).get("trilateration", {})

        self.rssi_d0: float = pl_config.get("rssi_d0", DEFAULT_TX_POWER)
        self.n: float = pl_config.get("n", LEGACY_PATH_LOSS_N)
        self.d0: float = pl_config.get("d0", 1.0)
        self.sigma: float = pl_config.get("sigma", 4.0)
        self.solver_method: str = tri_config.get("solver", "lm")
        self.initial_guess: str = tri_config.get("initial_guess", "centroid")

        self._anchors: list[AnchorPosition] = []

    def set_anchors(self, anchors: list[AnchorPosition]):
        """Set anchor node positions.

        Args:
            anchors: List of AnchorPosition objects with known 3D positions.
        """
        self._anchors = anchors
        logger.info("Set %d anchor positions", len(anchors))

    def load_anchors_from_config(self):
        """Load anchor positions from the anchors.yaml config file."""
        from ..utils.config import load_config

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
        logger.info("Loaded %d anchors from config", len(self._anchors))

    # ------------------------------------------------------------------
    # New API: ProcessedScan → Position
    # ------------------------------------------------------------------

    def localize_from_scans(
        self,
        scans: list,
        anchors: list[AnchorPosition],
        tx_power: float = DEFAULT_TX_POWER,
        n: float = DEFAULT_PATH_LOSS_N,
    ) -> Position:
        """Estimate 2D position from processed scan results.

        Each scan's BSSID is matched against anchor IDs (via the anchor's
        ``ip`` field or ``anchor_id``).  Distance is estimated using the
        log-distance path-loss model with *tx_power* at 1 m reference.

        Weights are set to each AP's ``stability_score`` so that stable
        readings dominate the least-squares solution.

        Args:
            scans: List of ProcessedScan objects from the processing pipeline.
            anchors: List of AnchorPosition with known (x, y, z) positions.
            tx_power: Reference RSSI at 1 m (default -30 dBm).
            n: Path-loss exponent (default 2.0).

        Returns:
            Position with estimated x, y, error, timestamp, and ap_count.

        Raises:
            ValueError: If fewer than 3 matching APs are found.
        """
        # Build a lookup: bssid → anchor position
        anchor_by_bssid: dict[str, AnchorPosition] = {}
        for a in anchors:
            # Allow matching by anchor_id or ip (some configs store MAC as ip)
            anchor_by_bssid[a.anchor_id] = a
            if a.ip:
                anchor_by_bssid[a.ip] = a

        anchor_positions = []
        distances = []
        weights = []

        for scan in scans:
            anchor = anchor_by_bssid.get(scan.bssid)
            if anchor is None:
                continue
            dist = path_loss_distance(scan.rssi_smoothed, tx_power, n)
            anchor_positions.append(anchor.position[:2])  # 2D only
            distances.append(dist)
            weights.append(max(scan.stability_score, 0.1))  # floor weight

        if len(anchor_positions) < 3:
            raise ValueError(
                f"Need at least 3 anchors with matching BSSIDs for "
                f"trilateration, got {len(anchor_positions)}. "
                f"Ensure anchor IPs match scanned BSSIDs."
            )

        anchor_positions = np.array(anchor_positions)
        distances = np.array(distances)
        weights = np.array(weights)

        return self._solve_2d(
            anchor_positions, distances, weights,
            timestamp=scans[0].last_seen if scans else datetime.now(),
            ap_count=len(anchor_positions),
        )

    # ------------------------------------------------------------------
    # Legacy API: dict → LocalizedPosition
    # ------------------------------------------------------------------

    def localize(self, rssi_measurements: dict[str, float]) -> LocalizedPosition:
        """Estimate 3D position from RSSI measurements at multiple anchors.

        Args:
            rssi_measurements: Dict mapping anchor_id → RSSI (dBm).

        Returns:
            LocalizedPosition with estimated (x, y, z) coordinates.
        """
        if not self._anchors:
            raise RuntimeError(
                "No anchors set. Call set_anchors() or load_anchors_from_config() first."
            )

        anchor_positions = []
        distances = []
        for anchor in self._anchors:
            if anchor.anchor_id in rssi_measurements:
                anchor_positions.append(anchor.position)
                distances.append(
                    self.rssi_to_distance(rssi_measurements[anchor.anchor_id])
                )

        if len(anchor_positions) < 3:
            raise ValueError(
                f"Need at least 3 anchors with RSSI measurements for 2D, "
                f"4 for 3D. Got {len(anchor_positions)}."
            )

        anchor_positions = np.array(anchor_positions)
        distances = np.array(distances)

        # Pad to 3D if needed
        if anchor_positions.shape[1] < 3:
            anchor_positions = np.hstack([
                anchor_positions,
                np.zeros((len(anchor_positions), 3 - anchor_positions.shape[1])),
            ])

        # Initial guess
        x0 = np.mean(anchor_positions, axis=0) if self.initial_guess == "centroid" else np.zeros(3)

        result = least_squares(
            self._residuals, x0,
            args=(anchor_positions, distances),
            method=self.solver_method,
        )

        if not result.success:
            logger.warning("Least-squares solver did not converge: %s", result.message)

        return LocalizedPosition(
            timestamp=datetime.now(),
            position=result.x,
            method="trilateration",
            anchors_used=list(rssi_measurements.keys()),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def rssi_to_distance(self, rssi: float) -> float:
        """Convert RSSI to estimated distance using path-loss model."""
        return path_loss_distance(rssi, self.rssi_d0, self.n, self.d0)

    @staticmethod
    def _residuals(
        x: np.ndarray, anchors: np.ndarray, distances: np.ndarray,
    ) -> np.ndarray:
        """Residual for least-squares: ‖x - anchor_i‖ - distance_i."""
        return np.linalg.norm(x - anchors, axis=1) - distances

    def _solve_2d(
        self,
        anchor_positions: np.ndarray,
        distances: np.ndarray,
        weights: np.ndarray,
        timestamp: datetime,
        ap_count: int,
    ) -> Position:
        """Weighted least-squares 2D trilateration.

        Args:
            anchor_positions: (N, 2) array of anchor x,y positions.
            distances: (N,) estimated distances.
            weights: (N,) per-anchor weights (stability scores).
            timestamp: Timestamp for the result.
            ap_count: Number of APs used.

        Returns:
            Position with x, y, estimated_error_meters, timestamp, ap_count.
        """
        # Initial guess: weighted centroid
        weighted_pos = anchor_positions * weights[:, np.newaxis]
        x0 = weighted_pos.sum(axis=0) / weights.sum()

        # Weighted residual function
        def weighted_residuals(pos: np.ndarray) -> np.ndarray:
            raw = np.linalg.norm(pos - anchor_positions, axis=1) - distances
            return raw * np.sqrt(weights)

        result = least_squares(weighted_residuals, x0, method=self.solver_method)

        x, y = result.x[0], result.x[1]

        # Estimated error: RMS of weighted residuals at solution
        residual_rms = float(np.sqrt(np.mean(result.fun ** 2)))

        return Position(
            x=x,
            y=y,
            estimated_error_meters=round(residual_rms, 3),
            timestamp=timestamp,
            ap_count=ap_count,
        )