"""Kalman filter for position tracking and smoothing."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

logger = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SmoothedPosition:
    """Kalman-filtered 2D position estimate."""

    x: float
    y: float
    velocity_x: float
    velocity_y: float
    confidence: float  # 0–1, derived from trace of position covariance


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Linear Kalman filter for tracking and smoothing localized positions.

    Supports both 2D and 3D modes:

    - **2D mode** (dims=2): State = [x, y, vx, vy].  Accepts ``Position``
      objects from trilateration and outputs ``SmoothedPosition``.
    - **3D mode** (dims=3): State = [x, y, z, vx, vy, vz].  Legacy mode
      accepting raw numpy arrays.

    Uses a constant-velocity motion model with configurable process
    and measurement noise.
    """

    def __init__(self, dims: int = 2, config: Optional[dict] = None):
        """Initialize Kalman filter.

        Args:
            dims: Spatial dimensions — 2 for floor-plane tracking,
                  3 for full 3D tracking.
            config: Algorithm configuration dict (from algorithm.yaml).
        """
        if config is not None:
            from ..utils.config import load_config
            if isinstance(config, str):
                config = load_config(config)

        kf_config = (config or {}).get("kalman", {})
        self.process_noise = kf_config.get("process_noise", 0.1)
        self.measurement_noise = kf_config.get("measurement_noise", 1.0)
        self.initial_uncertainty = kf_config.get("initial_uncertainty", 10.0)

        self.dims = dims
        state_size = dims * 2  # position + velocity

        # State: [x, y, (z,) vx, vy, (vz)]
        self.state = np.zeros(state_size)
        self.P = np.eye(state_size) * self.initial_uncertainty

        # State transition (constant velocity)
        dt = kf_config.get("dt", 1.0)  # default 1s between updates
        self.F = np.eye(state_size)
        for i in range(dims):
            self.F[i, dims + i] = dt

        # Measurement matrix (observe position only)
        self.H = np.zeros((dims, state_size))
        for i in range(dims):
            self.H[i, i] = 1

        # Process noise covariance
        self.Q = np.eye(state_size) * self.process_noise

        # Measurement noise covariance
        self.R = np.eye(dims) * self.measurement_noise

        self._initialized = False

    # ------------------------------------------------------------------
    # 2D API: Position → SmoothedPosition
    # ------------------------------------------------------------------

    def update_position(self, position) -> SmoothedPosition:
        """Update filter with a Position from trilateration and return smoothed result.

        Args:
            position: A Position object (from trilateration.py) with x, y.

        Returns:
            SmoothedPosition with smoothed x, y, velocities, and confidence.
        """
        measurement = np.array([position.x, position.y])
        return self._update_and_build_result(measurement)

    def predict_position(self) -> SmoothedPosition:
        """Predict next position using the motion model.

        Returns:
            SmoothedPosition with predicted x, y, velocities, and confidence.
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._build_result()

    # ------------------------------------------------------------------
    # Legacy 3D API
    # ------------------------------------------------------------------

    def initialize(self, position: np.ndarray):
        """Initialize the filter with a known position (3D legacy API).

        Args:
            position: Initial (x, y, z) or (x, y) position.
        """
        self.state[:len(position)] = position
        self.state[len(position):] = 0
        self.P = np.eye(len(self.state)) * self.initial_uncertainty
        self._initialized = True

    def predict(self) -> np.ndarray:
        """Predict the next state using the motion model (legacy API).

        Returns:
            Predicted position array.
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[: self.dims].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update the filter with a position measurement (legacy API).

        Args:
            measurement: Measured position array.

        Returns:
            Updated position estimate.
        """
        if not self._initialized:
            self.initialize(measurement)
            return self.state[: self.dims].copy()

        # Innovation
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        I = np.eye(len(self.state))
        self.P = (I - K @ self.H) @ self.P

        return self.state[: self.dims].copy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Current position estimate."""
        return self.state[: self.dims].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate."""
        return self.state[self.dims:].copy()

    @property
    def position_uncertainty(self) -> np.ndarray:
        """Position uncertainty (standard deviation in each axis)."""
        return np.sqrt(np.diag(self.P)[: self.dims])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_and_build_result(self, measurement: np.ndarray) -> SmoothedPosition:
        """Run a full predict + update cycle and build SmoothedPosition."""
        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        if not self._initialized:
            self.state[: self.dims] = measurement
            self._initialized = True
            self.P = np.eye(len(self.state)) * self.initial_uncertainty
            return self._build_result()

        # Update
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(len(self.state))
        self.P = (I - K @ self.H) @ self.P

        return self._build_result()

    def _build_result(self) -> SmoothedPosition:
        """Build a SmoothedPosition from current state."""
        pos_uncertainty = self.position_uncertainty
        # Confidence: map uncertainty to 0–1. Low uncertainty → high confidence.
        # Using: confidence = 1 / (1 + mean(pos_uncertainty))
        mean_unc = float(np.mean(pos_uncertainty))
        confidence = round(1.0 / (1.0 + mean_unc), 2)

        if self.dims == 2:
            return SmoothedPosition(
                x=round(float(self.state[0]), 3),
                y=round(float(self.state[1]), 3),
                velocity_x=round(float(self.state[2]), 3),
                velocity_y=round(float(self.state[3]), 3),
                confidence=confidence,
            )
        else:
            # 3D: still return SmoothedPosition (2D projection + z in velocity_x)
            return SmoothedPosition(
                x=round(float(self.state[0]), 3),
                y=round(float(self.state[1]), 3),
                velocity_x=round(float(self.state[3]), 3),
                velocity_y=round(float(self.state[4]), 3),
                confidence=confidence,
            )