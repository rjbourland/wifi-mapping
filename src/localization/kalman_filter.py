"""Kalman filter for position tracking and smoothing."""

import numpy as np
from typing import Optional

from ..utils.config import load_config


class KalmanFilter:
    """Linear Kalman filter for tracking and smoothing localized positions.

    State vector: [x, y, z, vx, vy, vz] (position and velocity).
    Measurement vector: [x, y, z] (localized position).

    Uses a constant-velocity motion model with configurable process
    and measurement noise.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize Kalman filter.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
        """
        if config is None:
            config = load_config("algorithm")

        kf_config = config.get("kalman", {})
        self.process_noise = kf_config.get("process_noise", 0.1)
        self.measurement_noise = kf_config.get("measurement_noise", 1.0)
        self.initial_uncertainty = kf_config.get("initial_uncertainty", 10.0)

        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.P = np.eye(6) * self.initial_uncertainty

        # State transition matrix (constant velocity model)
        self.F = np.eye(6)
        dt = 0.02  # 50 Hz update rate
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt

        # Measurement matrix (observe position only)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1  # observe x
        self.H[1, 1] = 1  # observe y
        self.H[2, 2] = 1  # observe z

        # Process noise covariance
        self.Q = np.eye(6) * self.process_noise

        # Measurement noise covariance
        self.R = np.eye(3) * self.measurement_noise

        self._initialized = False

    def initialize(self, position: np.ndarray):
        """Initialize the filter with a known position.

        Args:
            position: Initial (x, y, z) position.
        """
        self.state[:3] = position
        self.state[3:] = 0  # Zero initial velocity
        self.P = np.eye(6) * self.initial_uncertainty
        self._initialized = True

    def predict(self) -> np.ndarray:
        """Predict the next state using the motion model.

        Returns:
            Predicted position [x, y, z].
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:3].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update the filter with a position measurement.

        Args:
            measurement: Measured [x, y, z] position.

        Returns:
            Updated position estimate [x, y, z].
        """
        if not self._initialized:
            self.initialize(measurement)
            return self.state[:3].copy()

        # Innovation (measurement residual)
        y = measurement - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ y

        # Covariance update
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

        return self.state[:3].copy()

    @property
    def position(self) -> np.ndarray:
        """Current position estimate."""
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate."""
        return self.state[3:].copy()

    @property
    def position_uncertainty(self) -> np.ndarray:
        """Position uncertainty (standard deviation in each axis)."""
        return np.sqrt(np.diag(self.P)[:3])