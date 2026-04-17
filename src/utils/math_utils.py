"""Math utilities for geometry, distance, and coordinate transforms."""

import numpy as np
from typing import Union


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def path_loss_distance(rssi: float, rssi_d0: float = -30.0, n: float = 3.0, d0: float = 1.0) -> float:
    """Estimate distance from RSSI using the log-distance path-loss model.

    d = d0 * 10^((rssi_d0 - rssi) / (10 * n))

    Args:
        rssi: Measured RSSI in dBm (negative).
        rssi_d0: Reference RSSI at distance d0 (default: -30 dBm).
        n: Path-loss exponent (2.0=free space, 2.7-3.5=indoor).
        d0: Reference distance in meters (default: 1.0 m).

    Returns:
        Estimated distance in meters.
    """
    return d0 * 10 ** ((rssi_d0 - rssi) / (10.0 * n))


def path_loss_rssi(distance: float, rssi_d0: float = -30.0, n: float = 3.0, d0: float = 1.0) -> float:
    """Predict RSSI at a given distance using the path-loss model.

    rssi(d) = rssi_d0 - 10 * n * log10(d / d0)

    Args:
        distance: Distance in meters.
        rssi_d0: Reference RSSI at d0.
        n: Path-loss exponent.
        d0: Reference distance in meters.

    Returns:
        Predicted RSSI in dBm.
    """
    if distance < d0:
        distance = d0
    return rssi_d0 - 10.0 * n * np.log10(distance / d0)


def centroid(points: np.ndarray) -> np.ndarray:
    """Compute the centroid (geometric mean) of a set of points.

    Args:
        points: Array of shape (N, D) where N is number of points, D is dimensions.

    Returns:
        Centroid of shape (D,).
    """
    return np.mean(points, axis=0)


def angle_of_arrival_1d(csi: np.ndarray, subcarrier_spacing: float = 312.5e3) -> float:
    """Estimate Angle of Arrival from 2-antenna CSI using phase difference.

    This is a simplified 2-antenna AoA estimator. For proper MUSIC-based
    AoA estimation, use src.localization.aoa_estimation.

    Args:
        csi: Complex CSI matrix, shape (2, num_subcarriers).
        subcarrier_spacing: Subcarrier spacing in Hz (312.5 kHz for 802.11).

    Returns:
        Estimated angle in degrees.
    """
    if csi.shape[0] != 2:
        raise ValueError("1D AoA estimation requires exactly 2 antennas")

    phase_diff = np.angle(csi[1, :] * np.conj(csi[0, :]))
    mean_phase_diff = np.mean(phase_diff)

    wavelength = 3e8 / (subcarrier_spacing * csi.shape[1])  # Approximate
    antenna_spacing = wavelength / 2  # Half-wavelength spacing assumed

    sin_theta = mean_phase_diff * wavelength / (2 * np.pi * antenna_spacing)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    return float(np.degrees(np.arcsin(sin_theta)))


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """2D rotation matrix around the Z axis.

    Args:
        angle_deg: Rotation angle in degrees.

    Returns:
        2x2 rotation matrix.
    """
    theta = np.radians(angle_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])


def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    """3D rotation matrix from Euler angles (in degrees).

    Args:
        rx: Rotation around X axis (degrees).
        ry: Rotation around Y axis (degrees).
        rz: Rotation around Z axis (degrees).

    Returns:
        3x3 rotation matrix.
    """
    ax, ay, az = np.radians([rx, ry, rz])

    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx