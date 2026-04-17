"""Feature extraction from CSI data for localization and detection."""

import logging
from typing import Optional

import numpy as np
from scipy import signal, stats

logger = logging.getLogger(__name__)


def extract_amplitude(csi: np.ndarray) -> np.ndarray:
    """Extract amplitude from complex CSI matrix.

    Args:
        csi: Complex CSI matrix, shape (num_antennas, num_subcarriers) or
             (num_packets, num_antennas, num_subcarriers).

    Returns:
        Amplitude array of same shape.
    """
    return np.abs(csi)


def extract_phase(csi: np.ndarray, unwrap: bool = True) -> np.ndarray:
    """Extract phase from complex CSI matrix.

    Args:
        csi: Complex CSI matrix.
        unwrap: Whether to unwrap phase to remove 2pi discontinuities.

    Returns:
        Phase array of same shape, in radians.
    """
    phase = np.angle(csi)
    if unwrap:
        phase = np.unwrap(phase, axis=-1)
    return phase


def extract_phase_difference(csi: np.ndarray) -> np.ndarray:
    """Extract phase difference between consecutive subcarriers.

    Useful for CFO removal and AoA estimation.

    Args:
        csi: Complex CSI matrix, shape (num_antennas, num_subcarriers).

    Returns:
        Phase difference array, shape (num_antennas, num_subcarriers - 1).
    """
    phase = np.angle(csi)
    return np.diff(phase, axis=-1)


def extract_doppler_shift(
    csi_packets: np.ndarray, window_size: int = 10
) -> np.ndarray:
    """Extract Doppler shift from a sequence of CSI packets.

    Uses the power conjugate multiplication method from Widar.

    Args:
        csi_packets: Complex CSI packets, shape (num_packets, num_antennas, num_subcarriers).
        window_size: Number of packets for each Doppler estimate.

    Returns:
        Doppler shift estimates, shape (num_packets - window_size + 1,).
    """
    # Power conjugate multiplication: H[t] * conj(H[t-1])
    doppler = np.zeros(csi_packets.shape[0] - 1)
    for t in range(1, csi_packets.shape[0]):
        # Phase change across all antennas and subcarriers
        delta = csi_packets[t] * np.conj(csi_packets[t - 1])
        doppler[t - 1] = np.mean(np.angle(delta))

    # Smooth with moving average
    if len(doppler) >= window_size:
        kernel = np.ones(window_size) / window_size
        doppler = np.convolve(doppler, kernel, mode="valid")

    return doppler


def compute_variance_features(
    csi_packets: np.ndarray, window_size: int = 20
) -> dict[str, np.ndarray]:
    """Compute variance-based features from a stream of CSI packets.

    Used for motion detection and activity recognition.

    Args:
        csi_packets: Complex CSI packets, shape (num_packets, num_antennas, num_subcarriers).
        window_size: Sliding window size for variance computation.

    Returns:
        Dict with keys:
        - 'amplitude_variance': Variance of amplitude over time per subcarrier
        - 'phase_variance': Variance of phase over time per subcarrier
        - 'total_variance': Sum of amplitude variances across all subcarriers
        - 'motion_score': Scalar motion indicator (higher = more motion)
    """
    amplitude = np.abs(csi_packets)
    phase = np.unwrap(np.angle(csi_packets), axis=-1)

    # Sliding window variance
    num_packets = csi_packets.shape[0]
    num_antennas = csi_packets.shape[1]
    num_subcarriers = csi_packets.shape[2]

    amp_var = np.zeros((max(num_packets - window_size + 1, 1), num_antennas, num_subcarriers))
    phase_var = np.zeros_like(amp_var)

    for i in range(amp_var.shape[0]):
        start = i
        end = min(i + window_size, num_packets)
        amp_var[i] = np.var(amplitude[start:end], axis=0)
        phase_var[i] = np.var(phase[start:end], axis=0)

    total_var = np.sum(amp_var, axis=(1, 2))
    motion_score = total_var / (np.mean(total_var) + 1e-10)

    return {
        "amplitude_variance": amp_var,
        "phase_variance": phase_var,
        "total_variance": total_var,
        "motion_score": motion_score,
    }


def extract_subcarrier_features(csi: np.ndarray) -> dict[str, float]:
    """Extract statistical features from a single CSI matrix.

    Useful for fingerprinting-based localization.

    Args:
        csi: Complex CSI matrix, shape (num_antennas, num_subcarriers).

    Returns:
        Dict of feature name -> value, including:
        - mean_amp, std_amp, max_amp, min_amp
        - mean_phase, std_phase
        - kurtosis_amp, skew_amp
        - rms_amp (root mean square amplitude)
    """
    amp = np.abs(csi)
    phase = np.angle(csi)

    return {
        "mean_amp": float(np.mean(amp)),
        "std_amp": float(np.std(amp)),
        "max_amp": float(np.max(amp)),
        "min_amp": float(np.min(amp)),
        "mean_phase": float(np.mean(phase)),
        "std_phase": float(np.std(phase)),
        "kurtosis_amp": float(stats.kurtosis(amp.flatten())),
        "skew_amp": float(stats.skew(amp.flatten())),
        "rms_amp": float(np.sqrt(np.mean(amp ** 2))),
    }