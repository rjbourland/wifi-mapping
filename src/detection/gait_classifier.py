"""CSI-based gait pattern classification."""

import logging
from typing import Optional

import numpy as np
from scipy import signal

from ..utils.config import load_config

logger = logging.getLogger(__name__)


class GaitClassifier:
    """Classifies gait patterns from CSI Doppler signatures.

    Extracts gait features from CSI time series and classifies
    walking patterns. Uses frequency-domain analysis of the
    Doppler shift pattern caused by walking movement.

    This is a research-grade classifier — for production use,
    consider training a deep learning model (e.g., Widar 3.0 BVP approach).
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize gait classifier.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
        """
        if config is None:
            config = load_config("algorithm")

        self._templates: dict[str, np.ndarray] = {}

    def extract_gait_features(
        self, csi_packets: np.ndarray, sample_rate: float = 50.0
    ) -> dict:
        """Extract gait features from CSI packets.

        Args:
            csi_packets: Complex CSI packets, shape (num_packets, num_antennas, num_subcarriers).
            sample_rate: CSI sampling rate in Hz.

        Returns:
            Dict with gait features:
            - 'step_frequency_hz': Estimated step frequency
            - 'step_frequency_bpm': Steps per minute
            - 'doppler_variance': Variance of Doppler shifts
            - 'gait_period_s': Estimated gait period in seconds
        """
        # Compute Doppler shifts (Widar-style conjugate multiplication)
        doppler = np.zeros(csi_packets.shape[0] - 1)
        for t in range(1, csi_packets.shape[0]):
            delta = csi_packets[t] * np.conj(csi_packets[t - 1])
            doppler[t - 1] = np.mean(np.angle(delta))

        # Compute step frequency via FFT of Doppler
        n = len(doppler)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        spectrum = np.abs(np.fft.rfft(doppler))

        # Find dominant frequency in walking range (0.5-3 Hz = 30-180 steps/min)
        walking_mask = (freqs >= 0.5) & (freqs <= 3.0)
        walking_spectrum = spectrum[walking_mask]
        walking_freqs = freqs[walking_mask]

        if len(walking_spectrum) == 0 or np.max(walking_spectrum) == 0:
            return {
                "step_frequency_hz": 0.0,
                "step_frequency_bpm": 0.0,
                "doppler_variance": float(np.var(doppler)),
                "gait_period_s": 0.0,
            }

        peak_idx = np.argmax(walking_spectrum)
        step_freq = walking_freqs[peak_idx]

        return {
            "step_frequency_hz": float(step_freq),
            "step_frequency_bpm": float(step_freq * 60.0),
            "doppler_variance": float(np.var(doppler)),
            "gait_period_s": float(1.0 / step_freq) if step_freq > 0 else 0.0,
        }