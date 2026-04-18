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

        self._templates: dict[str, dict] = {}
        self._init_default_templates()

    def _init_default_templates(self):
        """Create default gait templates from synthetic signatures."""
        # Walking: ~1.8 Hz step frequency, moderate doppler variance
        self._templates["walking"] = {
            "step_frequency_hz": 1.8,
            "doppler_variance": 0.15,
        }
        # Running: ~2.8 Hz step frequency, high doppler variance
        self._templates["running"] = {
            "step_frequency_hz": 2.8,
            "doppler_variance": 0.6,
        }
        # Stationary: near-zero frequency, very low variance
        self._templates["stationary"] = {
            "step_frequency_hz": 0.1,
            "doppler_variance": 0.01,
        }

    def add_template(
        self, label: str, csi_packets: np.ndarray, sample_rate: float = 50.0
    ):
        """Record a gait template from CSI data.

        Args:
            label: Template name (e.g. 'walking', 'running').
            csi_packets: Complex CSI packets, shape (N, antennas, subcarriers).
            sample_rate: CSI sampling rate in Hz.
        """
        features = self.extract_gait_features(csi_packets, sample_rate)
        self._templates[label] = {
            "step_frequency_hz": features["step_frequency_hz"],
            "doppler_variance": features["doppler_variance"],
        }

    def classify(
        self, csi_packets: np.ndarray, sample_rate: float = 50.0
    ) -> dict:
        """Classify gait pattern from CSI data.

        Compares extracted features against stored templates using
        weighted distance in (step_frequency, doppler_variance) space.

        Args:
            csi_packets: Complex CSI packets, shape (N, antennas, subcarriers).
            sample_rate: CSI sampling rate in Hz.

        Returns:
            Dict with:
            - 'gait_type': Best-matching template label.
            - 'confidence': Match confidence (0-1).
            - 'step_frequency_hz', 'step_frequency_bpm', 'doppler_variance',
              'gait_period_s': Raw feature values.
        """
        features = self.extract_gait_features(csi_packets, sample_rate)

        if not self._templates:
            return {
                "gait_type": "unknown",
                "confidence": 0.0,
                **features,
            }

        best_label = "unknown"
        best_score = float("inf")

        for label, template in self._templates.items():
            # Weighted distance: frequency matters more than variance
            freq_diff = abs(features["step_frequency_hz"] - template["step_frequency_hz"])
            var_diff = abs(features["doppler_variance"] - template["doppler_variance"])
            # Normalize: freq range ~0-3 Hz, variance range ~0-1
            distance = 2.0 * freq_diff / 3.0 + var_diff
            if distance < best_score:
                best_score = distance
                best_label = label

        # Confidence: 1 at zero distance, decaying with distance
        confidence = 1.0 / (1.0 + 5.0 * best_score)

        return {
            "gait_type": best_label,
            "confidence": min(confidence, 1.0),
            **features,
        }

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