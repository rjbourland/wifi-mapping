"""Variance-based motion detection from CSI data."""

import logging
from typing import Optional

import numpy as np

from ..utils.config import load_config
from ..processing.feature_extraction import compute_variance_features

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detects motion from CSI variance patterns.

    Uses amplitude variance across subcarriers and time to detect
    when a person or object is moving in the environment. When someone
    moves through a Fresnel zone, the CSI amplitude and phase fluctuate,
    creating detectable variance signatures.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize motion detector.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
        """
        if config is None:
            config = load_config("algorithm")

        det_config = config.get("motion_detection", {})
        self.variance_threshold = det_config.get("variance_threshold", 0.5)
        self.window_size = det_config.get("window_size", 20)
        self.min_duration = det_config.get("min_duration", 0.5)

        self._motion_buffer: list[bool] = []
        self._is_motion = False

    def detect(self, csi_packets: np.ndarray) -> dict:
        """Detect motion from a window of CSI packets.

        Args:
            csi_packets: Complex CSI packets, shape (num_packets, num_antennas, num_subcarriers).

        Returns:
            Dict with keys:
            - 'is_motion': Boolean — motion detected in this window
            - 'motion_score': Scalar motion intensity (higher = more motion)
            - 'amplitude_variance': Variance array per subcarrier
            - 'total_variance': Sum of amplitude variances
        """
        features = compute_variance_features(csi_packets, self.window_size)
        motion_score = features["motion_score"][-1] if len(features["motion_score"]) > 0 else 0.0

        is_motion = motion_score > self.variance_threshold

        # Debounce: require motion for min_duration consecutive windows
        self._motion_buffer.append(is_motion)
        min_windows = max(1, int(self.min_duration * 10))  # Approximate

        sustained_motion = (
            len(self._motion_buffer) >= min_windows
            and all(self._motion_buffer[-min_windows:])
        )

        self._is_motion = sustained_motion or is_motion

        return {
            "is_motion": self._is_motion,
            "motion_score": float(motion_score),
            "amplitude_variance": features["amplitude_variance"][-1] if len(features["amplitude_variance"]) > 0 else None,
            "total_variance": float(features["total_variance"][-1]) if len(features["total_variance"]) > 0 else 0.0,
        }

    def reset(self):
        """Reset the motion detection state."""
        self._motion_buffer = []
        self._is_motion = False