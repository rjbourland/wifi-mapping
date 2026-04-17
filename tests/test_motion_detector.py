"""Tests for motion detection module."""

import numpy as np
import pytest

from src.detection.motion_detector import MotionDetector


class TestMotionDetector:
    """Test cases for CSI-based motion detection."""

    def test_no_motion(self):
        """Test that static CSI data produces no motion detection."""
        detector = MotionDetector()

        # Static CSI: constant amplitude across all packets
        num_packets = 50
        csi = np.ones((num_packets, 2, 52), dtype=complex)
        # Add small noise to simulate measurement noise
        csi += np.random.randn(*csi.shape) * 0.01 + 1j * np.random.randn(*csi.shape) * 0.01

        result = detector.detect(csi)
        assert result["is_motion"] is False, "Should not detect motion from static data"
        assert result["motion_score"] < 1.0, "Motion score should be low for static data"

    def test_motion_detected(self):
        """Test that varying CSI data produces motion detection."""
        detector = MotionDetector({"variance_threshold": 0.3, "window_size": 10, "min_duration": 0.1})

        # Simulate motion: large variance in CSI amplitude over time
        num_packets = 50
        t = np.linspace(0, 2 * np.pi, num_packets)
        # Create large amplitude variations simulating motion
        amplitudes = 1.0 + 5.0 * np.sin(t)[:, np.newaxis, np.newaxis]  # Shape: (50, 1, 1)
        phases = np.random.randn(num_packets, 2, 52)
        csi = amplitudes * np.exp(1j * phases)

        result = detector.detect(csi)
        assert result["motion_score"] > 0.5, f"Motion score should be high for moving data, got {result['motion_score']}"

    def test_reset(self):
        """Test that reset clears motion detection state."""
        detector = MotionDetector()
        detector._motion_buffer = [True, True]
        detector._is_motion = True

        detector.reset()
        assert detector._motion_buffer == []
        assert detector._is_motion is False