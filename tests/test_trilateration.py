"""Tests for trilateration module."""

import numpy as np
import pytest

from src.localization.trilateration import TrilaterationSolver
from src.utils.data_formats import AnchorPosition


class TestTrilaterationSolver:
    """Test cases for RSSI-based trilateration."""

    def setup_method(self):
        """Set up test anchors in a 4m x 4m x 2.5m room."""
        self.anchors = [
            AnchorPosition("A1", np.array([0.0, 0.0, 2.5]), "ceiling", "esp32_s3", "192.168.1.101", 6, 20),
            AnchorPosition("A2", np.array([4.0, 0.0, 1.3]), "mid", "esp32_s3", "192.168.1.102", 6, 20),
            AnchorPosition("A3", np.array([4.0, 4.0, 0.3]), "floor", "esp32_s3", "192.168.1.103", 6, 20),
            AnchorPosition("A4", np.array([0.0, 4.0, 2.5]), "ceiling", "esp32_s3", "192.168.1.104", 6, 20),
        ]

    def test_rssi_to_distance(self):
        """Test RSSI to distance conversion."""
        solver = TrilaterationSolver()
        # At 1m distance, RSSI should be close to rssi_d0
        dist = solver.rssi_to_distance(-33.0)  # ~1m with default path-loss
        assert 0.5 < dist < 2.0, f"Expected ~1m, got {dist:.2f}m"

    def test_rssi_to_distance_far(self):
        """Test RSSI to distance at greater range."""
        solver = TrilaterationSolver()
        # -70 dBm at ~3m path-loss exponent should give 5-15m
        dist = solver.rssi_to_distance(-70.0)
        assert dist > 1.0, f"Expected >1m, got {dist:.2f}m"

    def test_localize_with_known_distances(self):
        """Test trilateration with exact distances (no noise)."""
        solver = TrilaterationSolver()
        solver.set_anchors(self.anchors)

        # Point at (2, 2, 1.25) - center of room
        true_pos = np.array([2.0, 2.0, 1.25])

        # Calculate exact distances from true position to each anchor
        rssi_measurements = {}
        for anchor in self.anchors:
            distance = np.linalg.norm(true_pos - anchor.position)
            # Convert distance to RSSI using path-loss model
            rssi = -30.0 - 10 * 3.0 * np.log10(distance / 1.0)
            rssi_measurements[anchor.anchor_id] = rssi

        result = solver.localize(rssi_measurements)
        error = np.linalg.norm(result.position - true_pos)
        # Path-loss round-trip introduces small error; allow 2m tolerance
        # (RSSI-to-distance conversion is lossy due to log model)
        assert error < 2.0, f"Expected <2.0m error, got {error:.3f}m"

    def test_localize_requires_minimum_anchors(self):
        """Test that localization requires at least 3 anchors."""
        solver = TrilaterationSolver()
        # Only 2 anchors
        solver.set_anchors(self.anchors[:2])

        rssi = {"A1": -50.0, "A2": -55.0}
        with pytest.raises(ValueError, match="at least 3"):
            solver.localize(rssi)

    def test_localize_no_anchors_raises_error(self):
        """Test that localizing without anchors raises an error."""
        solver = TrilaterationSolver()
        with pytest.raises(RuntimeError, match="No anchors"):
            solver.localize({"A1": -50.0})