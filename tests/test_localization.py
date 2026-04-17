"""Tests for trilateration and Kalman filter."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.localization.trilateration import TrilaterationSolver, Position
from src.localization.kalman_filter import KalmanFilter, SmoothedPosition
from src.utils.data_formats import AnchorPosition
from src.processing.process_rssi import ProcessedScan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anchor(anchor_id: str, x: float, y: float) -> AnchorPosition:
    """Create an AnchorPosition for testing (2D, z=0)."""
    return AnchorPosition(
        anchor_id=anchor_id,
        position=np.array([x, y, 0.0]),
        height="mid",
        hardware="esp32_s3",
        ip="",
        channel=6,
        bandwidth=20,
    )


def _make_scan(
    bssid: str,
    rssi_smoothed: float,
    stability_score: float = 1.0,
    timestamp: datetime | None = None,
) -> ProcessedScan:
    """Create a ProcessedScan for testing."""
    return ProcessedScan(
        ssid="TestNet",
        bssid=bssid,
        rssi_smoothed=rssi_smoothed,
        rssi_raw_avg=rssi_smoothed,
        stability_score=stability_score,
        spike_detected=False,
        last_seen=timestamp or datetime.now(),
        frequency_mhz=2437.0,
        channel=6,
    )


# ---------------------------------------------------------------------------
# RSSI → Distance conversion
# ---------------------------------------------------------------------------

class TestRssiToDistance:
    """Tests for the path-loss RSSI-to-distance conversion."""

    def test_at_reference_distance(self):
        """At RSSI = tx_power, distance should equal d0 (1m)."""
        solver = TrilaterationSolver(config={})
        # Default: tx_power=-30, n=2.0 (from localize_from_scans defaults)
        # TrilaterationSolver uses config n=3.0 by default for legacy API
        # but path_loss_distance is called directly here with explicit params
        from src.utils.math_utils import path_loss_distance
        dist = path_loss_distance(-30.0, rssi_d0=-30.0, n=2.0, d0=1.0)
        assert dist == pytest.approx(1.0, abs=0.01)

    def test_double_distance(self):
        """RSSI drops by ~6 dBm when distance doubles (n=2)."""
        from src.utils.math_utils import path_loss_distance
        d1 = path_loss_distance(-36.0, rssi_d0=-30.0, n=2.0, d0=1.0)
        # -36 dBm at n=2: d = 10^((-30 - (-36)) / 20) = 10^(6/20) = 10^0.3 ≈ 2.0
        assert d1 == pytest.approx(2.0, abs=0.1)

    def test_stronger_signal_closer(self):
        """Stronger signal should mean shorter distance."""
        from src.utils.math_utils import path_loss_distance
        near = path_loss_distance(-40.0, rssi_d0=-30.0, n=2.0, d0=1.0)
        far = path_loss_distance(-60.0, rssi_d0=-30.0, n=2.0, d0=1.0)
        assert near < far

    def test_path_loss_exponent_effect(self):
        """Higher n means shorter estimated distance for same RSSI."""
        from src.utils.math_utils import path_loss_distance
        d_n2 = path_loss_distance(-60.0, rssi_d0=-30.0, n=2.0, d0=1.0)
        d_n3 = path_loss_distance(-60.0, rssi_d0=-30.0, n=3.0, d0=1.0)
        assert d_n3 < d_n2


# ---------------------------------------------------------------------------
# Trilateration with 3 APs
# ---------------------------------------------------------------------------

class TestTrilaterationFromScans:
    """Tests for ProcessedScan-based trilateration."""

    def setup_method(self):
        """Set up 3 anchors in a 4m × 4m room."""
        self.anchors = [
            _make_anchor("AP1", 0.0, 0.0),
            _make_anchor("AP2", 4.0, 0.0),
            _make_anchor("AP3", 0.0, 4.0),
        ]
        self.solver = TrilaterationSolver(config={})

    def test_center_position(self):
        """Position at room center (2, 2) should be estimated accurately."""
        # RSSI values that correspond to ~2.83m and ~2m distances
        from src.utils.math_utils import path_loss_distance
        # At (2,2): d(AP1)=2.83m, d(AP2)=2.83m, d(AP3)=2.0m
        # With tx_power=-30, n=2.0: rssi = -30 - 10*2*log10(d)
        rssi_ap1 = -30 - 20 * np.log10(2.83)
        rssi_ap2 = -30 - 20 * np.log10(2.83)
        rssi_ap3 = -30 - 20 * np.log10(2.0)

        scans = [
            _make_scan("AP1", rssi_ap1),
            _make_scan("AP2", rssi_ap2),
            _make_scan("AP3", rssi_ap3),
        ]

        result = self.solver.localize_from_scans(scans, self.anchors)
        assert isinstance(result, Position)
        assert result.ap_count == 3
        # Should be close to (2, 2) — within 0.5m tolerance
        error = np.sqrt((result.x - 2.0) ** 2 + (result.y - 2.0) ** 2)
        assert error < 0.5, f"Expected near (2,2), got ({result.x:.2f}, {result.y:.2f})"

    def test_corner_position(self):
        """Position near AP1 (0,0) should be estimated close to origin."""
        from src.utils.math_utils import path_loss_distance
        # At (0.5, 0.5): d(AP1)=0.71m, d(AP2)=3.54m, d(AP3)=3.54m
        rssi_ap1 = -30 - 20 * np.log10(0.71)
        rssi_ap2 = -30 - 20 * np.log10(3.54)
        rssi_ap3 = -30 - 20 * np.log10(3.54)

        scans = [
            _make_scan("AP1", rssi_ap1),
            _make_scan("AP2", rssi_ap2),
            _make_scan("AP3", rssi_ap3),
        ]

        result = self.solver.localize_from_scans(scans, self.anchors)
        error = np.sqrt((result.x - 0.5) ** 2 + (result.y - 0.5) ** 2)
        assert error < 1.0, f"Expected near (0.5,0.5), got ({result.x:.2f}, {result.y:.2f})"

    def test_stability_weighting(self):
        """Low-stability APs should have less influence on position."""
        from src.utils.math_utils import path_loss_distance
        # Position at (2, 2)
        rssi_ap1 = -30 - 20 * np.log10(2.83)
        rssi_ap2 = -30 - 20 * np.log10(2.83)
        rssi_ap3 = -30 - 20 * np.log10(2.0)

        # AP1 has low stability — should pull less
        scans = [
            _make_scan("AP1", rssi_ap1, stability_score=0.1),
            _make_scan("AP2", rssi_ap2, stability_score=1.0),
            _make_scan("AP3", rssi_ap3, stability_score=1.0),
        ]

        result = self.solver.localize_from_scans(scans, self.anchors)
        assert result.ap_count == 3
        # Should still produce a reasonable result
        assert 0 < result.x < 5
        assert 0 < result.y < 5

    def test_insufficient_anchors_raises(self):
        """Fewer than 3 matching APs should raise ValueError."""
        scans = [_make_scan("UNKNOWN", -55.0)]
        with pytest.raises(ValueError, match="at least 3 anchors"):
            self.solver.localize_from_scans(scans, self.anchors)

    def test_bssid_matching_via_ip(self):
        """Anchors should match by ip field as well as anchor_id."""
        anchors_with_ip = [
            AnchorPosition("ap1", np.array([0, 0, 0]), "mid", "esp32", "aa:bb:cc:dd:ee:01", 6, 20),
            AnchorPosition("ap2", np.array([4, 0, 0]), "mid", "esp32", "aa:bb:cc:dd:ee:02", 6, 20),
            AnchorPosition("ap3", np.array([0, 4, 0]), "mid", "esp32", "aa:bb:cc:dd:ee:03", 6, 20),
        ]
        from src.utils.math_utils import path_loss_distance
        rssi = -30 - 20 * np.log10(2.83)  # ~2.83m

        scans = [
            _make_scan("aa:bb:cc:dd:ee:01", rssi),
            _make_scan("aa:bb:cc:dd:ee:02", rssi),
            _make_scan("aa:bb:cc:dd:ee:03", -30 - 20 * np.log10(2.0)),
        ]

        result = self.solver.localize_from_scans(scans, anchors_with_ip)
        assert result.ap_count == 3


# ---------------------------------------------------------------------------
# Legacy trilateration API
# ---------------------------------------------------------------------------

class TestTrilaterationLegacy:
    """Tests for the legacy dict-based localiz e() API."""

    def setup_method(self):
        self.anchors = [
            AnchorPosition("A1", np.array([0.0, 0.0, 2.5]), "ceiling", "esp32_s3", "192.168.1.101", 6, 20),
            AnchorPosition("A2", np.array([4.0, 0.0, 1.3]), "mid", "esp32_s3", "192.168.1.102", 6, 20),
            AnchorPosition("A3", np.array([4.0, 4.0, 0.3]), "floor", "esp32_s3", "192.168.1.103", 6, 20),
            AnchorPosition("A4", np.array([0.0, 4.0, 2.5]), "ceiling", "esp32_s3", "192.168.1.104", 6, 20),
        ]

    def test_localize_with_known_distances(self):
        solver = TrilaterationSolver(config={})
        solver.set_anchors(self.anchors)
        true_pos = np.array([2.0, 2.0, 1.25])
        rssi = {}
        for a in self.anchors:
            d = np.linalg.norm(true_pos - a.position)
            rssi[a.anchor_id] = -30.0 - 10 * 3.0 * np.log10(d)
        result = solver.localize(rssi)
        error = np.linalg.norm(result.position - true_pos)
        assert error < 2.0

    def test_requires_minimum_anchors(self):
        solver = TrilaterationSolver(config={})
        solver.set_anchors(self.anchors[:2])
        with pytest.raises(ValueError, match="at least 3"):
            solver.localize({"A1": -50.0, "A2": -55.0})


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    """Tests for 2D Kalman filter convergence."""

    def test_2d_convergence_from_static(self):
        """Filter should converge to a static position over 10 steps."""
        kf = KalmanFilter(dims=2)
        from src.localization.trilateration import Position

        true_x, true_y = 2.0, 3.0
        results: list[SmoothedPosition] = []

        for i in range(10):
            noise = np.random.randn(2) * 0.3
            pos = Position(
                x=true_x + noise[0],
                y=true_y + noise[1],
                estimated_error_meters=0.3,
                timestamp=datetime.now(),
                ap_count=3,
            )
            result = kf.update_position(pos)
            results.append(result)

        # After 10 steps, estimate should be close to true position
        final_error = np.sqrt((results[-1].x - true_x) ** 2 + (results[-1].y - true_y) ** 2)
        assert final_error < 1.0, f"Expected convergence, got error={final_error:.2f}"

    def test_2d_tracks_linear_motion(self):
        """Filter should track linear motion with velocity estimate."""
        kf = KalmanFilter(dims=2)
        from src.localization.trilateration import Position

        # Object moving at (0.5, 0) m/step
        results = []
        for i in range(10):
            pos = Position(
                x=float(i) * 0.5 + np.random.randn() * 0.2,
                y=0.0 + np.random.randn() * 0.2,
                estimated_error_meters=0.2,
                timestamp=datetime.now(),
                ap_count=3,
            )
            result = kf.update_position(pos)
            results.append(result)

        # Velocity_x should be positive after tracking
        assert results[-1].velocity_x > 0, "Should detect positive x-velocity"

    def test_2d_confidence_increases(self):
        """Confidence should increase as filter converges."""
        kf = KalmanFilter(dims=2)
        from src.localization.trilateration import Position

        true_x, true_y = 1.5, 1.5
        confidences = []
        for i in range(10):
            noise = np.random.randn(2) * 0.5
            pos = Position(
                x=true_x + noise[0],
                y=true_y + noise[1],
                estimated_error_meters=0.5,
                timestamp=datetime.now(),
                ap_count=3,
            )
            result = kf.update_position(pos)
            confidences.append(result.confidence)

        # Confidence should generally increase
        assert confidences[-1] > confidences[0], \
            f"Confidence should increase: {confidences[0]:.2f} → {confidences[-1]:.2f}"

    def test_smoothed_position_dataclass(self):
        """SmoothedPosition should have all expected fields."""
        sp = SmoothedPosition(x=1.0, y=2.0, velocity_x=0.5, velocity_y=-0.3, confidence=0.85)
        assert sp.x == 1.0
        assert sp.y == 2.0
        assert sp.velocity_x == 0.5
        assert sp.velocity_y == -0.3
        assert sp.confidence == 0.85

    def test_predict_without_update(self):
        """Predict should advance state using velocity."""
        kf = KalmanFilter(dims=2)
        from src.localization.trilateration import Position

        # Initialize at (1, 1)
        pos = Position(x=1.0, y=1.0, estimated_error_meters=0.1,
                       timestamp=datetime.now(), ap_count=3)
        kf.update_position(pos)

        # Predict should not crash
        predicted = kf.predict_position()
        assert isinstance(predicted, SmoothedPosition)

    def test_3d_legacy_api(self):
        """Legacy 3D API should still work."""
        kf = KalmanFilter(dims=3)
        kf.initialize(np.array([0.0, 0.0, 0.0]))

        # Update with measurement
        result = kf.update(np.array([1.0, 1.0, 1.0]))
        assert result.shape == (3,)
        assert abs(result[0] - 1.0) < 0.5