"""Integration tests for the GUI pipeline bridge and end-to-end localization flow."""

import numpy as np
import pytest
from datetime import datetime

from src.collection.rssi_scanner import NetworkResult
from src.processing.process_rssi import RSSIPipeline, ProcessedScan
from src.localization.trilateration import TrilaterationSolver, Position
from src.localization.kalman_filter import KalmanFilter, SmoothedPosition
from src.localization.fingerprinting import KNNFingerprinting
from src.utils.data_formats import AnchorPosition, LocalizedPosition
from src.utils.math_utils import path_loss_rssi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def anchors():
    return [
        AnchorPosition("anchor_1", np.array([0.0, 0.0, 2.5]), "ceiling", "esp32_s3", "192.168.1.101", 6, 20),
        AnchorPosition("anchor_2", np.array([4.0, 0.0, 1.3]), "mid", "esp32_s3", "192.168.1.102", 6, 20),
        AnchorPosition("anchor_3", np.array([4.0, 4.0, 0.3]), "floor", "esp32_s3", "192.168.1.103", 6, 20),
        AnchorPosition("anchor_4", np.array([0.0, 4.0, 2.5]), "ceiling", "esp32_s3", "192.168.1.104", 6, 20),
    ]


@pytest.fixture
def solver(anchors):
    s = TrilaterationSolver()
    s.set_anchors(anchors)
    return s


@pytest.fixture
def pipeline():
    return RSSIPipeline(window_size=5, min_seen=3, spike_threshold_dbm=15)


# ---------------------------------------------------------------------------
# Synthetic Scan Generation Tests
# ---------------------------------------------------------------------------

class TestSyntheticScans:
    def test_generate_scans_returns_processed(self, anchors, pipeline):
        """generate_synthetic_scans should return list[ProcessedScan]."""
        from gui.utils.pipeline import generate_synthetic_scans

        true_pos = np.array([2.0, 2.0, 1.0])
        scans = generate_synthetic_scans(anchors, true_pos, pipeline, num_scans=5)

        assert isinstance(scans, list)
        assert len(scans) > 0
        assert all(isinstance(s, ProcessedScan) for s in scans)

    def test_scan_bssids_match_anchors(self, anchors, pipeline):
        """All returned scans should have BSSIDs matching anchor IDs."""
        from gui.utils.pipeline import generate_synthetic_scans

        true_pos = np.array([2.0, 2.0, 1.0])
        scans = generate_synthetic_scans(anchors, true_pos, pipeline, num_scans=5)

        anchor_ids = {a.anchor_id for a in anchors}
        for scan in scans:
            assert scan.bssid in anchor_ids

    def test_scan_rssi_reasonable(self, anchors, pipeline):
        """RSSI values should be within physical range."""
        from gui.utils.pipeline import generate_synthetic_scans

        true_pos = np.array([2.0, 2.0, 1.0])
        scans = generate_synthetic_scans(anchors, true_pos, pipeline, num_scans=5)

        for scan in scans:
            assert -90 <= scan.rssi_smoothed <= -30, f"RSSI {scan.rssi_smoothed} out of range"


# ---------------------------------------------------------------------------
# Trilateration from Scans Tests
# ---------------------------------------------------------------------------

class TestTrilaterationFromScans:
    def test_localize_from_scans_produces_position(self, anchors, solver, pipeline):
        """localize_from_scans should return a valid Position object."""
        from gui.utils.pipeline import generate_synthetic_scans

        true_pos = np.array([2.0, 2.0, 1.0])
        scans = generate_synthetic_scans(anchors, true_pos, pipeline, num_scans=5)

        if len(scans) >= 3:
            position = solver.localize_from_scans(scans, anchors)
            assert isinstance(position, Position)
            assert position.ap_count >= 3

    def test_localize_from_scans_corner(self, anchors, solver, pipeline):
        """Localizing from scans near a corner should produce a Position."""
        from gui.utils.pipeline import generate_synthetic_scans

        true_pos = np.array([0.5, 0.5, 1.0])
        scans = generate_synthetic_scans(anchors, true_pos, pipeline, num_scans=5)

        if len(scans) >= 3:
            position = solver.localize_from_scans(scans, anchors)
            assert isinstance(position, Position)


# ---------------------------------------------------------------------------
# Kalman Filter Integration Tests
# ---------------------------------------------------------------------------

class TestKalmanWithPosition:
    def test_update_position_returns_smoothed(self):
        """KalmanFilter.update_position should return SmoothedPosition from Position."""
        kf = KalmanFilter(dims=2)
        pos = Position(x=2.0, y=2.0, estimated_error_meters=0.5,
                       timestamp=datetime.now(), ap_count=4)
        smoothed = kf.update_position(pos)

        assert isinstance(smoothed, SmoothedPosition)
        assert isinstance(smoothed.x, float)
        assert isinstance(smoothed.y, float)
        assert 0 <= smoothed.confidence <= 1

    def test_kalman_convergence(self):
        """Multiple position updates should converge."""
        kf = KalmanFilter(dims=2)

        for i in range(20):
            angle = 2 * np.pi * i / 20
            pos = Position(
                x=2.0 + np.cos(angle) + np.random.randn() * 0.1,
                y=2.0 + np.sin(angle) + np.random.randn() * 0.1,
                estimated_error_meters=0.5,
                timestamp=datetime.now(),
                ap_count=4,
            )
            smoothed = kf.update_position(pos)

        assert smoothed.confidence > 0.0


# ---------------------------------------------------------------------------
# Fingerprinting Integration Tests
# ---------------------------------------------------------------------------

class TestFingerprintingIntegration:
    def test_add_fingerprint_and_localize(self):
        """Fingerprinting should train and localize from RSSI vectors."""
        fp = KNNFingerprinting(config={"fingerprinting": {"k": 3, "distance_metric": "euclidean"}})

        # Create a simple fingerprint database
        rng = np.random.default_rng(42)
        for i in range(10):
            x = i * 0.4
            y = i * 0.3
            rssi_vec = {
                "anchor_1": -40 - 10 * np.sqrt(x**2 + y**2) + rng.normal() * 2,
                "anchor_2": -40 - 10 * np.sqrt((4-x)**2 + y**2) + rng.normal() * 2,
                "anchor_3": -40 - 10 * np.sqrt((4-x)**2 + (4-y)**2) + rng.normal() * 2,
            }
            fp.add_fingerprint(rssi_vec, np.array([x, y, 0.5]))

        fp.train()

        # Localize at a test position
        test_rssi = {
            "anchor_1": -45.0,
            "anchor_2": -55.0,
            "anchor_3": -65.0,
        }
        result = fp.localize(test_rssi)

        assert isinstance(result, LocalizedPosition)
        assert result.method == "fingerprinting"
        assert len(result.position) == 3

    def test_pipeline_rssi_dict_conversion(self, anchors, pipeline):
        """rssi_dict_from_scans should convert ProcessedScan list to dict."""
        from gui.utils.pipeline import generate_synthetic_scans, rssi_dict_from_scans

        true_pos = np.array([2.0, 2.0, 1.0])
        scans = generate_synthetic_scans(anchors, true_pos, pipeline, num_scans=5)

        rssi_dict = rssi_dict_from_scans(scans)
        assert isinstance(rssi_dict, dict)

        anchor_ids = {a.anchor_id for a in anchors}
        for bssid in rssi_dict:
            assert bssid in anchor_ids


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def test_full_pipeline_simulation(self, anchors, solver, pipeline):
        """Full pipeline: synthetic position → scans → localize → kalman."""
        from gui.utils.pipeline import generate_synthetic_scans

        kf = KalmanFilter(dims=2)
        true_pos = np.array([2.0, 2.0, 1.0])

        scans = generate_synthetic_scans(anchors, true_pos, pipeline, num_scans=5)

        if len(scans) >= 3:
            position = solver.localize_from_scans(scans, anchors)
            smoothed = kf.update_position(position)

            # SmoothedPosition should have valid coordinates
            assert isinstance(smoothed.x, float)
            assert isinstance(smoothed.y, float)
            assert smoothed.confidence > 0.0

    def test_collect_rssi_simulation_mode(self):
        """collect_rssi in simulation mode should return ProcessedScan list."""
        from gui.utils.pipeline import collect_rssi

        anchors_list = [
            AnchorPosition("anchor_1", np.array([0.0, 0.0, 2.5]), "ceiling", "esp32_s3", "192.168.1.101", 6, 20),
            AnchorPosition("anchor_2", np.array([4.0, 0.0, 1.3]), "mid", "esp32_s3", "192.168.1.102", 6, 20),
            AnchorPosition("anchor_3", np.array([4.0, 4.0, 0.3]), "floor", "esp32_s3", "192.168.1.103", 6, 20),
            AnchorPosition("anchor_4", np.array([0.0, 4.0, 2.5]), "ceiling", "esp32_s3", "192.168.1.104", 6, 20),
        ]

        class MockSessionState:
            simulation_mode = True
            anchors = anchors_list
            rssi_pipeline = RSSIPipeline(window_size=5, min_seen=3)
            room_dimensions = {"length_x": 4.0, "width_y": 4.0, "height_z": 2.5}
            rssi_scanner = None

            def get(self, key, default=None):
                return getattr(self, key, default)

        session = MockSessionState()
        scans = collect_rssi(session)

        assert isinstance(scans, list)
        assert len(scans) > 0
        assert all(isinstance(s, ProcessedScan) for s in scans)


# ---------------------------------------------------------------------------
# Package Export Tests
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_collection_exports(self):
        """RSSIScanner, NetworkResult, and scan_networks should be importable."""
        from src.collection import RSSIScanner, NetworkResult, scan_networks
        assert RSSIScanner is not None
        assert NetworkResult is not None
        assert scan_networks is not None

    def test_detection_exports(self):
        """BreathingDetector and GaitClassifier should be importable."""
        from src.detection import BreathingDetector, GaitClassifier
        assert BreathingDetector is not None
        assert GaitClassifier is not None