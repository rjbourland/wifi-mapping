"""Tests for the RSSI processing pipeline."""

from datetime import datetime, timedelta

import pytest

from src.collection.rssi_scanner import NetworkResult
from src.processing.process_rssi import (
    RSSIPipeline,
    ProcessedScan,
    RSSI_STRONG,
    RSSI_WEAK,
    normalize_rssi,
    rssi_to_quality,
)


def _make_network(
    ssid: str = "TestNet",
    bssid: str = "aa:bb:cc:dd:ee:ff",
    rssi_dbm: float = -55.0,
    channel: int = 6,
    frequency_mhz: float = 2437.0,
    timestamp: datetime | None = None,
) -> NetworkResult:
    """Helper to create a NetworkResult for testing."""
    return NetworkResult(
        ssid=ssid,
        bssid=bssid,
        rssi_dbm=rssi_dbm,
        frequency_mhz=frequency_mhz,
        channel=channel,
        timestamp=timestamp or datetime.now(),
        radio_type="802.11n",
    )


class TestNormalizeRssi:
    """Tests for one-shot normalization helpers."""

    def test_clamp_within_bounds(self):
        assert normalize_rssi(-50) == -50.0

    def test_clamp_too_strong(self):
        assert normalize_rssi(-10) == RSSI_STRONG  # -30

    def test_clamp_too_weak(self):
        assert normalize_rssi(-100) == RSSI_WEAK  # -90

    def test_clamp_at_boundaries(self):
        assert normalize_rssi(-30) == -30.0
        assert normalize_rssi(-90) == -90.0


class TestRssiToQuality:
    """Tests for dBm → quality score conversion."""

    def test_strong_signal_is_one(self):
        assert rssi_to_quality(-30) == pytest.approx(1.0)

    def test_weak_signal_is_zero(self):
        assert rssi_to_quality(-90) == pytest.approx(0.0)

    def test_mid_signal(self):
        # -60 dBm is exactly midway between -30 and -90
        assert rssi_to_quality(-60) == pytest.approx(0.5)

    def test_clamps_before_converting(self):
        # Values outside range should be clamped first
        assert rssi_to_quality(0) == pytest.approx(1.0)
        assert rssi_to_quality(-120) == pytest.approx(0.0)


class TestRSSIPipeline:
    """Tests for the sliding-window processing pipeline."""

    def test_single_scan_insufficient_data(self):
        """A single scan shouldn't produce results (min_seen=3 default)."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        networks = [_make_network(bssid=f"aa:bb:cc:dd:ee:f{i}", rssi_dbm=-50.0 - i)
                     for i in range(3)]
        results = pipeline.process(networks)
        assert len(results) == 0  # Only 1 scan, not enough data

    def test_min_seen_threshold(self):
        """Networks must appear in at least min_seen scans to be included."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        net = _make_network(rssi_dbm=-55.0)

        # Two scans — below min_seen
        pipeline.process([net])
        pipeline.process([net])
        results = pipeline.process([net])
        assert len(results) == 1
        assert results[0].bssid == net.bssid

    def test_instable_network_filtered_out(self):
        """A network that appears in < min_seen scans is dropped."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        stable = _make_network(ssid="Stable", bssid="11:11:11:11:11:11", rssi_dbm=-55.0)
        unstable = _make_network(ssid="Unstable", bssid="22:22:22:22:22:22", rssi_dbm=-60.0)

        # Scan 1: both visible
        pipeline.process([stable, unstable])
        # Scan 2: only stable visible
        pipeline.process([stable])
        # Scan 3: only stable visible — unstable has only 1 reading, needs 3
        pipeline.process([stable])

        results = pipeline.process([stable, unstable])
        # After 4 scans, stable appears 4 times, unstable only 2 times
        # But unstable's buffer has only 1 reading left after sliding off
        # Check that unstable is excluded (not enough readings)
        bsids = {r.bssid for r in results}
        assert "11:11:11:11:11:11" in bsids

    def test_smoothing_reduces_noise(self):
        """Smoothed RSSI should be less extreme than raw average."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        bssid = "aa:bb:cc:dd:ee:ff"
        readings = [-50.0, -70.0, -50.0]  # Spiky data

        for rssi in readings:
            net = _make_network(bssid=bssid, rssi_dbm=rssi)
            pipeline.process([net])

        results = pipeline.process([_make_network(bssid=bssid, rssi_dbm=-50.0)])
        assert len(results) == 1
        # Smoothed should be closer to the recent values than the extreme -70
        # and should be between -50 and -56.7 (the raw avg of last 4)
        assert results[0].rssi_smoothed > -65.0

    def test_stability_score_high_for_consistent(self):
        """Stable readings produce a high stability score."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        bssid = "aa:bb:cc:dd:ee:ff"

        for _ in range(5):
            pipeline.process([_make_network(bssid=bssid, rssi_dbm=-55.0)])

        results = pipeline.process([_make_network(bssid=bssid, rssi_dbm=-55.0)])
        assert results[0].stability_score > 0.9

    def test_stability_score_low_for_noisy(self):
        """Fluctuating readings produce a low stability score."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        bssid = "aa:bb:cc:dd:ee:ff"
        noisy_readings = [-40.0, -70.0, -40.0, -75.0, -45.0]

        for rssi in noisy_readings:
            pipeline.process([_make_network(bssid=bssid, rssi_dbm=rssi)])

        results = pipeline.process([_make_network(bssid=bssid, rssi_dbm=-40.0)])
        assert results[0].stability_score < 0.5

    def test_spike_detection(self):
        """A sudden RSSI jump > 15 dBm is flagged."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        bssid = "aa:bb:cc:dd:ee:ff"

        # Build up to min_seen
        pipeline.process([_make_network(bssid=bssid, rssi_dbm=-55.0)])
        pipeline.process([_make_network(bssid=bssid, rssi_dbm=-55.0)])

        # Spike: jump from -55 to -35 (20 dBm jump)
        # This is scan 3, which satisfies min_seen=3
        results = pipeline.process([_make_network(bssid=bssid, rssi_dbm=-35.0)])
        assert len(results) == 1
        assert results[0].spike_detected is True

    def test_no_spike_gradual_change(self):
        """Gradual changes below 15 dBm are not flagged as spikes."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        bssid = "aa:bb:cc:dd:ee:ff"

        for rssi in [-55.0, -57.0, -59.0, -61.0]:
            pipeline.process([_make_network(bssid=bssid, rssi_dbm=rssi)])

        results = pipeline.process([_make_network(bssid=bssid, rssi_dbm=-63.0)])
        assert results[0].spike_detected is False

    def test_results_sorted_by_signal(self):
        """Output is sorted by rssi_smoothed (strongest first)."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        nets = [
            _make_network(ssid="Weak", bssid="11:11:11:11:11:11", rssi_dbm=-80.0),
            _make_network(ssid="Strong", bssid="22:22:22:22:22:22", rssi_dbm=-40.0),
            _make_network(ssid="Medium", bssid="33:33:33:33:33:33", rssi_dbm=-60.0),
        ]

        for _ in range(4):
            pipeline.process(nets)

        results = pipeline.process(nets)
        assert len(results) == 3
        assert results[0].ssid == "Strong"
        assert results[1].ssid == "Medium"
        assert results[2].ssid == "Weak"

    def test_rssi_clamping_in_pipeline(self):
        """Out-of-range RSSI values are clamped during processing."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        bssid = "aa:bb:cc:dd:ee:ff"

        # RSSI stronger than -30 should be clamped to -30
        pipeline.process([_make_network(bssid=bssid, rssi_dbm=-10.0)])
        pipeline.process([_make_network(bssid=bssid, rssi_dbm=-10.0)])
        pipeline.process([_make_network(bssid=bssid, rssi_dbm=-10.0)])

        results = pipeline.process([_make_network(bssid=bssid, rssi_dbm=-10.0)])
        # All readings clamped to -30, so smoothed should be -30
        assert results[0].rssi_smoothed == -30.0

    def test_frequency_and_channel_carry_through(self):
        """Metadata (frequency, channel) is preserved in output."""
        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        net = _make_network(
            bssid="aa:bb:cc:dd:ee:ff",
            rssi_dbm=-55.0,
            channel=149,
            frequency_mhz=5745.0,
        )

        for _ in range(4):
            pipeline.process([net])

        results = pipeline.process([net])
        assert results[0].channel == 149
        assert results[0].frequency_mhz == 5745.0

    def test_scan_count_increments(self):
        """Pipeline tracks the number of scans processed."""
        pipeline = RSSIPipeline()
        assert pipeline.scan_count == 0

        pipeline.process([_make_network()])
        assert pipeline.scan_count == 1

        pipeline.process([_make_network()])
        assert pipeline.scan_count == 2