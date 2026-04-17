"""RSSI signal smoothing, filtering, and normalization pipeline.

Consumes NetworkResult objects from rssi_scanner, maintains a sliding
window of recent scans, and produces ProcessedScan results with smoothed
RSSI, stability scores, and spike detection.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from ..collection.rssi_scanner import NetworkResult

logger = logging.getLogger(__name__)

# Realistic RSSI bounds used for clamping
RSSI_STRONG = -30   # Best signal expected indoors
RSSI_WEAK = -90     # Weakest usable signal

# Default thresholds
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MIN_SEEN = 3
DEFAULT_SPIKE_THRESHOLD_DBM = 15


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ProcessedScan:
    """A processed WiFi network measurement after smoothing and filtering."""

    ssid: str
    bssid: str
    rssi_smoothed: float
    rssi_raw_avg: float
    stability_score: float          # 0.0 (unstable) to 1.0 (rock-solid)
    spike_detected: bool            # True if last scan jumped > threshold from previous
    last_seen: datetime
    frequency_mhz: float
    channel: int


# ---------------------------------------------------------------------------
# Internal scan buffer
# ---------------------------------------------------------------------------

@dataclass
class _ScanWindow:
    """Sliding window of RSSI readings for a single BSSID."""

    bssid: str
    ssid: str
    readings: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    frequencies: list[float] = field(default_factory=list)
    channels: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RSSIPipeline:
    """Sliding-window RSSI processing pipeline.

    Accumulates consecutive scan results and produces ProcessedScan objects
    with smoothed RSSI, stability scores, and spike flags.

    Usage::

        pipeline = RSSIPipeline(window_size=5, min_seen=3)
        for _ in range(5):
            networks = scan_networks()
            results = pipeline.process(networks)
        for r in results:
            print(f"{r.ssid}: {r.rssi_smoothed:.1f} dBm  stability={r.stability_score:.2f}")
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_seen: int = DEFAULT_MIN_SEEN,
        spike_threshold_dbm: float = DEFAULT_SPIKE_THRESHOLD_DBM,
    ):
        self.window_size = window_size
        self.min_seen = min_seen
        self.spike_threshold = spike_threshold_dbm
        self._buffers: dict[str, _ScanWindow] = {}
        self._scan_count = 0

    def process(self, networks: list[NetworkResult]) -> list[ProcessedScan]:
        """Feed one scan into the pipeline and return processed results.

        Networks that have been seen in fewer than ``min_seen`` of the last
        ``window_size`` scans are dropped from the output.

        Args:
            networks: Fresh scan results from rssi_scanner.

        Returns:
            List of ProcessedScan sorted by rssi_smoothed (strongest first).
        """
        self._scan_count += 1

        # Index current scan by BSSID
        current_by_bssid: dict[str, NetworkResult] = {n.bssid: n for n in networks}
        seen_bssids: set[str] = set()

        # Update existing buffers or create new ones
        for bssid, net in current_by_bssid.items():
            seen_bssids.add(bssid)
            if bssid not in self._buffers:
                self._buffers[bssid] = _ScanWindow(
                    bssid=bssid, ssid=net.ssid,
                )
            buf = self._buffers[bssid]
            buf.ssid = net.ssid  # SSID can change (roaming)
            buf.readings.append(self._clamp(net.rssi_dbm))
            buf.timestamps.append(net.timestamp)
            buf.frequencies.append(net.frequency_mhz)
            buf.channels.append(net.channel)
            # Trim to window
            if len(buf.readings) > self.window_size:
                buf.readings.pop(0)
                buf.timestamps.pop(0)
                buf.frequencies.pop(0)
                buf.channels.pop(0)

        # Mark networks NOT seen this scan (they keep existing data but
        # won't get a new reading; their window slides off naturally)

        # Build results — only networks seen enough times
        results: list[ProcessedScan] = []
        for bssid, buf in list(self._buffers.items()):
            if len(buf.readings) < self.min_seen:
                continue

            rssi_raw_avg = self._mean(buf.readings)
            rssi_smoothed = self._exponential_smooth(buf.readings)
            stability = self._stability_score(buf.readings)
            spike = self._detect_spike(buf.readings)

            # Use the most recent metadata
            last_freq = buf.frequencies[-1]
            last_channel = buf.channels[-1]
            last_seen = buf.timestamps[-1]

            results.append(ProcessedScan(
                ssid=buf.ssid,
                bssid=buf.bssid,
                rssi_smoothed=round(rssi_smoothed, 1),
                rssi_raw_avg=round(rssi_raw_avg, 1),
                stability_score=round(stability, 2),
                spike_detected=spike,
                last_seen=last_seen,
                frequency_mhz=last_freq,
                channel=last_channel,
            ))

        # Prune buffers that haven't been seen for a full window
        if self._scan_count > self.window_size:
            stale = [
                bssid for bssid, buf in self._buffers.items()
                if bssid not in seen_bssids and len(buf.readings) == 0
            ]
            for bssid in stale:
                del self._buffers[bssid]

        # For buffers not seen this scan, drop the oldest reading to slide
        for bssid, buf in self._buffers.items():
            if bssid not in seen_bssids and len(buf.readings) > 0:
                buf.readings.pop(0)
                buf.timestamps.pop(0)
                buf.frequencies.pop(0)
                buf.channels.pop(0)

        results.sort(key=lambda r: r.rssi_smoothed, reverse=True)
        logger.info("Pipeline produced %d stable networks from %d scanned",
                     len(results), len(networks))
        return results

    @property
    def scan_count(self) -> int:
        """Number of scans fed into the pipeline so far."""
        return self._scan_count

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _clamp(rssi: float) -> float:
        """Clamp RSSI to realistic bounds."""
        return max(RSSI_WEAK, min(RSSI_STRONG, rssi))

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Arithmetic mean of a list of floats."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _exponential_smooth(values: list[float], alpha: float = 0.4) -> float:
        """Exponential moving average (more weight on recent readings).

        Args:
            values: RSSI readings, oldest first.
            alpha: Smoothing factor (0 < alpha <= 1). Higher = more responsive.

        Returns:
            Smoothed value.
        """
        if not values:
            return 0.0
        smoothed = values[0]
        for v in values[1:]:
            smoothed = alpha * v + (1 - alpha) * smoothed
        return smoothed

    @staticmethod
    def _stability_score(values: list[float]) -> float:
        """Compute a 0–1 stability score based on RSSI variance.

        Low variance → score near 1 (stable).
        High variance → score near 0 (unstable).
        Uses the relationship: score = 1 / (1 + k * variance), where k scales
        so that a variance of 25 (std ≈ 5 dBm) yields a score of ~0.5.
        """
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        k = 0.04  # tuned so variance=25 → score ≈ 0.5
        return 1.0 / (1.0 + k * variance)

    @staticmethod
    def _detect_spike(values: list[float]) -> bool:
        """Flag if the most recent reading jumped > threshold from the previous."""
        if len(values) < 2:
            return False
        return abs(values[-1] - values[-2]) > DEFAULT_SPIKE_THRESHOLD_DBM


# ---------------------------------------------------------------------------
# Convenience: single-scan normalization (no history needed)
# ---------------------------------------------------------------------------

def normalize_rssi(rssi_dbm: float) -> float:
    """Clamp an RSSI value to realistic bounds and return it.

    Useful for one-shot normalization without a pipeline.

    Args:
        rssi_dbm: Raw RSSI in dBm.

    Returns:
        Clamped RSSI value in the range [RSSI_WEAK, RSSI_STRONG].
    """
    return max(RSSI_WEAK, min(RSSI_STRONG, rssi_dbm))


def rssi_to_quality(rssi_dbm: float) -> float:
    """Convert RSSI in dBm to a 0–1 quality score (linear mapping).

    -90 dBm → 0.0 (worst)
    -30 dBm → 1.0 (best)

    Args:
        rssi_dbm: RSSI in dBm (should already be clamped for best results).

    Returns:
        Quality score in [0, 1].
    """
    clamped = max(RSSI_WEAK, min(RSSI_STRONG, rssi_dbm))
    return (clamped - RSSI_WEAK) / (RSSI_STRONG - RSSI_WEAK)


# ---------------------------------------------------------------------------
# CLI: scan → pipeline → processed table
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from ..collection.rssi_scanner import scan_networks

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = RSSIPipeline(window_size=5, min_seen=3, spike_threshold_dbm=15)
    num_scans = 5

    print(f"Running {num_scans} scans (1 every 3 seconds)...\n")

    for i in range(num_scans):
        try:
            networks = scan_networks()
        except (PermissionError, RuntimeError) as exc:
            print(f"Scan error: {exc}")
            raise SystemExit(1)

        results = pipeline.process(networks)
        print(f"[Scan {i + 1}/{num_scans}] {len(networks)} networks found, "
              f"{len(results)} stable after filtering")

        if i < num_scans - 1:
            time.sleep(3)

    # Final processed table
    results = pipeline.process(networks) if pipeline.scan_count == num_scans else results
    if not results:
        print("\nNo stable networks after filtering (need 3+ appearances in 5 scans).")
        raise SystemExit(0)

    hdr = (f"{'SSID':<20} {'BSSID':<19} {'Smoothed':>9} {'Raw Avg':>8} "
           f"{'Stab':>5} {'Spike':>6} {'Freq':>10} {'Ch':>4}")
    print(f"\n{'=' * len(hdr)}")
    print("PROCESSED WiFi SCAN RESULTS")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ssid = r.ssid[:19] if r.ssid else "(hidden)"
        freq = f"{r.frequency_mhz:.0f} MHz" if r.frequency_mhz else "N/A"
        spike_flag = "SPIKE" if r.spike_detected else ""
        ch = str(r.channel) if r.channel else "N/A"
        print(f"{ssid:<20} {r.bssid:<19} {r.rssi_smoothed:>8.1f}  "
              f"{r.rssi_raw_avg:>7.1f}  {r.stability_score:>5.2f} {spike_flag:>6} "
              f"{freq:>10} {ch:>4}")
    print(f"\n{len(results)} stable network(s) — {datetime.now():%Y-%m-%d %H:%M:%S}")