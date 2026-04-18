"""Pipeline bridge — centralizes sim-vs-live decision for all GUI pages.

Every page calls collect_rssi() and run_localization() instead of
directly using generate_synthetic_rssi() or the legacy solver.localize() API.
This ensures both simulation and live paths exercise the same code.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from src.collection.rssi_scanner import RSSIScanner, NetworkResult
from src.processing.process_rssi import RSSIPipeline, ProcessedScan
from src.localization.trilateration import TrilaterationSolver, Position
from src.localization.kalman_filter import KalmanFilter, SmoothedPosition
from src.localization.fingerprinting import KNNFingerprinting
from src.utils.data_formats import AnchorPosition, LocalizedPosition
from src.utils.math_utils import path_loss_rssi


def generate_synthetic_scans(
    anchors: list[AnchorPosition],
    true_position: np.ndarray,
    pipeline: RSSIPipeline,
    noise_std: float = 3.0,
    rssi_d0: float = -30.0,
    n: float = 3.0,
    num_scans: int = 5,
) -> list[ProcessedScan]:
    """Generate synthetic ProcessedScan list by running simulated RSSI
    measurements through the real RSSIPipeline.

    Produces num_scans slightly-jittered position samples, converts each
    to NetworkResult objects via path-loss model, feeds them through the
    pipeline, and returns the accumulated stable scans.

    Args:
        anchors: List of anchor positions.
        true_position: True (x, y, z) position.
        pipeline: RSSIPipeline instance (will be reset if needed).
        noise_std: Shadow fading noise std (dBm).
        rssi_d0: Reference RSSI at 1m.
        n: Path-loss exponent.
        num_scans: Number of simulated scans to accumulate.

    Returns:
        List of ProcessedScan objects with matching anchor BSSIDs.
    """
    all_scans = []

    for i in range(num_scans):
        # Slight position jitter to simulate movement/noise
        jitter = np.random.randn(3) * 0.05
        pos = true_position + jitter

        networks = []
        for anchor in anchors:
            distance = np.linalg.norm(pos - anchor.position)
            rssi = path_loss_rssi(distance, rssi_d0, n)
            rssi += np.random.randn() * noise_std

            # Use anchor_id as both SSID and BSSID for simulation
            freq = 2412 + (anchor.channel - 1) * 5 if anchor.channel <= 13 else 5180 + (anchor.channel - 36) * 5

            networks.append(NetworkResult(
                ssid=f"WiFi-{anchor.anchor_id}",
                bssid=anchor.anchor_id,
                rssi_dbm=rssi,
                frequency_mhz=freq,
                channel=anchor.channel,
                timestamp=datetime.now(),
            ))

        processed = pipeline.process(networks)

        # Filter to only scans matching our anchors
        anchor_ids = {a.anchor_id for a in anchors}
        matching = [s for s in processed if s.bssid in anchor_ids]
        all_scans.extend(matching)

    return all_scans


def collect_rssi(session_state) -> list[ProcessedScan]:
    """Collect RSSI data — simulation or live mode.

    In simulation mode, generates synthetic data at a random position.
    In live mode, calls RSSIScanner.scan() and processes through RSSIPipeline.

    Args:
        session_state: Streamlit session state dict-like object. Expected keys:
            simulation_mode, anchors, rssi_pipeline, room_dimensions, and
            optionally rssi_scanner (for live mode).

    Returns:
        List of ProcessedScan objects ready for localization.
    """
    anchors = session_state.anchors
    pipeline = session_state.rssi_pipeline
    room = session_state.room_dimensions

    if session_state.simulation_mode:
        # Random position inside room bounds
        true_pos = np.array([
            np.random.uniform(0.5, room["length_x"] - 0.5),
            np.random.uniform(0.5, room["width_y"] - 0.5),
            np.random.uniform(0.3, room["height_z"] - 0.3),
        ])
        return generate_synthetic_scans(anchors, true_pos, pipeline)
    else:
        # Live mode — use real scanner
        if session_state.get("rssi_scanner") is None:
            session_state.rssi_scanner = RSSIScanner()

        networks = session_state.rssi_scanner.scan()
        if not networks:
            return []

        processed = pipeline.process(networks)

        # Filter to anchors
        anchor_ids = {a.anchor_id for a in anchors}
        anchor_ips = {a.ip for a in anchors if a.ip}
        matching = [s for s in processed
                     if s.bssid in anchor_ids or s.bssid in anchor_ips]
        return matching


def run_localization(
    session_state,
    scans: list[ProcessedScan],
    method: str = "trilateration",
    use_kalman: bool = True,
) -> tuple[Optional[Position], Optional[LocalizedPosition], Optional[SmoothedPosition]]:
    """Run localization on ProcessedScan data.

    Args:
        session_state: Streamlit session state.
        scans: List of ProcessedScan objects from collect_rssi().
        method: "trilateration" or "fingerprinting".
        use_kalman: Whether to apply Kalman filtering.

    Returns:
        Tuple of (Position, LocalizedPosition, SmoothedPosition).
        Position is set for trilateration, LocalizedPosition for fingerprinting.
        SmoothedPosition is set if use_kalman is True and Position is available.
        May be (None, None, None) if insufficient data.
    """
    solver: TrilaterationSolver = session_state.trilateration_solver
    anchors: list[AnchorPosition] = session_state.anchors
    kalman: KalmanFilter = session_state.kalman_filter
    fingerprinting: KNNFingerprinting = session_state.fingerprinting

    position = None
    localized = None
    smoothed = None

    if not scans:
        return position, localized, smoothed

    if method == "trilateration":
        try:
            position = solver.localize_from_scans(scans, anchors)
            if use_kalman:
                smoothed = kalman.update_position(position)
        except ValueError as e:
            import logging
            logging.getLogger(__name__).warning("Trilateration failed: %s", e)

    elif method == "fingerprinting":
        # Convert ProcessedScan → {bssid: rssi_smoothed} dict
        rssi_dict = {s.bssid: s.rssi_smoothed for s in scans}
        try:
            localized = fingerprinting.localize(rssi_dict)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Fingerprinting failed: %s", e)

    return position, localized, smoothed


def rssi_dict_from_scans(scans: list[ProcessedScan]) -> dict[str, float]:
    """Convert ProcessedScan list to {bssid: rssi_smoothed} dict.

    Useful for fingerprinting and for legacy rssi_history logging.
    """
    return {s.bssid: s.rssi_smoothed for s in scans}