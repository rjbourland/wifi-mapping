"""Data loading helpers for the WiFi Mapping dashboard."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_anchors, get_room_dimensions
from src.utils.data_formats import AnchorPosition
from src.localization.trilateration import TrilaterationSolver
from src.localization.fingerprinting import KNNFingerprinting
from src.localization.kalman_filter import KalmanFilter
from src.mapping.point_cloud import PointCloudAccumulator
from src.detection.motion_detector import MotionDetector
from src.detection.breathing_detector import BreathingDetector
from src.detection.gait_classifier import GaitClassifier


def init_session_state():
    """Initialize Streamlit session state with default values."""
    import streamlit as st

    defaults = {
        "anchors": load_anchors(),
        "room_dimensions": get_room_dimensions(),
        "trilateration_solver": TrilaterationSolver(),
        "kalman_filter": KalmanFilter(),
        "fingerprinting": KNNFingerprinting(),
        "point_cloud": PointCloudAccumulator(),
        "motion_detector": MotionDetector(),
        "breathing_detector": BreathingDetector(),
        "gait_classifier": GaitClassifier(),
        "localized_positions": [],
        "motion_events": [],
        "rssi_history": [],
        "csi_samples": [],
        "collection_active": False,
        "simulation_mode": True,
        "last_position": None,
        "last_method": "none",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Wire anchors into solver
    st.session_state.trilateration_solver.set_anchors(st.session_state.anchors)


def load_anchors() -> list[AnchorPosition]:
    """Load anchor positions from config.

    Returns:
        List of AnchorPosition objects.
    """
    try:
        anchors_config = load_config("anchors")
    except FileNotFoundError:
        # Return defaults if config missing
        anchors_config = {
            "anchors": {
                "anchor_1": {"position": [0.0, 0.0, 2.5], "height": "ceiling", "hardware": "esp32_s3", "ip": "", "channel": 6, "bandwidth": 20},
                "anchor_2": {"position": [4.0, 0.0, 1.3], "height": "mid", "hardware": "esp32_s3", "ip": "", "channel": 6, "bandwidth": 20},
                "anchor_3": {"position": [4.0, 4.0, 0.3], "height": "floor", "hardware": "esp32_s3", "ip": "", "channel": 6, "bandwidth": 20},
                "anchor_4": {"position": [0.0, 4.0, 2.5], "height": "ceiling", "hardware": "esp32_s3", "ip": "", "channel": 6, "bandwidth": 20},
            }
        }

    anchors = []
    for aid, info in anchors_config.get("anchors", {}).items():
        anchors.append(AnchorPosition(
            anchor_id=aid,
            position=np.array(info["position"]),
            height=info.get("height", "mid"),
            hardware=info.get("hardware", "unknown"),
            ip=info.get("ip", ""),
            channel=info.get("channel", 6),
            bandwidth=info.get("bandwidth", 20),
        ))
    return anchors


def generate_synthetic_rssi(
    anchors: list[AnchorPosition],
    true_position: np.ndarray,
    noise_std: float = 3.0,
    rssi_d0: float = -30.0,
    n: float = 3.0,
) -> dict[str, float]:
    """Generate synthetic RSSI measurements for testing.

    Args:
        anchors: List of anchor positions.
        true_position: True (x, y, z) position of the target.
        noise_std: Standard deviation of shadow fading noise (dB).
        rssi_d0: Reference RSSI at 1m.
        n: Path-loss exponent.

    Returns:
        Dict mapping anchor_id -> RSSI (dBm).
    """
    from src.utils.math_utils import path_loss_rssi

    measurements = {}
    for anchor in anchors:
        distance = np.linalg.norm(true_position - anchor.position)
        rssi = path_loss_rssi(distance, rssi_d0, n)
        rssi += np.random.randn() * noise_std
        measurements[anchor.anchor_id] = rssi
    return measurements


def generate_synthetic_csi(
    num_packets: int = 50,
    num_antennas: int = 2,
    num_subcarriers: int = 52,
    motion: bool = False,
) -> np.ndarray:
    """Generate synthetic CSI data for testing.

    Args:
        num_packets: Number of CSI packets.
        num_antennas: Number of antennas.
        num_subcarriers: Number of subcarriers.
        motion: Whether to simulate motion (large variance).

    Returns:
        Complex CSI array, shape (num_packets, num_antennas, num_subcarriers).
    """
    base_amp = 1.0
    if motion:
        t = np.linspace(0, 4 * np.pi, num_packets)
        amplitude_variation = 1.0 + 2.0 * np.sin(t)[:, np.newaxis, np.newaxis]
    else:
        amplitude_variation = 1.0

    csi = amplitude_variation * (
        np.random.randn(num_packets, num_antennas, num_subcarriers)
        + 1j * np.random.randn(num_packets, num_antennas, num_subcarriers)
    ) * base_amp + base_amp

    return csi


def positions_to_dataframe(positions: list) -> pd.DataFrame:
    """Convert a list of LocalizedPosition objects to a DataFrame.

    Args:
        positions: List of LocalizedPosition objects.

    Returns:
        DataFrame with columns: timestamp, x, y, z, method, confidence, anchors_used.
    """
    if not positions:
        return pd.DataFrame(columns=["timestamp", "x", "y", "z", "method", "confidence"])

    records = []
    for p in positions:
        records.append({
            "timestamp": p.timestamp,
            "x": p.position[0],
            "y": p.position[1],
            "z": p.position[2] if len(p.position) > 2 else 0.0,
            "method": p.method,
            "confidence": p.confidence,
            "anchors_used": ", ".join(p.anchors_used),
        })
    return pd.DataFrame(records)


def rssi_history_to_dataframe(rssi_history: list) -> pd.DataFrame:
    """Convert RSSI history to a DataFrame.

    Args:
        rssi_history: List of dicts with 'timestamp', 'anchor_id', 'rssi'.

    Returns:
        DataFrame with columns: timestamp, anchor_id, rssi.
    """
    if not rssi_history:
        return pd.DataFrame(columns=["timestamp", "anchor_id", "rssi"])
    return pd.DataFrame(rssi_history)