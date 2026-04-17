"""Data schema definitions for CSI and RSSI samples."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class RSSISample:
    """A single RSSI measurement from one anchor."""

    timestamp: datetime
    anchor_id: str
    ssid: str
    bssid: str
    rssi_dbm: float  # Signal strength in dBm (negative)

    @property
    def rssi_linear(self) -> float:
        """Convert dBm to linear power (mW)."""
        return 10 ** (self.rssi_dbm / 10.0)


@dataclass
class CSISample:
    """A single CSI measurement from one anchor.

    Attributes:
        timestamp: Measurement time.
        anchor_id: Which anchor captured this sample.
        channel: WiFi channel number.
        bandwidth: Channel bandwidth in MHz.
        num_subcarriers: Number of CSI subcarriers.
        csi_matrix: Complex CSI matrix, shape (num_antennas, num_subcarriers).
        rssi: Per-antenna RSSI in dBm.
        noise_floor: Noise floor in dBm.
        carrier_freq: Carrier frequency in Hz.
    """

    timestamp: datetime
    anchor_id: str
    channel: int
    bandwidth: int
    num_subcarriers: int
    csi_matrix: np.ndarray  # Complex, shape (num_antennas, num_subcarriers)
    rssi: Optional[list[float]] = None  # Per-antenna RSSI
    noise_floor: Optional[float] = None
    carrier_freq: Optional[float] = None

    @property
    def amplitude(self) -> np.ndarray:
        """CSI amplitude (magnitude of complex values)."""
        return np.abs(self.csi_matrix)

    @property
    def phase(self) -> np.ndarray:
        """CSI phase (angle of complex values), in radians."""
        return np.angle(self.csi_matrix)

    @property
    def unwrapped_phase(self) -> np.ndarray:
        """Unwrapped phase to remove 2pi jumps."""
        return np.unwrap(self.phase, axis=-1)


@dataclass
class AnchorPosition:
    """3D position of an anchor node."""

    anchor_id: str
    position: np.ndarray  # (x, y, z) in meters
    height: str  # 'floor', 'mid', or 'ceiling'
    hardware: str  # 'esp32_s3', 'ax210', 'nexmon_pi'
    ip: str
    channel: int
    bandwidth: int


@dataclass
class GroundTruthPoint:
    """A ground-truth position measurement for calibration."""

    timestamp: datetime
    position: np.ndarray  # (x, y, z) in meters
    label: str = ""  # Optional label (e.g., "desk_center", "doorway")
    los_from: list[str] = field(default_factory=list)  # Anchor IDs with LoS
    nlos_from: list[str] = field(default_factory=list)  # Anchor IDs with NLoS


@dataclass
class LocalizedPosition:
    """Result of a localization estimate."""

    timestamp: datetime
    position: np.ndarray  # (x, y, z) in meters
    method: str  # 'trilateration', 'fingerprinting', 'aoa', 'hybrid'
    error_2d: Optional[float] = None  # Horizontal error in meters (if ground truth known)
    error_3d: Optional[float] = None  # Full 3D error in meters (if ground truth known)
    confidence: Optional[float] = None  # 0-1 confidence score
    anchors_used: list[str] = field(default_factory=list)