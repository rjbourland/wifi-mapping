"""CSI data collection from WiFi nodes via UDP or serial."""

import socket
import struct
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.data_formats import CSISample

logger = logging.getLogger(__name__)


class CSICollector:
    """Collects CSI data from WiFi anchor nodes.

    Supports two collection modes:
    - UDP: For Intel AX210 (FeitCSI) or Nexmon nodes that stream CSI over network
    - Serial: For ESP32 nodes that output CSI over UART

    Data is stored in Parquet format for efficient columnar access.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize the CSI collector.

        Args:
            config: Collection configuration dict (from collection.yaml).
                     If None, loads from default config.
        """
        if config is None:
            from ..utils.config import load_config
            config = load_config("collection")

        self.config = config
        self.data_dir = Path(config.get("data_dir", "./data/raw"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._udp_socket: Optional[socket.socket] = None
        self._serial_port = None
        self._running = False

    def start_udp_listener(self, port: int = 5500, interface: str = "0.0.0.0"):
        """Start a UDP listener to receive CSI data from network nodes.

        Args:
            port: UDP port to listen on.
            interface: Network interface to bind to.
        """
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.bind((interface, port))
        self._udp_socket.settimeout(5.0)
        self._running = True
        logger.info(f"UDP listener started on {interface}:{port}")

    def start_serial_listener(self, port: str = "", baud_rate: int = 921600):
        """Start a serial listener to receive CSI data from ESP32 nodes.

        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0', 'COM3').
                   If empty, auto-detects the first available port.
            baud_rate: Serial baud rate.
        """
        import serial
        import serial.tools.list_ports

        if not port:
            ports = list(serial.tools.list_ports.comports())
            if not ports:
                raise RuntimeError("No serial ports found")
            port = ports[0].device
            logger.info(f"Auto-detected serial port: {port}")

        self._serial_port = serial.Serial(port, baud_rate, timeout=5.0)
        self._running = True
        logger.info(f"Serial listener started on {port} at {baud_rate} baud")

    def collect_samples(
        self,
        num_samples: int = 100,
        anchor_id: str = "unknown",
        channel: int = 6,
        bandwidth: int = 20,
    ) -> list[CSISample]:
        """Collect CSI samples from the configured source.

        Args:
            num_samples: Number of samples to collect.
            anchor_id: Identifier for the anchor node.
            channel: WiFi channel number.
            bandwidth: Channel bandwidth in MHz.

        Returns:
            List of CSISample objects.
        """
        samples = []
        for i in range(num_samples):
            sample = self._read_one_sample(anchor_id, channel, bandwidth)
            if sample is not None:
                samples.append(sample)
            if (i + 1) % 10 == 0:
                logger.info(f"Collected {i + 1}/{num_samples} samples")

        return samples

    def save_samples(self, samples: list[CSISample], filename: Optional[str] = None):
        """Save collected CSI samples to Parquet file.

        Args:
            samples: List of CSISample objects.
            filename: Output filename. Auto-generated if None.
        """
        if not samples:
            logger.warning("No samples to save")
            return

        if filename is None:
            ts = samples[0].timestamp.strftime("%Y%m%d_%H%M%S")
            anchor = samples[0].anchor_id
            filename = f"csi_{anchor}_{ts}.parquet"

        filepath = self.data_dir / filename

        records = []
        for s in samples:
            record = {
                "timestamp": s.timestamp,
                "anchor_id": s.anchor_id,
                "channel": s.channel,
                "bandwidth": s.bandwidth,
                "num_subcarriers": s.num_subcarriers,
                "csi_real": s.csi_matrix.real.flatten().tolist(),
                "csi_imag": s.csi_matrix.imag.flatten().tolist(),
                "csi_shape": list(s.csi_matrix.shape),
                "rssi": s.rssi,
                "noise_floor": s.noise_floor,
                "carrier_freq": s.carrier_freq,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.to_parquet(filepath, engine="pyarrow")
        logger.info(f"Saved {len(samples)} samples to {filepath}")

    def _read_one_sample(
        self, anchor_id: str, channel: int, bandwidth: int
    ) -> Optional[CSISample]:
        """Read a single CSI sample from the configured source.

        This is a placeholder that generates synthetic data for testing.
        Override with actual CSI reading logic for your hardware.
        """
        # Placeholder: generate synthetic CSI for testing
        num_antennas = 2
        num_subcarriers = 52 if bandwidth == 20 else 242

        csi = np.random.randn(num_antennas, num_subcarriers) + 1j * np.random.randn(
            num_antennas, num_subcarriers
        )

        return CSISample(
            timestamp=datetime.now(),
            anchor_id=anchor_id,
            channel=channel,
            bandwidth=bandwidth,
            num_subcarriers=num_subcarriers,
            csi_matrix=csi,
            rssi=[-40.0, -42.0],
            noise_floor=-95.0,
            carrier_freq=2.412e9 if channel <= 13 else 5.18e9,
        )

    def stop(self):
        """Stop all listeners."""
        self._running = False
        if self._udp_socket:
            self._udp_socket.close()
            self._udp_socket = None
        if self._serial_port:
            self._serial_port.close()
            self._serial_port = None
        logger.info("CSI collector stopped")