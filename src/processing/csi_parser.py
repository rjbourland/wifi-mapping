"""Parse CSI data from multiple hardware formats."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..utils.data_formats import CSISample

logger = logging.getLogger(__name__)


class CSIParser:
    """Parses CSI data from multiple hardware formats into a unified CSISample.

    Supported formats:
    - Intel 5300 (.dat files from Linux CSI Tool)
    - Nexmon (.pcap files)
    - ESP32 (CSV/text over serial)
    - FeitCSI (binary from AX210)
    - CSIKit (any format via the CSIKit library)
    """

    def __init__(self, hardware: str = "esp32"):
        """Initialize parser for a specific hardware format.

        Args:
            hardware: Hardware format to parse. One of:
                'esp32', 'intel5300', 'nexmon', 'feitcsi', 'csikit'.
        """
        self.hardware = hardware.lower()
        if self.hardware not in ("esp32", "intel5300", "nexmon", "feitcsi", "csikit"):
            raise ValueError(f"Unknown hardware format: {hardware}. "
                             f"Supported: esp32, intel5300, nexmon, feitcsi, csikit")

    def parse_file(self, filepath: Path) -> list[CSISample]:
        """Parse a CSI data file into a list of CSISample objects.

        Args:
            filepath: Path to the CSI data file.

        Returns:
            List of CSISample objects.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSI file not found: {filepath}")

        if self.hardware == "csikit":
            return self._parse_csikit(filepath)
        elif self.hardware == "esp32":
            return self._parse_esp32_csv(filepath)
        elif self.hardware == "intel5300":
            return self._parse_intel5300(filepath)
        elif self.hardware == "nexmon":
            return self._parse_nexmon(filepath)
        elif self.hardware == "feitcsi":
            return self._parse_feitcsi(filepath)

        return []

    def parse_esp32_line(self, line: str, anchor_id: str = "esp32") -> Optional[CSISample]:
        """Parse a single ESP32 CSI line from serial output.

        ESP32 CSI output format (typical):
        CSI_DATA,<timestamp>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,<smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec>,<sgi>,<noise_floor>,<antenna>,<channel>,<secondary_channel>,<len>,<csi_data>,<rx_end_ts>

        Args:
            line: Raw CSV line from ESP32 serial output.
            anchor_id: Identifier for this anchor node.

        Returns:
            CSISample or None if parsing fails.
        """
        try:
            parts = line.strip().split(",")
            if len(parts) < 19 or parts[0] != "CSI_DATA":
                return None

            rssi = float(parts[2])
            bandwidth = int(parts[6])
            channel = int(parts[15])
            noise_floor = float(parts[13])
            csi_len = int(parts[16])
            csi_raw = parts[17]

            # Parse CSI hex string into complex numbers
            # ESP32 CSI: interleaved real/imaginary bytes in hex
            csi_bytes = bytes.fromhex(csi_raw)
            num_subcarriers = csi_len // 2  # Each subcarrier has I and Q

            csi_matrix = np.zeros((1, num_subcarriers), dtype=complex)
            for i in range(num_subcarriers):
                real = csi_bytes[2 * i] if 2 * i < len(csi_bytes) else 0
                imag = csi_bytes[2 * i + 1] if 2 * i + 1 < len(csi_bytes) else 0
                csi_matrix[0, i] = complex(real, imag)

            return CSISample(
                timestamp=datetime.now(),
                anchor_id=anchor_id,
                channel=channel,
                bandwidth=bandwidth,
                num_subcarriers=num_subcarriers,
                csi_matrix=csi_matrix,
                rssi=[rssi],
                noise_floor=noise_floor,
                carrier_freq=2.412e9 + (channel - 1) * 0.005e9 if channel <= 13 else 5.0e9,
            )
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse ESP32 CSI line: {e}")
            return None

    def _parse_esp32_csv(self, filepath: Path) -> list[CSISample]:
        """Parse ESP32 CSV file containing CSI data.

        Args:
            filepath: Path to the CSV file.

        Returns:
            List of CSISample objects.
        """
        samples = []
        with open(filepath) as f:
            for line in f:
                sample = self.parse_esp32_line(line)
                if sample is not None:
                    samples.append(sample)
        logger.info(f"Parsed {len(samples)} ESP32 CSI samples from {filepath}")
        return samples

    def _parse_intel5300(self, filepath: Path) -> list[CSISample]:
        """Parse Intel 5300 .dat file.

        The Intel 5300 CSI tool outputs binary .dat files with:
        - 3 antennas x 30 subcarrier groups
        - 8-bit quantization per I/Q component
        """
        # Intel 5300 binary format parsing
        # This is a simplified parser — use CSIKit for production
        logger.warning("Intel 5300 parser is a placeholder. Use CSIKit for production parsing.")
        return self._parse_csikit(filepath)

    def _parse_nexmon(self, filepath: Path) -> list[CSISample]:
        """Parse Nexmon PCAP file containing CSI data.

        Nexmon outputs PCAP files with custom CSI headers.
        """
        logger.warning("Nexmon parser is a placeholder. Use CSIKit for production parsing.")
        return self._parse_csikit(filepath)

    def _parse_feitcsi(self, filepath: Path) -> list[CSISample]:
        """Parse FeitCSI binary output from Intel AX210.

        FeitCSI outputs custom binary format with full 802.11ax CSI.
        """
        logger.warning("FeitCSI parser is a placeholder. Use CSIKit for production parsing.")
        return self._parse_csikit(filepath)

    def _parse_csikit(self, filepath: Path) -> list[CSISample]:
        """Parse CSI file using the CSIKit library.

        CSIKit supports Intel 5300, Atheros, Nexmon, ESP32, and PicoScenes formats.
        Requires: pip install csikit
        """
        try:
            from csikit import CSIData
        except ImportError:
            logger.error(
                "CSIKit is required for this parser. Install with: pip install csikit"
            )
            return []

        try:
            csi_data = CSIData(str(filepath))
            samples = []
            for i in range(len(csi_data.timestamps)):
                csi_matrix = csi_data.get_csi(i)  # shape: (num_antennas, num_subcarriers)
                samples.append(CSISample(
                    timestamp=datetime.fromtimestamp(csi_data.timestamps[i]),
                    anchor_id="unknown",
                    channel=csi_data.channel if hasattr(csi_data, 'channel') else 0,
                    bandwidth=csi_data.bandwidth if hasattr(csi_data, 'bandwidth') else 20,
                    num_subcarriers=csi_matrix.shape[-1],
                    csi_matrix=csi_matrix,
                    rssi=[csi_data.rssi[i]] if hasattr(csi_data, 'rssi') else None,
                    noise_floor=getattr(csi_data, 'noise_floor', None),
                    carrier_freq=getattr(csi_data, 'carrier_freq', None),
                ))
            logger.info(f"Parsed {len(samples)} CSI samples from {filepath} via CSIKit")
            return samples
        except Exception as e:
            logger.error(f"CSIKit parsing failed: {e}")
            return []