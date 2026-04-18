"""Hardware abstraction for Streamlit GUI.

Wraps RSSIScanner and CSICollector for use in the dashboard,
handling platform differences and permission errors gracefully.
"""

import logging
from typing import Optional

from src.collection.rssi_scanner import RSSIScanner, NetworkResult
from src.processing.process_rssi import RSSIPipeline, ProcessedScan

logger = logging.getLogger(__name__)


class HardwareManager:
    """Manages real hardware connections for the GUI.

    Provides a clean interface for starting/stopping data collection
    from WiFi adapters and CSI-capable hardware.
    """

    def __init__(self):
        self.rssi_scanner: Optional[RSSIScanner] = None
        self.pipeline = RSSIPipeline(window_size=5, min_seen=3)
        self._active = False

    def start_rssi(self, interface: str = "") -> bool:
        """Start RSSI scanning on the given interface.

        Args:
            interface: Network interface name (e.g. 'wlan0' on Linux).
                       Empty string for auto-detect.

        Returns:
            True if scanner started successfully, False otherwise.
        """
        try:
            self.rssi_scanner = RSSIScanner(interface=interface)
            self._active = True
            logger.info("RSSI scanner started on interface '%s'", interface or "auto")
            return True
        except Exception as e:
            logger.error("Failed to start RSSI scanner: %s", e)
            self.rssi_scanner = None
            return False

    def scan_rssi(self) -> list[NetworkResult]:
        """Perform a single RSSI scan.

        Returns:
            List of NetworkResult objects, or empty list on failure.
        """
        if self.rssi_scanner is None:
            return []

        try:
            return self.rssi_scanner.scan()
        except PermissionError as e:
            logger.error("Permission denied for RSSI scan: %s", e)
            return []
        except Exception as e:
            logger.error("RSSI scan failed: %s", e)
            return []

    def scan_and_process(self) -> list[ProcessedScan]:
        """Scan and process RSSI data in one step.

        Returns:
            List of ProcessedScan objects (stable, above min_seen threshold).
        """
        raw = self.scan_rssi()
        if not raw:
            return []
        return self.pipeline.process(raw)

    def stop(self):
        """Stop all collection."""
        self._active = False
        self.rssi_scanner = None
        self.pipeline = RSSIPipeline(window_size=5, min_seen=3)

    @property
    def is_active(self) -> bool:
        return self._active and self.rssi_scanner is not None