"""WiFi RSSI scanning for nearby access points.

Supports Windows (netsh) and Linux (iw / iwlist) platforms.
Returns network details including SSID, BSSID, RSSI, frequency, channel,
and timestamp for each discovered network.
"""

import logging
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NetworkResult:
    """A single WiFi network discovered during a scan."""

    ssid: str
    bssid: str
    rssi_dbm: float
    frequency_mhz: float
    channel: int
    timestamp: datetime
    radio_type: str = ""


# ---------------------------------------------------------------------------
# Frequency ↔ channel helpers
# ---------------------------------------------------------------------------

_FREQ_5GHZ = {
    36: 5180, 40: 5200, 44: 5220, 48: 5240,
    52: 5260, 56: 5280, 60: 5300, 64: 5320,
    100: 5500, 104: 5520, 108: 5540, 112: 5560,
    116: 5580, 120: 5600, 124: 5620, 128: 5640,
    132: 5660, 136: 5680, 140: 5700, 144: 5720,
    149: 5745, 153: 5765, 157: 5785, 161: 5805, 165: 5825,
}

_FREQ_6GHZ = {
    1: 5955, 5: 5975, 9: 5995, 13: 6015, 17: 6035,
    21: 6055, 25: 6075, 29: 6095, 33: 6115, 37: 6135,
    41: 6155, 45: 6175, 49: 6195, 53: 6215, 57: 6235,
    61: 6255, 65: 6275, 69: 6295, 73: 6315, 77: 6335,
    81: 6355, 85: 6375, 89: 6395, 93: 6415, 97: 6435,
    101: 6455, 105: 6475, 109: 6495, 113: 6515, 117: 6535,
    121: 6555, 125: 6575, 129: 6595, 133: 6615, 137: 6635,
    141: 6655, 145: 6675, 149: 6695, 153: 6715, 157: 6735,
    161: 6755, 165: 6775, 169: 6795, 173: 6815, 177: 6835,
    181: 6855, 185: 6875, 189: 6895, 193: 6915, 197: 6935,
    201: 6955, 205: 6975, 209: 6995, 213: 7015, 217: 7035,
    221: 7055, 225: 7075, 229: 7095, 233: 7115,
}


def _channel_to_freq(channel: int) -> float:
    """Convert WiFi channel number to frequency in MHz."""
    if 1 <= channel <= 13:
        return 2407.0 + channel * 5
    if channel == 14:
        return 2484.0
    if channel in _FREQ_5GHZ:
        return float(_FREQ_5GHZ[channel])
    if channel in _FREQ_6GHZ:
        return float(_FREQ_6GHZ[channel])
    return 0.0


def _freq_to_channel(freq_mhz: float) -> int:
    """Convert frequency in MHz to WiFi channel number."""
    if 2412 <= freq_mhz <= 2484:
        return 14 if freq_mhz == 2484 else int((freq_mhz - 2407) / 5)
    for lookup in (_FREQ_5GHZ, _FREQ_6GHZ):
        for ch, f in lookup.items():
            if abs(f - freq_mhz) < 2:
                return ch
    return 0


def _signal_pct_to_dbm(pct: int) -> float:
    """Convert Windows signal quality percentage to approximate dBm.

    Uses the standard approximation: dBm ≈ (quality / 2) − 100.
    """
    return round((pct / 2.0) - 100.0, 1)


# ---------------------------------------------------------------------------
# Windows scanner (netsh)
# ---------------------------------------------------------------------------

def _scan_windows() -> list[NetworkResult]:
    """Scan WiFi networks on Windows using ``netsh wlan show networks``."""
    timestamp = datetime.now()

    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"netsh command failed: {exc.stderr.strip() or exc}"
        ) from exc
    except subprocess.TimeoutExpired:
        raise RuntimeError("netsh command timed out after 30 seconds")
    except FileNotFoundError:
        raise RuntimeError("netsh not found — this command requires Windows")

    output = result.stdout
    if not output.strip():
        raise RuntimeError(
            "netsh returned empty output — no wireless interface may be available"
        )

    networks: list[NetworkResult] = []
    cur_ssid: str | None = None
    cur_bssid: str | None = None
    cur_signal: int | None = None
    cur_channel: int | None = None
    cur_radio: str = ""

    def flush() -> None:
        nonlocal cur_bssid, cur_signal, cur_channel, cur_radio
        if cur_ssid is not None and cur_bssid is not None:
            rssi = (
                _signal_pct_to_dbm(cur_signal) if cur_signal is not None else 0.0
            )
            freq = _channel_to_freq(cur_channel) if cur_channel else 0.0
            networks.append(
                NetworkResult(
                    ssid=cur_ssid,
                    bssid=cur_bssid,
                    rssi_dbm=rssi,
                    frequency_mhz=freq,
                    channel=cur_channel or 0,
                    timestamp=timestamp,
                    radio_type=cur_radio,
                )
            )
        cur_bssid = None
        cur_signal = None
        cur_channel = None
        cur_radio = ""

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # SSID header: "SSID 1 : NetworkName" or "SSID : NetworkName"
        m = re.match(r"^SSID\s+(?:\d+\s+)?:\s*(.+)$", stripped)
        if m:
            flush()
            cur_ssid = m.group(1).strip() or "Unknown"
            continue

        # BSSID line: "BSSID 1 : aa:bb:cc:dd:ee:ff"
        m = re.match(r"^\s*BSSID\s+\d+\s*:\s*([0-9a-fA-F:]+)", line)
        if m:
            flush()
            cur_bssid = m.group(1).strip()
            continue

        # Only parse sub-fields when inside a BSSID block
        if cur_ssid is None or cur_bssid is None:
            continue

        m = re.match(r"^\s*Signal\s*:\s*(\d+)%", line)
        if m:
            cur_signal = int(m.group(1))
            continue

        m = re.match(r"^\s*Channel\s*:\s*(\d+)", line)
        if m:
            cur_channel = int(m.group(1))
            continue

        m = re.match(r"^\s*Radio type\s*:\s*(.+)$", line)
        if m:
            cur_radio = m.group(1).strip()
            continue

    flush()

    if not networks:
        logger.warning("netsh completed but no networks were parsed")

    return networks


# ---------------------------------------------------------------------------
# Linux scanners (iw / iwlist)
# ---------------------------------------------------------------------------

def _parse_iwlist(output: str, timestamp: datetime) -> list[NetworkResult]:
    """Parse ``iwlist <iface> scan`` output."""
    networks: list[NetworkResult] = []
    cur: dict | None = None

    for line in output.splitlines():
        stripped = line.strip()

        m = re.match(r"Cell\s+\d+\s+-\s+Address:\s*([0-9a-fA-F:]+)", stripped)
        if m:
            if cur and cur.get("ssid"):
                networks.append(_from_dict(cur, timestamp))
            cur = {"bssid": m.group(1), "ssid": "", "rssi_dbm": 0.0,
                   "frequency_mhz": 0.0, "channel": 0, "radio_type": ""}
            continue

        if cur is None:
            continue

        m = re.match(r'ESSID:"(.*)"', stripped)
        if m:
            cur["ssid"] = m.group(1) or "Unknown"
            continue

        m = re.match(r"Frequency:([\d.]+)\s*GHz\s*(?:\(Channel\s+(\d+)\))?", stripped)
        if m:
            cur["frequency_mhz"] = float(m.group(1)) * 1000
            cur["channel"] = int(m.group(2)) if m.group(2) else _freq_to_channel(cur["frequency_mhz"])
            continue

        m = re.match(r"Frequency:([\d.]+)\s*MHz", stripped)
        if m:
            cur["frequency_mhz"] = float(m.group(1))
            cur["channel"] = _freq_to_channel(cur["frequency_mhz"])
            continue

        m = re.match(r"Signal level=(-?\d+)\s*dBm", stripped)
        if m:
            cur["rssi_dbm"] = float(m.group(1))
            continue

    if cur and cur.get("ssid"):
        networks.append(_from_dict(cur, timestamp))
    return networks


def _parse_iw(output: str, timestamp: datetime) -> list[NetworkResult]:
    """Parse ``iw dev <iface> scan`` output."""
    networks: list[NetworkResult] = []
    cur: dict | None = None

    for line in output.splitlines():
        stripped = line.strip()

        m = re.match(r"BSS\s+([0-9a-fA-F:]+)", stripped)
        if m:
            if cur and cur.get("ssid"):
                networks.append(_from_dict(cur, timestamp))
            cur = {"bssid": m.group(1), "ssid": "", "rssi_dbm": 0.0,
                   "frequency_mhz": 0.0, "channel": 0, "radio_type": ""}
            continue

        if cur is None:
            continue

        m = re.match(r"SSID:\s*(.+)", stripped)
        if m:
            cur["ssid"] = m.group(1).strip() or "Unknown"
            continue

        m = re.match(r"freq:\s*(\d+)", stripped)
        if m:
            cur["frequency_mhz"] = float(m.group(1))
            cur["channel"] = _freq_to_channel(cur["frequency_mhz"])
            continue

        m = re.match(r"signal:\s*(-?[\d.]+)\s*dBm", stripped)
        if m:
            cur["rssi_dbm"] = float(m.group(1))
            continue

    if cur and cur.get("ssid"):
        networks.append(_from_dict(cur, timestamp))
    return networks


def _from_dict(data: dict, timestamp: datetime) -> NetworkResult:
    """Build a NetworkResult from a parsed-accumulator dict."""
    return NetworkResult(
        ssid=data.get("ssid") or "Unknown",
        bssid=data.get("bssid") or "Unknown",
        rssi_dbm=data.get("rssi_dbm", 0.0),
        frequency_mhz=data.get("frequency_mhz", 0.0),
        channel=data.get("channel", 0),
        timestamp=timestamp,
        radio_type=data.get("radio_type", ""),
    )


def _scan_linux(interface: str = "wlan0") -> list[NetworkResult]:
    """Scan WiFi networks on Linux using ``iw`` (preferred) or ``iwlist``."""
    timestamp = datetime.now()

    # Try iw first (newer), fall back to iwlist
    for cmd, args, parser in [
        ("iw", ["iw", "dev", interface, "scan"], _parse_iw),
        ("iwlist", ["iwlist", interface, "scan"], _parse_iwlist),
    ]:
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError:
            logger.debug("%s not found, trying next scanner", cmd)
            continue
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{cmd} command timed out after 30 seconds")

        if result.returncode != 0:
            stderr = result.stderr.lower()
            if "operation not permitted" in stderr or "permission denied" in stderr:
                raise PermissionError(
                    f"{cmd} scan requires root privileges on {interface}. "
                    f"Run with sudo or set CAP_NET_ADMIN capability."
                )
            if "no such device" in stderr or "network is down" in stderr:
                raise RuntimeError(
                    f"Interface '{interface}' is not available or is down"
                )
            logger.debug("%s returned rc=%d, stderr=%s", cmd, result.returncode, result.stderr)
            continue

        if "no scan results" in result.stdout.lower():
            logger.warning("%s returned no scan results on %s", cmd, interface)
            return []

        networks = parser(result.stdout, timestamp)
        if networks:
            return networks

    raise RuntimeError(
        f"No wireless scanner found on interface '{interface}'. "
        f"Ensure 'iw' or 'iwlist' is installed and the interface exists."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_networks(interface: str = "") -> list[NetworkResult]:
    """Scan for nearby WiFi networks.

    Detects the current OS and dispatches to the appropriate scanner:
    - Windows → netsh
    - Linux   → iw / iwlist

    Args:
        interface: Wireless interface for Linux (e.g. ``'wlan0'``).
                   Ignored on Windows.

    Returns:
        List of NetworkResult sorted by signal strength (strongest first).

    Raises:
        RuntimeError: If scanning fails or no scanner is available.
        PermissionError: If elevated privileges are required (Linux).
    """
    os_name = platform.system()

    if os_name == "Windows":
        logger.info("Scanning WiFi networks on Windows via netsh")
        results = _scan_windows()
    elif os_name == "Linux":
        iface = interface or "wlan0"
        logger.info("Scanning WiFi networks on Linux via interface %s", iface)
        results = _scan_linux(iface)
    else:
        raise RuntimeError(f"Unsupported platform: {os_name}")

    results.sort(key=lambda n: n.rssi_dbm, reverse=True)
    logger.info("Found %d network(s)", len(results))
    return results


class RSSIScanner:
    """High-level WiFi RSSI scanner with caching.

    Usage::

        scanner = RSSIScanner(interface="wlan0")
        networks = scanner.scan()
        for net in networks:
            print(f"{net.ssid}: {net.rssi_dbm} dBm on channel {net.channel}")
    """

    def __init__(self, interface: str = ""):
        self.interface = interface
        self._last_scan: list[NetworkResult] = []

    def scan(self) -> list[NetworkResult]:
        """Perform a WiFi scan and cache the results."""
        self._last_scan = scan_networks(self.interface)
        return self._last_scan

    @property
    def last_scan(self) -> list[NetworkResult]:
        """Return the most recent scan results."""
        return self._last_scan


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        networks = scan_networks()
    except PermissionError as exc:
        print(f"\nPermission error: {exc}")
        print("Try running with sudo / administrator privileges.")
        raise SystemExit(1)
    except RuntimeError as exc:
        print(f"\nScan error: {exc}")
        raise SystemExit(1)

    if not networks:
        print("No WiFi networks found.")
        raise SystemExit(0)

    hdr = f"{'SSID':<25} {'BSSID':<19} {'RSSI':>7} {'Freq':>10} {'Ch':>4} {'Radio':<10}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for net in networks:
        ssid = net.ssid[:24] if net.ssid else "(hidden)"
        freq = f"{net.frequency_mhz:.0f} MHz" if net.frequency_mhz else "N/A"
        ch = str(net.channel) if net.channel else "N/A"
        print(f"{ssid:<25} {net.bssid:<19} {net.rssi_dbm:>6.1f}  {freq:>10} {ch:>4} {net.radio_type:<10}")
    print(f"\n{len(networks)} network(s) found — {datetime.now():%Y-%m-%d %H:%M:%S}")