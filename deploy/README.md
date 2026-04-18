# WiFi Mapping — Raspberry Pi Deployment Guide

## Prerequisites

- Raspberry Pi 4 (or 5) running Raspberry Pi OS Bookworm (64-bit recommended)
- WiFi adapter capable of monitor mode (built-in Broadcom with Nexmon patch, or USB dongle)
- ESP32-S3 anchor nodes (3+ recommended) configured as APs
- Python 3.10+

## Quick Setup

```bash
# From the project directory on the Pi:
chmod +x deploy/setup_pi.sh
sudo ./deploy/setup_pi.sh
```

The script will:
1. Install system packages (iw, python3-venv, etc.)
2. Copy the project to `/opt/wifi-mapping`
3. Create a Python venv and install dependencies
4. Optionally set up monitor mode (`mon0`)
5. Install the systemd service
6. Apply Pi-specific config

## Manual Setup

### 1. Flash Raspberry Pi OS

```bash
# Use Raspberry Pi Imager: https://www.raspberrypi.com/software/
# Choose: Raspberry Pi OS (64-bit) Bookworm
# Enable SSH in advanced options
```

### 2. Nexmon CSI Patch (for Broadcom BCM43455c0)

```bash
# Clone Nexmon CSI
git clone https://github.com/nexmon/nexmon_csi.git
cd nexmon_csi

# Follow the build instructions for Pi 4
# This patches the WiFi firmware to expose CSI data
# After patching, create monitor interface:
sudo iw phy phy0 interface add mon0 type monitor
sudo ip link set mon0 up
sudo iw dev mon0 set channel 36
```

### 3. Configure Anchor Nodes

1. Flash each ESP32-S3 with AP firmware
2. Set static IPs: 192.168.1.101–104
3. Configure SSID and channel to match `pi_config.yaml`
4. Mount anchors at known positions (update anchors.yaml)

### 4. Start the Dashboard

```bash
# Via systemd (recommended):
sudo systemctl start wifi-mapping
sudo systemctl status wifi-mapping

# Or manually:
cd /opt/wifi-mapping
source venv/bin/activate
streamlit run gui/app.py --server.port 8501 --server.address 0.0.0.0
```

### 5. Access from Laptop

Open `http://<pi-ip>:8501` in your browser.

## Troubleshooting

### WiFi Scan Permissions
```bash
# RSSI scanning requires root on Linux
sudo setcap cap_net_raw+ep /opt/wifi-mapping/venv/bin/python3
```

### Monitor Mode Issues
```bash
# Check interface status
iw dev mon0 info
iw dev mon0 station dump

# Restart monitor mode
sudo ip link set mon0 down
sudo iw dev mon0 del
sudo iw phy phy0 interface add mon0 type monitor
sudo ip link set mon0 up
```

### Service Logs
```bash
journalctl -u wifi-mapping -f
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  ESP32-S3   │     │  Raspberry Pi│     │  Laptop       │
│  AP Anchors │────▶│  Scanner +   │────▶│  Dashboard    │
│  (x3-4)     │WiFi │  Dashboard   │HTTP│  (Browser)    │
└─────────────┘     └──────────────┘     └───────────────┘
```

- ESP32 anchors broadcast WiFi beacons
- Pi scans RSSI via `iw` (or CSI via Nexmon)
- Dashboard runs Streamlit on port 8501
- Laptop accesses dashboard via LAN