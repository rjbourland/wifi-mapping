#!/bin/bash
# WiFi Mapping — Raspberry Pi Setup Script
# Run on a fresh Raspberry Pi OS (Bookworm+) with a WiFi adapter.
#
# Usage: chmod +x deploy/setup_pi.sh && sudo ./deploy/setup_pi.sh

set -euo pipefail

INSTALL_DIR="/opt/wifi-mapping"
VENV_DIR="${INSTALL_DIR}/venv"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== WiFi Mapping Pi Setup ==="
echo "Project: ${PROJECT_DIR}"
echo "Install: ${INSTALL_DIR}"

# 1. System packages
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv \
    iw wireless-tools net-tools \
    libpcap-dev libbluetooth-dev \
    nodejs npm \
    || true

# 2. Create install directory
echo "[2/6] Setting up install directory..."
sudo mkdir -p "${INSTALL_DIR}"
sudo cp -r "${PROJECT_DIR}"/* "${INSTALL_DIR}/"

# 3. Python venv + dependencies
echo "[3/6] Creating Python virtual environment..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install -r "${INSTALL_DIR}/requirements.txt" --quiet

# Install Streamlit (not in core requirements)
"${VENV_DIR}/bin/pip" install streamlit plotly --quiet

# 4. Monitor mode setup (for Nexmon-patched Pi)
echo "[4/6] Configuring monitor interface..."
read -p "Set up monitor mode interface? (y/N): " setup_mon
if [[ "${setup_mon}" =~ ^[Yy]$ ]]; then
    read -p "WiFi interface (default: wlan0): " wifi_iface
    wifi_iface="${wifi_iface:-wlan0}"

    # Create monitor interface
    sudo iw dev "${wifi_iface}" interface add mon0 type monitor 2>/dev/null || true
    sudo ip link set mon0 up
    sudo iw dev mon0 set channel 36 2>/dev/null || true
    echo "Monitor interface mon0 created on ${wifi_iface}"
fi

# 5. Systemd service
echo "[5/6] Installing systemd service..."
sudo cp "${INSTALL_DIR}/deploy/wifi-mapping.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable wifi-mapping.service
echo "Service installed. Start with: sudo systemctl start wifi-mapping"

# 6. Pi-specific config overlay
echo "[6/6] Applying Pi config overlay..."
if [[ -f "${INSTALL_DIR}/deploy/pi_config.yaml" ]]; then
    mkdir -p "${INSTALL_DIR}/configs"
    cp "${INSTALL_DIR}/deploy/pi_config.yaml" "${INSTALL_DIR}/configs/pi.yaml"
    echo "Pi config saved to ${INSTALL_DIR}/configs/pi.yaml"
fi

echo ""
echo "=== Setup Complete ==="
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8501"
echo "Start service: sudo systemctl start wifi-mapping"
echo "View logs:    journalctl -u wifi-mapping -f"