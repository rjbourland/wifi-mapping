# ESP32-S3 CSI Firmware Setup Guide

## Overview
To move from RSSI to CSI (Channel State Information), we use the ESP32-S3's ability to extract the amplitude and phase of subcarriers in the OFDM symbol.

## Hardware Requirements
- **Device:** ESP32-S3 (WROOM-1 or similar).
- **Programmer:** USB-to-UART bridge (built-in on most S3 boards).
- **Antenna:** External SMA antenna is recommended for better SNR.

## Software Stack
1. **ESP-IDF (v5.1+):** The official IoT Development Framework.
2. **ESP-CSI Toolkit:** A community or vendor-provided library for CSI extraction.
3. **Python Setup:** `pip install pyserial pandas numpy`.

## Setup Steps

### 1. Environment Preparation
- Install [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/index.html).
- Configure the environment variables (`export.sh` or `export.bat`).

### 2. Firmware Configuration
- **CSI Enable:** Enable `CONFIG_WIFI_CSI_ENABLE` in `menuconfig`.
- **Channel Selection:** Set the WiFi channel (e.g., 1, 6, 11) and bandwidth (20MHz).
- **Packet Generation:** Set up a "Sender" node to transmit Null Data Packets (NDP) or specialized CSI frames.

### 3. Data Collection Workflow
- **Flash:** Upload the CSI firmare to the ESP32-S3.
- **Serial Capture:** The ESP32 will output CSI data (complex numbers) over the UART port.
- **Python Capture:** Use a Python script to read the serial stream and save it to `.bin` or `.csv` files.

## Data Structure (CSI Packet)
Each CSI packet typically contains:
- Timestamp
- RSSI
- Subcarrier Index (e.g., 64 subcarriers for 20MHz)
- Amplitude (Real + Imaginary parts)
