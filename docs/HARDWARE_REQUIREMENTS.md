# WiFi Mapping — Hardware Requirements & Setup Guide

> **Project**: WiFi Signal Triangulation & 3D Indoor Mapping  
> **Phase**: 2 — Hardware Selection & Test Environment  
> **Last Updated**: April 2026

---

## 1. Recommended Hardware Setups

### 1.1 Research Setup (Recommended for Accuracy)

| Component | Item | Est. Cost | Notes |
|-----------|------|-----------|-------|
| NIC | Intel AX210 (M.2 NGFF) | $20-25 | WiFi 6E, 2.4/5/6 GHz, 2x2 MIMO, 160 MHz |
| Adapter | M.2 NGFF to PCIe or USB adapter | $10-15 | Required for desktop/laptop without M.2 slot |
| CSI Tool | FeitCSI (open-source) | Free | 802.11a/g/n/ac/ax, all bandwidths, FTM support |
| Collection server | Any Linux box | $0 (existing) | Ubuntu 20.04+ recommended |

**Total per node**: ~$30-40  
**Minimum for 3D**: 4 nodes + 1 collection server = ~$120-160

**Why AX210 + FeitCSI**:
- Most modern open-source CSI tool (2024+)
- WiFi 6 (802.11ax) support — 160 MHz channels for 0.59 m range bins
- 2x2 MIMO enables AoA estimation
- 6 GHz band (AX210 variant) for less interference
- FTM (802.11mc/az) support built-in
- Active development and community support

**Known issues with FeitCSI**:
- Buffer overflow bug in HE (WiFi 6) mode — use non-HE mode for stable CSI
- 5 GHz band intermittent on some AX210 variants — test with `sudo modprobe iwlwifi`
- Single maintainer — consider contributing patches upstream

---

### 1.2 Budget Setup (Recommended for Proof-of-Concept)

| Component | Item | Est. Cost | Notes |
|-----------|------|-----------|-------|
| CSI Node | ESP32-S3-DevKitC-1 | $15.95 | Official Espressif dev board |
| Alternative | XIAO ESP32-S3 Sense | $13.99 | Smaller, built-in antenna + camera |
| Budget option | Generic ESP32-S3 (AliExpress) | $5-10 | Works but less reliable |
| CSI Tool | esp-csi (official Espressif) | Free | 2.4 GHz, 20 MHz only, 52 subcarriers |
| Collection | Any computer with WiFi | $0 (existing) | Receives serial data from ESP32 nodes |

**Total per node**: ~$10-16  
**Minimum for 3D**: 4 nodes + collection computer = ~$40-64

**Why ESP32-S3**:
- Lowest cost entry point for CSI research
- Official Espressif support (1,200+ stars, actively maintained)
- Easy to deploy in "swarms" for distributed mapping
- USB serial output makes data collection straightforward
- External antenna connector on some boards for better SNR

**Limitations**:
- **2.4 GHz only** — no 5 GHz band (more interference, coarser range bins)
- **20 MHz bandwidth** — 4.68 m range bins (vs 0.59 m with AX210 at 160 MHz)
- **1x1 MIMO** — no spatial diversity, no AoA estimation from a single node
- **TDMA limits multi-node sampling** to ~1.4 Hz with 9-node mesh
- **Stationary presence detection only ~38%** — insufficient for reliable presence detection without multiple links

---

### 1.3 Raspberry Pi Setup (Easy Deployment)

| Component | Item | Est. Cost | Notes |
|-----------|------|-----------|-------|
| SBC | Raspberry Pi 4 (1 GB) | $35 | Sufficient for Nexmon CSI collection |
| Alternative | Raspberry Pi 5 (4 GB) | $85 | Better processing but same WiFi limitations |
| CSI Tool | Nexmon CSI (open-source) | Free | BCM43455c0, 80 MHz, 242 subcarriers |
| Storage | 32 GB microSD | $8 | |

**Total per node**: ~$43 (Pi 4) or ~$93 (Pi 5)

**Why Raspberry Pi + Nexmon**:
- Self-contained Linux system — no external computer needed
- Built-in WiFi (no additional NIC purchase)
- Easy SSH access for remote data collection
- Actively maintained (updated Dec 2025)

**Critical limitation**: BCM43455c0 is **1x1 MIMO** — single antenna only. This means:
- No AoA estimation from a single Pi
- Must rely on RSSI or fingerprinting approaches
- Multiple Pi nodes needed for triangulation (RSSI-based)

**For multi-antenna Nexmon**: Asus RT-AC86U router with BCM4366c0 supports 4x4 MIMO but costs ~$130 per node.

---

### 1.4 Full Comparison Table

| Feature | AX210 + FeitCSI | ESP32-S3 + esp-csi | Pi 4 + Nexmon | Pi + BCM4366c0 |
|---------|----------------|--------------------|---------------|----------------| 
| **WiFi Standard** | 802.11ax (WiFi 6) | 802.11n (WiFi 4) | 802.11ac (WiFi 5) | 802.11ac (WiFi 5) |
| **Bands** | 2.4/5/6 GHz | 2.4 GHz | 2.4/5 GHz | 2.4/5 GHz |
| **Max Bandwidth** | 160 MHz | 20 MHz | 80 MHz | 80 MHz |
| **Range Bin Size** | 0.59 m | 4.68 m | 1.17 m | 1.17 m |
| **Subcarriers** | Full (up to 1992) | 52-56 | 242 | 242 |
| **MIMO** | 2x2 | 1x1 | 1x1 | 4x4 |
| **AoA Capable** | Yes | No | No | Yes |
| **FTM Support** | Yes | No | No | No |
| **Open Source** | Yes (GPL-3.0) | Yes (Apache-2.0) | Yes (MIT) | Yes (MIT) |
| **Cost/Node** | ~$30 | ~$10-16 | ~$43 | ~$130 |
| **Linux Required** | Yes | No (serial output) | Self-contained | Self-contained |
| **Best For** | Research, accuracy | Budget, PoC, IoT | Easy deployment | Multi-antenna AoA |

---

## 2. Test Environment Setup

### 2.1 Room Requirements

- Minimum 4m x 4m room (larger is better)
- Mix of open space and furniture for realistic multipath
- At least 1 interior wall for through-wall testing
- Power outlets at each anchor position
- Ethernet or WiFi backhaul to collection server

### 2.2 Anchor Placement Strategy

**For 3D localization**, anchors must be **non-coplanar** (at different heights):

```
Ceiling anchor ──────────────────── ~2.5m
    │                    
    │     Mid-height anchor ──────── ~1.3m
    │         │           
    │         │    Floor anchor ───── ~0.3m
    │         │        │    
    ▼         ▼        ▼    
   [A1]      [A2]      [A3]      [A4]
```

**Placement guidelines**:
- Place anchors at 3 different heights: floor (~0.3 m), mid (~1.3 m), ceiling (~2.5 m)
- Maximize angular separation between anchors as seen from the center of the room
- Avoid placing all anchors on the same wall (corridor problem)
- Minimum 4 anchors for 3D; 5-6 for redundancy
- Document exact (x, y, z) coordinates of each anchor in a reference frame

### 2.3 Ground-Truth Reference Grid

Mark reference positions on the floor and walls:

1. **Floor grid**: Use masking tape to mark a 0.5m x 0.5m grid across the room
2. **Wall markers**: Place reference points at multiple heights on each wall
3. **Height reference**: Use a laser measure or measuring tape to record heights
4. **Coordinate system**: Define origin at one corner of the room, with x-axis along one wall, y-axis along the perpendicular wall, z-axis up

**Ground-truth data collection**:
- At each grid point, record 30+ CSI samples while standing still
- Record the position in (x, y, z) coordinates
- Note LoS/NLoS conditions from each anchor
- Record ambient conditions (people present, door open/closed, furniture arrangement)

### 2.4 Collection Server Architecture

```
┌─────────────────────────────────────────────┐
│            Collection Server                 │
│            (Linux/Python)                    │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ UDP/Serial│  │  Config  │  │ Storage  │   │
│  │ Listener  │  │  Manager │  │ (Parquet)│   │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘   │
│        │             │             │          │
│        └─────────────┼─────────────┘          │
│                      │                        │
└──────────────────────┼────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
     ┌────┴───┐   ┌────┴───┐   ┌────┴───┐
     │ Node 1 │   │ Node 2 │   │ Node 3 │ ...
     │(Anchor)│   │(Anchor)│   │(Anchor)│
     └────────┘   └────────┘   └────────┘
```

**Server requirements**:
- Python 3.10+
- Network interface on same subnet as all nodes
- Sufficient disk space (1 GB/hour of CSI data per node at 100 Hz)
- NTP-synchronized clock for timestamping

---

## 3. Configuration Parameters

### 3.1 WiFi Channel Selection

| Band | Channel | Frequency | Bandwidth | Notes |
|------|---------|-----------|-----------|-------|
| 2.4 GHz | 1 | 2412 MHz | 20 MHz | Most crowded, avoid |
| 2.4 GHz | 6 | 2437 MHz | 20 MHz | Moderate interference |
| 2.4 GHz | 11 | 2462 MHz | 20 MHz | Common, moderate interference |
| **5 GHz** | **36** | **5180 MHz** | **20/40/80 MHz** | **Recommended for testing** |
| 5 GHz | 40 | 5200 MHz | 20/40/80 MHz | Good alternative |
| 5 GHz | 149 | 5745 MHz | 20/40/80 MHz | Less interference, check local regs |
| 6 GHz | 1 | 5955 MHz | 20-160 MHz | AX210 only, least interference |

**Recommendation**: Use 5 GHz channel 36 at 80 MHz bandwidth for initial testing. Switch to 160 MHz with AX210 for high-resolution work.

### 3.2 ESP32-S3 Specific Configuration

```
CONFIG_WIFI_CSI_ENABLE=y
CONFIG_WIFI_CHANNEL=36        # 5 GHz not available; use channel 6
CONFIG_WIFI_BANDWIDTH=20       # 20 MHz only
CONFIG_WIFI_CSI_RECV_CB=y
CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM=16
CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM=32
```

**Note**: ESP32-S3 is 2.4 GHz only — use channel 6 for 20 MHz bandwidth.

### 3.3 FeitCSI (AX210) Configuration

```bash
# Load kernel module with CSI parameters
sudo modprobe iwlwifi            # Standard driver (CSI via FeitCSI module)

# FeitCSI collection parameters
sudo FeitCSI -i wlan0 -c 36 -b 80 -n 100    # Channel 36, 80 MHz, 100 packets
```

---

## 4. Purchasing Guide

### 4.1 Budget Setup (4x ESP32-S3) — Total ~$50-65

| Item | Qty | Unit Price | Source | Notes |
|------|-----|-----------|--------|-------|
| ESP32-S3-DevKitC-1 | 4 | $15.95 | [Espressif Store](https://www.espressif.com/en/products/devkits) | Official dev board |
| USB-C cables | 4 | $3.00 | Amazon | For power + serial |
| Antenna (external SMA) | 4 | $5.00 | Amazon | Optional, improves SNR |
| **Total** | | **~$65** | | |

### 4.2 Research Setup (4x AX210 + Server) — Total ~$140-170

| Item | Qty | Unit Price | Source | Notes |
|------|-----|-----------|--------|-------|
| Intel AX210 NIC (M.2) | 4 | $20 | Amazon/eBay | M.2 NGFF variant |
| M.2 to PCIe adapter | 4 | $10 | Amazon | For desktop use |
| USB-C WiFi adapter (AX210) | 4 | $35 | Amazon | Alternative: integrated AX210 |
| Linux collection server | 1 | $0 | Existing | Any Ubuntu 20.04+ box |
| **Total (PCIe)** | | **~$120** | | |
| **Total (USB)** | | **~$140** | | |

### 4.3 Pi Deployment Setup (4x Pi 4 + Nexmon) — Total ~$185-300

| Item | Qty | Unit Price | Source | Notes |
|------|-----|-----------|--------|-------|
| Raspberry Pi 4 (1 GB) | 4 | $35 | Pi Shop / Amazon | 1 GB sufficient for collection |
| 32 GB microSD | 4 | $8 | Amazon | Class 10 / A2 |
| Power supply | 4 | $10 | Amazon | 5V 3A USB-C |
| **Total** | | **~$212** | | |

---

## 5. Hardware Decision Matrix

| If you need... | Choose... | Because... |
|----------------|-----------|------------|
| Highest accuracy (sub-meter 3D) | AX210 + FeitCSI | 160 MHz, 2x2 MIMO, AoA capable |
| Lowest cost proof-of-concept | ESP32-S3 + esp-csi | $10-15 per node, easy to start |
| Easy deployment (no Linux) | Pi 4 + Nexmon | Self-contained, SSH-only management |
| Multi-antenna AoA without AX210 | Asus RT-AC86U + Nexmon | 4x4 MIMO, 80 MHz |
| Through-wall breathing detection | Custom FMCW or Pi + Nexmon | FMCW for best accuracy; Nexmon for commodity |
| 802.11az/FTM ranging | Intel AX210 + FeitCSI | Only tool with FTM + CSI combined |
| Swarm deployment (10+ nodes) | ESP32-S3 | Low cost, low power, mesh capable |

---

## 6. Software Dependencies

### 6.1 Collection Server

```txt
# requirements.txt (core)
numpy>=1.24
pandas>=2.0
pyyaml>=6.0
pyarrow>=12.0          # Parquet storage
pyserial>=3.5           # ESP32 serial capture
csikit>=0.2             # Multi-format CSI parsing
matplotlib>=3.7         # Visualization
scipy>=1.10             # Signal processing
```

### 6.2 Optional (for advanced features)

```txt
# requirements-advanced.txt
open3d>=0.17            # 3D point cloud / visualization
torch>=2.0              # Deep learning localization
sklearn>=1.3            # Fingerprinting (k-NN)
```

### 6.3 ESP32 Firmware

- ESP-IDF v5.1+
- esp-csi component (from Espressif components registry)
- Flash using `idf.py flash monitor`

### 6.4 FeitCSI (AX210)

- Linux kernel 5.15+
- FeitCSI kernel module (build from source)
- Ubuntu 20.04+ recommended