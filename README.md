# WiFi Signal Triangulation & 3D Indoor Mapping

WiFi-based indoor positioning and device-free sensing using Channel State Information (CSI). Implements RSSI trilateration, CSI fingerprinting, MUSIC AoA estimation, 3D mapping, and motion/breathing detection.

## Architecture

```
wifi-mapping/
├── configs/              # YAML configuration files
│   ├── anchors.yaml      # Anchor node positions
│   ├── collection.yaml   # Data collection parameters
│   └── algorithm.yaml    # Algorithm hyperparameters
├── docs/
│   ├── LITERATURE_REVIEW.md    # Comprehensive research brief
│   ├── HARDWARE_REQUIREMENTS.md # Hardware selection guide
│   └── ESP32_CSI_SETUP.md      # ESP32 firmware setup guide
├── src/
│   ├── collection/       # Data collection
│   │   ├── rssi_scanner.py     # Windows netsh RSSI scanner
│   │   ├── csi_collector.py    # CSI data collection (UDP/serial)
│   │   └── ground_truth.py     # Ground-truth position logging
│   ├── processing/       # Signal processing
│   │   ├── process_rssi.py     # RSSI data processing
│   │   ├── csi_parser.py       # Multi-format CSI parsing
│   │   ├── phase_sanitizer.py  # CFO/STO/SFO phase correction
│   │   └── feature_extraction.py # Feature extraction for ML
│   ├── localization/     # Position estimation
│   │   ├── trilateration.py    # RSSI path-loss trilateration (3D)
│   │   ├── fingerprinting.py   # k-NN CSI fingerprinting
│   │   ├── aoa_estimation.py   # MUSIC AoA estimation
│   │   └── kalman_filter.py    # Position tracking/smoothing
│   ├── mapping/           # 3D mapping
│   │   ├── point_cloud.py      # Point cloud accumulation
│   │   ├── occupancy_grid.py   # Voxel occupancy mapping
│   │   └── visualization.py   # 3D visualization (matplotlib/Open3D)
│   ├── detection/         # Device-free sensing
│   │   ├── motion_detector.py # Variance-based motion detection
│   │   ├── breathing_detector.py # Breathing rate estimation
│   │   └── gait_classifier.py  # Gait pattern classification
│   └── utils/             # Utilities
│       ├── config.py            # YAML config loader
│       ├── data_formats.py      # Data schemas (CSISample, etc.)
│       └── math_utils.py        # Geometry and path-loss math
└── tests/
    ├── test_trilateration.py
    ├── test_csi_parser.py
    └── test_motion_detector.py
```

## Quick Start

### Install

```bash
cd wifi-mapping
pip install -e ".[dev]"
```

### RSSI Scanning (Windows, no hardware)

```bash
python -m src.collection.rssi_scanner
```

### CSI Collection (requires hardware)

```python
from src.collection.csi_collector import CSICollector
from src.utils.config import load_config

config = load_config("collection")
collector = CSICollector(config)

# Collect from ESP32 via serial
collector.start_serial_listener(port="COM3")
samples = collector.collect_samples(num_samples=100, anchor_id="esp32_1")
collector.save_samples(samples)
collector.stop()
```

### Localization

```python
from src.localization.trilateration import TrilaterationSolver

solver = TrilaterationSolver()
solver.load_anchors_from_config()

# RSSI measurements from anchors
rssi = {"A1": -45.0, "A2": -55.0, "A3": -50.0, "A4": -48.0}
position = solver.localize(rssi)
print(f"Estimated position: {position.position}")
```

### Motion Detection

```python
from src.detection.motion_detector import MotionDetector
import numpy as np

detector = MotionDetector()
# csi_packets: shape (num_packets, num_antennas, num_subcarriers)
result = detector.detect(csi_packets)
print(f"Motion: {result['is_motion']}, Score: {result['motion_score']:.2f}")
```

## Supported Hardware

| Hardware | Tool | WiFi Standard | Max BW | MIMO | Cost |
|----------|------|---------------|--------|------|------|
| Intel AX210 | FeitCSI | 802.11ax | 160 MHz | 2x2 | ~$30 |
| ESP32-S3 | esp-csi | 802.11n | 20 MHz | 1x1 | ~$10 |
| Raspberry Pi 4 | Nexmon | 802.11ac | 80 MHz | 1x1 | ~$55 |

See `docs/HARDWARE_REQUIREMENTS.md` for full details.

## Research

See `docs/LITERATURE_REVIEW.md` for the comprehensive research brief covering:
- SpotFi, Widar 3.0, WiSee, DLoc, SPRING+ positioning systems
- CSI extraction tools comparison (Intel 5300, ESP-CSI, Nexmon, FeitCSI)
- IEEE 802.11az/802.11bk next-gen positioning
- Passive WiFi radar capabilities
- Realistic accuracy expectations and hardware recommendations

## Configuration

Edit YAML files in `configs/`:

- **anchors.yaml**: Anchor node positions and hardware settings
- **collection.yaml**: Data collection parameters (sample rate, channels, etc.)
- **algorithm.yaml**: Algorithm hyperparameters (path-loss model, k-NN, MUSIC, Kalman filter)

## Testing

```bash
pytest tests/ -v
```