# WiFi Signal Triangulation & 3D Indoor Mapping — Literature Review & Feasibility Study

> **Project**: WiFi Signal Triangulation & 3D Indoor Mapping  
> **Phase**: 1 — Literature Review & Feasibility  
> **Date**: April 2026  
> **Status**: Research complete, feasibility confirmed

---

## Executive Summary

WiFi-based indoor positioning and device-free sensing is a mature research field with demonstrated sub-meter accuracy using commodity hardware. The most practical path forward for this project is:

1. **Primary hardware**: Intel AX210 NICs ($20 each) with FeitCSI (open-source, WiFi 6, 160 MHz, 2x2 MIMO)
2. **Budget hardware**: ESP32-S3 boards ($10-15 each) with esp-csi for proof-of-concept and swarm deployment
3. **Expected accuracy**: 0.5-1.5 m median 2D error with CSI-based methods; 1-3 m for height estimation
4. **Minimum anchors**: 4 non-coplanar APs for 3D localization; 3 for 2D

CSI extraction from commodity hardware is feasible and well-supported by open-source tools. The main challenge is multipath propagation in indoor environments, which degrades accuracy 2-4x in NLoS conditions.

---

## 1. Key Indoor Positioning Systems

### 1.1 SpotFi (Stanford, SIGCOMM 2015)

**Technique**: Joint AoA + Time-of-Flight super-resolution using MUSIC algorithm. Creates a virtual sensor array of 90 elements (3 antennas x 30 subcarriers) from 3 physical antennas via spatial smoothing across both antennas and subcarriers.

**Accuracy**: 40 cm median (LoS), 1.6 m (NLoS), 1.1 m (corridors)

**Requirements**: Intel 5300 NIC (3 antennas), 4+ APs for best results, as few as 10 packets per estimation

**Key insight**: No hardware modifications — uses only CSI and RSSI from commodity NICs. Gaussian Mean clustering on AoA-ToF estimates identifies the direct path even in multipath.

**Paper**: [SIGCOMM 2015](https://web.stanford.edu/~skatti/pubs/sigcomm15-spotfi.pdf)

---

### 1.2 Widar Series (Tsinghua University)

**Widar 1.0** (MobiHoc 2017): Decimeter-level passive tracking via velocity monitoring using CSI-Mobility model. Extracts Doppler-like frequency shifts from CSI power conjugate multiplication. Median tracking error: 25 cm (known initial position), 38 cm (without). Requires 1 transmitter + 2 receivers (3 links).

**Widar 2.0** (MobiSys 2018): Achieves decimeter tracking with a single WiFi link. Median error: 0.75 m.

**Widar 3.0** (MobiSys 2019 / TPAMI 2021): Extends to zero-effort cross-domain gesture recognition using Body-coordinate Velocity Profile (BVP). Gesture accuracy: ~85% cross-domain (6 gestures), up to 93.9% with SRCC features.

**Dataset**: Publicly available on [IEEE DataPort](https://www.ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset) — ~258,000 gesture instances across 16 subjects, 5 environments.

**Project site**: http://tns.thss.tsinghua.edu.cn/widar3.0

---

### 1.3 WiSee (University of Washington, MobiCom 2013)

**Technique**: WiFi gesture recognition via Doppler shift extraction. Transforms wideband OFDM WiFi into narrowband pulses via large FFT over consecutive OFDM symbols, extracting frequency-time Doppler profiles.

**Accuracy**: 94% classification across 9 gestures (5 users, 10 locations)

**Through-wall**: Yes — works through walls and in NLoS scenarios

**Hardware**: USRP-N210 software radios (not commodity — research SDR)

**Project**: https://wisee.cs.washington.edu/

---

### 1.4 WiVi (MIT, SIGCOMM 2013)

**Technique**: First system to detect/track humans through walls using WiFi (2.4 GHz, 20 MHz). Uses MIMO interference nulling to cancel wall reflections ("flash effect") and inverse SAR treating the moving body as an antenna array.

**Accuracy**: Distinguishes 0-3 humans through walls with 85-100% accuracy

**Hardware**: USRP N210 (SDR, not commodity WiFi)

**Project**: http://people.csail.mit.edu/fadel/wivi/

---

### 1.5 WiTrack / RF-Capture / Vital-Radio (MIT CSAIL, Katabi Lab)

Series of increasingly capable through-wall RF sensing systems using **custom FMCW radio hardware** (5.46-7.25 GHz sweep):

| System | Year | Capability | Accuracy | Through-Wall |
|--------|------|------------|----------|--------------|
| WiTrack | 2014 | 3D single-person tracking | 10-13 cm (x/y), 21 cm (z) | Yes |
| WiTrack 2.0 | 2015 | Multi-person (4 moving, 5 static) | ~12 cm (x/y) | Yes |
| RF-Capture | 2015 | Human silhouette capture | Hand: 2.19 cm; person ID: 88% | Yes |
| Vital-Radio | 2015 | Breathing + heart rate | Breathing: 99.3%; HR: 98.5% | Yes |

**Key limitation**: Requires custom FMCW radio hardware (not commodity WiFi). Commercialized as **Emerald** (200+ homes, monitoring Parkinson's, Alzheimer's, etc.).

---

### 1.6 DLoc (MIT/Microsoft, MobiCom 2020)

**Technique**: Deep learning-based WiFi localization using CSI. Encoder-decoder neural network converts CSI matrices into 2D AoA-ToF heatmaps, with consistency decoder correcting ToF offsets across APs.

**Accuracy**: 65 cm median (vs. 110 cm for SpotFi baseline)

**Dataset**: 105,000+ data points across 8 scenarios in 2000 sq ft. Includes MapFind: autonomous mapping robot (Turtlebot2 + LIDAR + Quantenna WiFi)

**Paper**: [MobiCom 2020](https://dspace.mit.edu/bitstream/handle/1721.1/146247/3372224.3380894.pdf)

---

### 1.7 SPRING+ (IMDEA Networks, IEEE TMC 2024)

**Technique**: Positioning from a **single WiFi AP** using CSI + Fine Time Measurements (FTM). Adaptive AoA estimation from CSI combined with robust first-path detection via FTM.

**Accuracy**: 1-1.8 m median 2D positioning error with a single AP

**Key insight**: Demonstrates that meaningful indoor positioning is possible with just one access point when combining CSI with FTM, though multi-AP setups significantly improve accuracy.

---

### 1.8 ML-Track (IEEE TMC 2025)

**Technique**: Multi-link bistatic WiFi Doppler with particle filter for passive device-free tracking.

**Accuracy**: 0.23 m median — the best reported passive tracking with commodity WiFi

**Key insight**: Uses commodity AP signals without modifying the transmitter. Cross-Ambiguity Function (CAF) processing of commercial WiFi signals.

---

## 2. CSI Extraction Tools & Chipsets

### 2.1 Comparison Table

| Tool | Chipset | WiFi Standard | Max BW | Subcarriers | MIMO | Open Source | Cost | Best For |
|------|---------|--------------|--------|-------------|------|-------------|------|----------|
| Linux CSI Tool | Intel 5300 | 11n | 40 MHz | 30 groups | 2x2 | Yes | ~$20 | Legacy baseline research |
| Atheros CSI Tool | AR9580/AR9462 | 11n | 40 MHz | 114 | 3x3 | Yes | ~$20 | Per-subcarrier 11n |
| Nexmon CSI | BCM43455c0 | 11n/ac | 80 MHz | 242 | **1x1** | Yes | ~$55 | Raspberry Pi deployments |
| Nexmon CSI | BCM4366c0 | 11ac | 80 MHz | 242 | 4x4 | Yes | ~$130 | Multi-antenna 11ac |
| ESP-CSI | ESP32/S2/C3/S3 | 11n | 20 MHz | 52 | 1x1 | Yes | $5-15 | IoT/presence detection |
| **FeitCSI** | **Intel AX210** | **11ax** | **160 MHz** | **Full** | **2x2** | **Yes** | **~$30** | **Modern WiFi 6 research** |
| PicoScenes | Intel AX210 | 11ax | 160 MHz | 1,992 | 2x2 | No | Free/$1,500 | Production-grade research |
| ZTECSITool | MT7916 | 11ax | 160 MHz | 512 | 3x2 | Partial | AP cost | WiFi 6 AP research |

### 2.2 Detailed Tool Notes

**Intel 5300 NIC + Linux CSI Tool**: The original and most cited CSI research tool. Modified `iwlwifi` kernel driver. 8-bit quantization, groups subcarriers (30 groups instead of individual). Requires older Linux kernels (3.2-4.2), no WPA support. Hardware is discontinued and increasingly scarce. **Not recommended for new projects.**

**Atheros CSI Tool**: Modified `ath9k` driver. Finer granularity (114 individual subcarriers) than Intel 5300. Only supports 802.11n chipsets using `ath9k` — modern Qualcomm/Atheros 802.11ac/ax chips (ath10k/ath11k) are NOT supported.

**ESP32 + ESP-CSI**: Official Espressif project (1,200+ stars, actively maintained). All ESP32 variants supported. 2.4 GHz only, 20 MHz bandwidth, 52-56 subcarriers. TDMA limits multi-node sampling to ~1.4 Hz. **Best for budget prototyping and IoT deployment, but insufficient for high-accuracy localization.**

**Nexmon CSI (Broadcom)**: Most actively maintained open-source CSI tool (updated Dec 2025). Supports Pi 3B+/4/5 built-in WiFi (BCM43455c0). Critical limitation: BCM43455c0 is **1x1 MIMO** (single antenna), preventing spatial/AoA estimation. BCM4366c0 (Asus RT-AC86U) supports 4x4 MIMO but costs ~$130.

**FeitCSI (Intel AX200/AX210)**: First open-source tool supporting 802.11ax (WiFi 6) CSI extraction. Supports all bandwidths (20-160 MHz), 6 GHz band (AX210), and FTM. Some open bugs (buffer overflow in HE mode, 5 GHz intermittent on AX210). Single maintainer. **Recommended for new research projects.**

**PicoScenes**: Most feature-rich and mature CSI platform. Commercial license at $1,500 for Pro features. Supports Wi-Fi 7 (802.11be) at 320 MHz. Best documentation and support but expensive.

### 2.3 Python Libraries for CSI Processing

| Library | Install | Formats | Key Feature |
|---------|---------|---------|-------------|
| **CSIKit** | `pip install csikit` | Intel 5300, AX200/210, Atheros, Nexmon, ESP32 | Multi-format parsing, filtering, visualization |
| **WSDP** | `pip install wsdp` | Widar, GaitID, XRF55, ZTE | 19 DL models, 26 preprocessing algorithms |
| **nexcsi** | Manual | Nexmon PCAP | Fast Python/NumPy Nexmon decoder |
| **SenseFi** | Manual | Widar, UT-HAR, NTU-Fi | PyTorch benchmark suite (9 architectures) |

### 2.4 Public Datasets

| Dataset | Hardware | Focus | Link |
|---------|----------|-------|------|
| **CSI-Bench** (NeurIPS 2025) | Multiple | Localization, HAR, fall detection, breathing | [Kaggle](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench) |
| **Widar 3.0** | Intel 5300 | Gesture/activity recognition (258K instances) | [Project site](http://tns.thss.tsinghua.edu.cn/widar3.0) |
| **NIC5300 Indoor Loc** | Intel 5300 + Nexus 5 | Indoor localization fingerprinting | [IEEE DataPort](https://ieee-dataport.org/documents/wifi-csi-data-indoor-localization-using-nic5300-and-nexus-5) |
| **CSUIndoorLoc** | Multiple | Indoor localization (CSI+RSSI) | [GitHub](https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav) |

---

## 3. IEEE 802.11az — Next Generation Positioning

### 3.1 Standard Status

IEEE 802.11az-2022 was published March 2023. It enhances Fine Timing Measurement (FTM) from 802.11mc with:

- **Sub-meter accuracy**: Target ~0.4 m at 160 MHz with 2x4 MIMO (LTF=2)
- **MAC/PHY security**: Secure LTF with AES-256 pseudo-random sequences (prevents MITM/spoofing)
- **Native MIMO support**: Spatial diversity for improved ranging
- **Trigger-based multi-user ranging**: Dense environment support
- **NTB ranging**: Non-Trigger Based mode for simpler operation

### 3.2 Successor: IEEE 802.11bk

Expected approval mid-2025. Uses 320 MHz channels (Wi-Fi 7) for **centimeter-level accuracy** approaching UWB.

### 3.3 Hardware Support

**Consumer devices with 802.11az support**:
- Samsung Galaxy S24 series
- Google Pixel 8/8 Pro/8a/9/9 Pro
- OnePlus 12
- Qualcomm FastConnect 7900 (WCN7880/WCN7881)

**Enterprise APs**: Cisco CW9172/9176/9178 series, Aruba AP734/735/754/755

**Security concern** (March 2026): Secure Wi-Fi ranging is highly sensitive to configuration choices; many implementations default to insecure modes, vulnerable to downgrade attacks.

### 3.4 Practical Assessment

802.11az is real and shipping in premium smartphones, but the AP ecosystem is still early. For a system builder today:
- **802.11mc FTM** is the practical choice (1-2 m accuracy with commodity hardware)
- **802.11az** provides an upgrade path to ~0.5 m as the ecosystem matures
- **CSI-based approaches** (SpotFi, DLoc) achieve 0.4-0.65 m today but require CSI extraction tools

---

## 4. Passive WiFi Radar (Device-Free Sensing)

### 4.1 Can WiFi Detect People Through Walls Without a Device?

**Yes.** This has been demonstrated extensively. Key results:

| System | Technique | Accuracy | Through-Wall |
|--------|-----------|----------|--------------|
| WiVi (2013) | MIMO nulling + ISAR | Count 0-3 people (85-100%) | Yes |
| WiTrack (2014) | FMCW (custom) | 10-13 cm (x/y) | Yes |
| WiTrack 2.0 (2015) | Multi-shift FMCW | ~12 cm, 4 people | Yes |
| RF-Capture (2015) | FMCW + body stitching | Person ID: 88% | Yes |
| Vital-Radio (2015) | FMCW + periodicity | Breathing: 99.3%; HR: 98.5% | Yes |
| RF-Pose (2018) | FMCW + deep learning | AP=62.4 (visible), 58.1 (through-wall) | Yes |
| ML-Track (2025) | Multi-link WiFi Doppler + particle filter | **0.23 m median** | Indoor room-scale |

### 4.2 Practical Limitations

- **Custom FMCW hardware** achieves centimeter-level accuracy through walls but costs thousands per unit
- **Commodity WiFi** achieves sub-meter tracking for **moving** targets only; stationary presence detection is poor (~38% accuracy with ESP32 mesh)
- **Breathing detection** works through walls with both custom and commodity hardware
- **Fresnel zone constraint**: At 2.4 GHz (λ = 12.5 cm), a 4-meter link's first Fresnel zone has max radius ~0.5 m. Motion outside this zone produces negligible CSI variation

---

## 5. Accuracy Expectations & Practical Considerations

### 5.1 Realistic Accuracy by Method

| Scenario | Hardware | Method | Accuracy |
|----------|----------|--------|----------|
| Best case (LoS, 3+ ant, CSI) | Intel 5300, 3 antennas | SpotFi super-resolution | 0.4 m median |
| Best case (single link, AI) | Intel 5300, 1 link | AI-driven DBSCAN | 0.63 m median |
| Typical indoor (moderate multipath) | Commodity WiFi | Various CSI methods | 1-2 m mean |
| NLoS / strong multipath | Commodity WiFi | Various CSI methods | 2-5+ m |
| FTM-only (no CSI) | Intel 8260, ESP32-S2 | 802.11mc FTM | 1.15-2.3 m mean |
| Zone classification (fingerprinting) | Standard 802.11ac/ax | Deep learning | >95% for ~1.3 m zones |
| 802.11az (2x4 MIMO, 160 MHz) | Qualcomm FC7900 | FTM NGP | Target 0.5 m (90th %ile) |

**For 3D localization**: WiTrack achieves 10-13 cm (x/y) and 21 cm (z) with custom FMCW. With commodity WiFi, expect 0.5-1 m (x/y) and 1-3 m (z) with 4+ APs.

### 5.2 Anchor Requirements for 3D

| Technique | Minimum Anchors | Recommended | Notes |
|-----------|----------------|-------------|-------|
| ToA trilateration (3D) | 4 non-coplanar | 4-6 | 4th anchor resolves ambiguity |
| TDoA multilateration (3D) | 4 non-coplanar | 4-6 | Extra anchor for clock sync |
| AoA estimation (3D) | 3 | 3-5 | Each provides azimuth + elevation |
| Hybrid ToA/AoA | 1-2 | 2-3 | Combines range and angle |
| RSSI fingerprinting (3D) | 4 | 5+ | More APs improve uniqueness |

**Critical**: Ceiling-mounted APs at the same height are **coplanar**, making the z-axis matrix singular in standard least-squares trilateration. Place anchors at different heights (floor, mid, ceiling level).

### 5.3 Multipath and Environmental Effects

- **Multipath is the dominant error source** — indoor environments typically have 6-8 significant signal paths
- **LoS to NLoS degradation**: SpotFi accuracy degrades from 0.4 m to 1.6 m (4x worse) when the direct path is blocked
- **Corridors are particularly challenging**: APs along one wall create geometrically correlated AoA measurements
- **Furniture movement requires re-calibration** for fingerprinting systems (FPNet drops from 97.5% to 64.6% after environmental changes)
- **Moving people** dynamically change the channel, introducing time-varying multipath

### 5.4 Error Sources (Ranked by Impact)

1. **Multipath propagation** — Reflected paths can be stronger than direct path. 2-4x accuracy degradation in NLoS.
2. **Limited antenna count** — 3-antenna APs can only resolve 2 multipath components with standard MUSIC. Super-resolution increases this to ~8-10 but requires wide bandwidth.
3. **Clock synchronization errors** — STO, SFO, and PDD corrupt ToF estimates. FTM mitigates via round-trip measurement.
4. **Quantized CSI** — Intel 5300 provides only 8 bits per I/Q component and groups subcarriers (30 groups vs 114 individual).
5. **Carrier Frequency Offset (CFO)** — Introduces phase errors across antennas and subcarriers.
6. **Environmental dynamics** — Moving people, doors, furniture rearrangement change the channel.

### 5.5 Bandwidth Matters

Each doubling of bandwidth halves the range bin uncertainty:

| Bandwidth | Range Bin | Notes |
|-----------|-----------|-------|
| 20 MHz | 4.68 m | ESP32 (20 MHz only) |
| 40 MHz | 2.34 m | Intel 5300, Atheros max |
| 80 MHz | 1.17 m | Nexmon BCM4366c0 max |
| 160 MHz | 0.59 m | Intel AX210 (FeitCSI) |
| 320 MHz | 0.29 m | Wi-Fi 7 (PicoScenes only) |

---

## 6. Feasibility Assessment

### 6.1 Can Commodity WiFi Hardware Provide Sufficient CSI Resolution?

**Yes**, with caveats:
- Intel AX210 + FeitCSI provides 2x2 MIMO at 160 MHz — sufficient for SpotFi-style AoA+ToF super-resolution
- ESP32 provides CSI at 20 MHz (4.68 m range bins) — adequate for presence detection and fingerprinting, but not for precise localization
- Nexmon on BCM4366c0 provides 4x4 MIMO at 80 MHz — good for AoA estimation but requires an Asus router (~$130)

### 6.2 Minimum Number of APs for 3D Localization?

**4 non-coplanar APs** minimum for 3D. 3 APs sufficient for 2D. APs must be at different heights to resolve the z-axis.

### 6.3 How Do Walls, Furniture, and Multipath Affect Accuracy?

- Expect **2-4x accuracy degradation** in NLoS vs LoS conditions
- **Corridors** are particularly challenging (geometrically correlated measurements)
- **Furniture rearrangement** requires fingerprinting recalibration
- **Mitigation strategies**: multi-AP fusion, fingerprinting (environment-aware), deep learning (DLoc-style), FTM for clock synchronization

### 6.4 Are Open-Source CSI Extraction Tools Available?

**Yes**:
- **FeitCSI** (Intel AX210) — open-source, modern, WiFi 6, 160 MHz
- **ESP-CSI** (ESP32) — official Espressif project, actively maintained
- **Nexmon CSI** (Broadcom/Pi) — actively maintained, 80 MHz
- **CSIKit** (Python) — unified parser for all formats
- **Atheros CSI Tool** — mature but 802.11n only

### 6.5 Hardware Recommendation

| Tier | Hardware | Cost | Capability |
|------|----------|------|------------|
| **Research** | Intel AX210 + FeitCSI | ~$30/node | WiFi 6, 160 MHz, 2x2 MIMO, 6 GHz — best accuracy |
| **Budget** | ESP32-S3 + esp-csi | ~$10-15/node | 2.4 GHz, 20 MHz, 1x1 — proof-of-concept, presence detection |
| **Pi-based** | Raspberry Pi 4 + Nexmon | ~$55-75/node | 80 MHz, 1x1 — easy deployment, no AoA |
| **Multi-antenna** | Asus RT-AC86U + Nexmon | ~$130/node | 80 MHz, 4x4 MIMO — best AoA without AX210 |

**Primary recommendation**: Start with 4x ESP32-S3 nodes for proof-of-concept (total ~$40-60), then upgrade to Intel AX210 + FeitCSI for high-accuracy localization work.

---

## 7. Key Papers & References

1. Kotaru et al., "SpotFi: Decimeter Level Localization Using WiFi" (SIGCOMM 2015)
2. Wu et al., "Widar3.0: Cross-Domain WiFi-based Activity Recognition" (MobiSys 2019)
3. Pu et al., "WiSee: Whole-Home Gesture Recognition Using WiFi" (MobiCom 2013)
4. Adib & Katabi, "WiVi: See Through Walls via WiFi" (SIGCOMM 2013)
5. Adib et al., "WiTrack: 3D Tracking via Body Radio Reflections" (NSDI 2014)
6. Adib et al., "RF-Capture: Capturing the Human Figure through Walls" (SIGCOMM 2015)
7. Zhao et al., "DLoc: Deep Learning Based Indoor Localization" (MobiCom 2020)
8. SPRING+: "Single-AP CSI+FTM Positioning" (IEEE TMC 2024)
9. ML-Track: "Multi-link WiFi Doppler Tracking" (IEEE TMC 2025)
10. IEEE 802.11az-2022: "Next Generation Positioning" standard
11. Halperin et al., "Linux 802.11n CSI Tool" (2011)
12. FeitCSI: https://github.com/KuskoSoft/FeitCSI
13. ESP-CSI: https://github.com/espressif/esp-csi
14. Nexmon CSI: https://github.com/seemoo-lab/nexmon_csi
15. CSIKit: https://github.com/Gi-z/CSIKit
16. CSI-Bench: https://github.com/Jenny-Zhu/CSI-Bench-Real-WiFi-Sensing-Benchmark