"""Settings — Configuration and hardware setup."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import yaml
import numpy as np

from gui.utils.theme import inject_theme, section_header
from gui.utils.data_loader import init_session_state
from src.utils.data_formats import AnchorPosition

st.set_page_config(page_title="Settings", page_icon="⚙", layout="wide")
inject_theme()
init_session_state()

st.title("⚙ Settings")
st.markdown(
    '<span style="color:#888; font-family:Fira Code,Consolas,monospace; font-size:0.85rem;">'
    "Anchor configuration • Algorithm parameters • Hardware setup"
    "</span>",
    unsafe_allow_html=True,
)

# --- Anchor Configuration ---
section_header("Anchor Configuration", "📍")

anchors = st.session_state.anchors
room = st.session_state.room_dimensions

# Room dimensions
st.markdown("**Room Dimensions**")
col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    new_rx = st.number_input("Length X (m)", 1.0, 20.0, float(room["length_x"]), 0.5, key="set_rx")
with col_r2:
    new_ry = st.number_input("Width Y (m)", 1.0, 20.0, float(room["width_y"]), 0.5, key="set_ry")
with col_r3:
    new_rz = st.number_input("Height Z (m)", 1.0, 5.0, float(room["height_z"]), 0.1, key="set_rz")

# Anchor positions (now editable)
st.markdown("**Anchor Positions**")
updated_anchors = []
for i, anchor in enumerate(anchors):
    with st.expander(f"{anchor.anchor_id} — {anchor.height} ({anchor.hardware})", expanded=False):
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            ax = st.number_input(
                "X", 0.0, 20.0, float(anchor.position[0]), 0.1,
                key=f"anchor_x_{i}",
            )
        with col_a2:
            ay = st.number_input(
                "Y", 0.0, 20.0, float(anchor.position[1]), 0.1,
                key=f"anchor_y_{i}",
            )
        with col_a3:
            az = st.number_input(
                "Z", 0.0, 5.0, float(anchor.position[2]), 0.1,
                key=f"anchor_z_{i}",
            )
        # Store updated anchor
        updated_anchors.append(AnchorPosition(
            anchor_id=anchor.anchor_id,
            position=np.array([ax, ay, az]),
            height=anchor.height,
            hardware=anchor.hardware,
            ip=anchor.ip,
            channel=anchor.channel,
            bandwidth=anchor.bandwidth,
        ))
        st.markdown(
            f'<span style="color:#888; font-family:Fira Code,Consolas,monospace; font-size:0.8rem;">'
            f"Hardware: {anchor.hardware} | Channel: {anchor.channel} | "
            f"Bandwidth: {anchor.bandwidth} MHz | IP: {anchor.ip or 'not set'}</span>",
            unsafe_allow_html=True,
        )

col_btn_anchor1, col_btn_anchor2 = st.columns(2)
with col_btn_anchor1:
    if st.button("💾 Save Anchor Config", use_container_width=True):
        config_path = Path(PROJECT_ROOT) / "configs" / "anchors.yaml"
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            config["room"]["length_x"] = new_rx
            config["room"]["width_y"] = new_ry
            config["room"]["height_z"] = new_rz

            for anchor in updated_anchors:
                aid = anchor.anchor_id
                if aid in config.get("anchors", {}):
                    config["anchors"][aid]["position"] = anchor.position.tolist()

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            st.toast("Anchor configuration saved!")
        except Exception as e:
            st.error(f"Failed to save config: {e}")

with col_btn_anchor2:
    if st.button("⚡ Apply to Session", use_container_width=True):
        # Update session state with new anchor positions
        st.session_state.anchors = updated_anchors
        st.session_state.room_dimensions = {
            "length_x": new_rx,
            "width_y": new_ry,
            "height_z": new_rz,
        }
        st.session_state.trilateration_solver.set_anchors(updated_anchors)
        st.toast("Applied anchor positions to live session!")

# --- Algorithm Parameters ---
section_header("Algorithm Parameters", "🎛")

try:
    from src.utils.config import load_config
    algo_config = load_config("algorithm")
except FileNotFoundError:
    algo_config = {}

# Path-loss model
st.markdown("**Path-Loss Model**")
col_pl1, col_pl2, col_pl3, col_pl4 = st.columns(4)
pl_config = algo_config.get("path_loss", {})
with col_pl1:
    st.number_input("Reference RSSI (dBm)", -50.0, 0.0,
                    float(pl_config.get("rssi_d0", -30.0)), 1.0, key="set_rssi_d0")
with col_pl2:
    st.number_input("Path-Loss Exponent (n)", 1.5, 6.0,
                    float(pl_config.get("n", 3.0)), 0.1, key="set_pl_n")
with col_pl3:
    st.number_input("Reference Distance (m)", 0.1, 10.0,
                    float(pl_config.get("d0", 1.0)), 0.1, key="set_pl_d0")
with col_pl4:
    st.number_input("Shadow Fading σ (dB)", 0.5, 10.0,
                    float(pl_config.get("sigma", 4.0)), 0.5, key="set_pl_sigma")

# Trilateration
st.markdown("**Trilateration**")
tri_config = algo_config.get("trilateration", {})
col_tri1, col_tri2 = st.columns(2)
with col_tri1:
    st.selectbox("Solver", ["lm", "trf"],
                 index=["lm", "trf"].index(tri_config.get("solver", "lm")),
                 key="set_tri_solver")
with col_tri2:
    st.selectbox("Initial Guess", ["centroid", "measured"],
                 index=["centroid", "measured"].index(tri_config.get("initial_guess", "centroid")),
                 key="set_tri_guess")

# Fingerprinting
st.markdown("**Fingerprinting (k-NN)**")
fp_config = algo_config.get("fingerprinting", {})
col_fp1, col_fp2, col_fp3 = st.columns(3)
with col_fp1:
    st.slider("k (neighbors)", 1, 20,
              int(fp_config.get("k", 5)), 1, key="set_fp_k")
with col_fp2:
    st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"],
                 index=["euclidean", "manhattan", "cosine"].index(fp_config.get("distance_metric", "euclidean")),
                 key="set_fp_metric")
with col_fp3:
    st.selectbox("Feature Type", ["rssi", "csi_amplitude", "csi_phase", "combined"],
                 index=["rssi", "csi_amplitude", "csi_phase", "combined"].index(fp_config.get("feature_type", "rssi")),
                 key="set_fp_feature")

# Kalman
st.markdown("**Kalman Filter**")
kalman_config = algo_config.get("kalman", {})
col_k1, col_k2, col_k3 = st.columns(3)
with col_k1:
    st.number_input("Process Noise", 0.001, 10.0,
                    float(kalman_config.get("process_noise", 0.1)), 0.01, key="set_kalman_pn")
with col_k2:
    st.number_input("Measurement Noise", 0.1, 50.0,
                    float(kalman_config.get("measurement_noise", 1.0)), 0.1, key="set_kalman_mn")
with col_k3:
    st.number_input("Initial Uncertainty", 0.1, 100.0,
                    float(kalman_config.get("initial_uncertainty", 10.0)), 1.0, key="set_kalman_iu")

# Motion detection
st.markdown("**Motion Detection**")
mot_config = algo_config.get("motion_detection", {})
col_mo1, col_mo2 = st.columns(2)
with col_mo1:
    st.number_input("Variance Threshold", 0.1, 5.0,
                    float(mot_config.get("variance_threshold", 0.5)), 0.1, key="set_mo_thresh")
with col_mo2:
    st.number_input("Window Size (samples)", 5, 100,
                    int(mot_config.get("window_size", 20)), 5, key="set_mo_window")

col_btn_algo1, col_btn_algo2 = st.columns(2)
with col_btn_algo1:
    if st.button("💾 Save Algorithm Config", use_container_width=True):
        config_path = Path(PROJECT_ROOT) / "configs" / "algorithm.yaml"
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            config["path_loss"]["rssi_d0"] = st.session_state.get("set_rssi_d0", -30.0)
            config["path_loss"]["n"] = st.session_state.get("set_pl_n", 3.0)
            config["path_loss"]["d0"] = st.session_state.get("set_pl_d0", 1.0)
            config["path_loss"]["sigma"] = st.session_state.get("set_pl_sigma", 4.0)

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            st.toast("Algorithm configuration saved!")
        except Exception as e:
            st.error(f"Failed to save config: {e}")

with col_btn_algo2:
    if st.button("⚡ Apply Algorithm Params to Session", use_container_width=True):
        # Propagate settings to live objects
        solver = st.session_state.trilateration_solver
        solver.n = st.session_state.get("set_pl_n", 3.0)
        solver.rssi_d0 = st.session_state.get("set_rssi_d0", -30.0)

        kalman = st.session_state.kalman_filter
        kalman.process_noise = st.session_state.get("set_kalman_pn", 0.1)
        kalman.measurement_noise = st.session_state.get("set_kalman_mn", 1.0)

        fp = st.session_state.fingerprinting
        fp.k = st.session_state.get("set_fp_k", 5)

        md = st.session_state.motion_detector
        md.variance_threshold = st.session_state.get("set_mo_thresh", 0.5)
        md.window_size = st.session_state.get("set_mo_window", 20)

        st.toast("Algorithm parameters applied to live session!")

# --- Hardware Configuration ---
section_header("Hardware Configuration", "🔌")

st.markdown("**ESP32 Serial Settings**")
col_hw1, col_hw2 = st.columns(2)
with col_hw1:
    st.text_input("Serial Port", value="COM3", key="set_serial_port")
with col_hw2:
    st.number_input("Baud Rate", value=921600, step=9600, key="set_baud")

st.markdown("**Intel AX210 / FeitCSI Settings**")
col_hw3, col_hw4 = st.columns(2)
with col_hw3:
    st.text_input("Network Interface", value="wlan0", key="set_interface")
with col_hw4:
    st.number_input("Channel", 1, 165, 36, key="set_channel")

st.markdown("**Raspberry Pi / Nexmon Settings**")
col_hw5, col_hw6 = st.columns(2)
with col_hw5:
    st.text_input("Monitor Interface", value="mon0", key="set_mon_if")
with col_hw6:
    st.number_input("Bandwidth (MHz)", 20, 80, 80, step=20, key="set_nex_bw")