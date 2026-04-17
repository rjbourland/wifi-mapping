"""Fingerprinting — Radio map management and k-NN localization."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from gui.utils.theme import inject_theme, section_header
from gui.utils.data_loader import init_session_state, generate_synthetic_rssi

st.set_page_config(page_title="Fingerprinting", page_icon="🗺", layout="wide")
inject_theme()
init_session_state()

st.title("🗺 Fingerprinting")
st.markdown(
    '<span style="color:#888; font-family:Fira Code,Consolas,monospace; font-size:0.85rem;">'
    "Radio map management • k-NN localization • Calibration"
    "</span>",
    unsafe_allow_html=True,
)

fp = st.session_state.fingerprinting
anchors = st.session_state.anchors
room = st.session_state.room_dimensions

# --- Controls ---
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])

with col_ctrl1:
    k = st.slider("k (neighbors)", 1, 15, 5, key="fp_k")
    fp.k = k

with col_ctrl2:
    if st.button("▶ Auto-Generate Radio Map", type="primary", use_container_width=True):
        _generate_radio_map()
        st.toast("Radio map generated!")

with col_ctrl3:
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🎓 Train k-NN", use_container_width=True):
            try:
                fp.train()
                st.toast(f"Trained k-NN with k={k}")
            except RuntimeError as e:
                st.error(str(e))
    with col_btn2:
        if st.button("🗑 Clear Database", use_container_width=True):
            st.session_state.fingerprinting = __import__(
                "src.localization.fingerprinting", fromlist=["KNNFingerprinting"]
            ).KNNFingerprinting()
            fp = st.session_state.fingerprinting
            st.rerun()

# --- Database Stats ---
section_header("Radio Map Database", "📊")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    db_size = len(fp._fingerprint_db) if fp._fingerprint_db is not None else 0
    st.metric("Fingerprints", db_size)
with col_s2:
    st.metric("Anchors Used", len(fp._anchor_ids) if fp._anchor_ids else 0)
with col_s3:
    st.metric("Trained", "Yes" if fp._is_trained else "No")

# --- Radio Map Heatmap ---
section_header("RSSI Radio Map", "🌡")

if fp._fingerprint_db is not None and len(fp._fingerprint_db) > 0:
    # Show radio map as heatmap
    fig_radio = go.Figure()

    # X-Y positions colored by RSSI from each anchor
    positions = fp._positions
    fingerprints = fp._fingerprint_db

    anchor_select = st.selectbox(
        "Show RSSI from Anchor",
        fp._anchor_ids if fp._anchor_ids else ["N/A"],
        key="fp_anchor_select",
    )

    if anchor_select != "N/A" and anchor_select in fp._anchor_ids:
        anchor_idx = list(fp._anchor_ids).index(anchor_select)
        rssi_values = fingerprints[:, anchor_idx]

        fig_radio.add_trace(go.Scatter(
            x=positions[:, 0], y=positions[:, 1],
            mode="markers",
            marker=dict(
                size=12,
                color=rssi_values,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="RSSI (dBm)"),
            ),
            text=[f"({p[0]:.1f},{p[1]:.1f}) RSSI: {r:.1f}" for p, r in zip(positions, rssi_values)],
            name="Fingerprint Points",
        ))

    fig_radio.update_layout(
        xaxis_title="X (m)", yaxis_title="Y (m)",
        xaxis=dict(range=[-0.5, room["length_x"] + 0.5], gridcolor="#1a1a2e"),
        yaxis=dict(range=[-0.5, room["width_y"] + 0.5], gridcolor="#1a1a2e", scaleanchor="x", scaleratio=1),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", family="Fira Code"),
        height=400,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_radio, use_container_width=True)
else:
    st.info("No fingerprints in database. Generate a radio map or add calibration points manually.")

# --- Manual Fingerprint Entry ---
section_header("Add Fingerprint Point", "➕")

with st.form("add_fingerprint"):
    col_fp1, col_fp2, col_fp3 = st.columns(3)
    with col_fp1:
        fp_x = st.number_input("X (m)", 0.0, room["length_x"], 2.0, 0.1, key="fp_x")
    with col_fp2:
        fp_y = st.number_input("Y (m)", 0.0, room["width_y"], 2.0, 0.1, key="fp_y")
    with col_fp3:
        fp_z = st.number_input("Z (m)", 0.0, room["height_z"], 1.0, 0.1, key="fp_z")

    st.markdown("**RSSI Values (dBm)**")
    rssi_cols = st.columns(min(len(anchors), 4))
    rssi_inputs = {}
    for i, anchor in enumerate(anchors):
        with rssi_cols[i % 4]:
            rssi_inputs[anchor.anchor_id] = st.number_input(
                anchor.anchor_id, -100.0, 0.0, -50.0, 1.0, key=f"fp_rssi_{anchor.anchor_id}"
            )

    submitted = st.form_submit_button("Add Fingerprint")
    if submitted:
        fp.add_fingerprint(rssi_inputs, np.array([fp_x, fp_y, fp_z]))
        st.toast(f"Added fingerprint at ({fp_x}, {fp_y}, {fp_z})")

# --- Test Localization ---
if fp._is_trained:
    section_header("Test Localization", "🧪")

    with st.form("test_fp_localize"):
        st.markdown("**Enter RSSI values to estimate position:**")
        test_cols = st.columns(min(len(anchors), 4))
        test_rssi = {}
        for i, anchor in enumerate(anchors):
            with test_cols[i % 4]:
                test_rssi[anchor.anchor_id] = st.number_input(
                    f"{anchor.anchor_id} (dBm)", -100.0, 0.0, -50.0, 1.0, key=f"test_rssi_{anchor.anchor_id}"
                )

        test_submitted = st.form_submit_button("Estimate Position")
        if test_submitted:
            result = fp.localize(test_rssi)
            st.markdown(
                f'<div style="background:#1a1a2e; border:1px solid #2a2a4e; border-radius:8px; padding:16px; '
                f'font-family:Fira Code,Consolas,monospace;">'
                f'<div style="color:#00d4ff; font-size:1.3rem;">'
                f"Position: ({result.position[0]:.2f}, {result.position[1]:.2f}, {result.position[2]:.2f})</div>"
                f'<div style="color:#888; font-size:0.8rem; margin-top:8px;">'
                f"Confidence: {result.confidence:.2f} | Anchors: {', '.join(result.anchors_used)}</div></div>",
                unsafe_allow_html=True,
            )


def _generate_radio_map():
    """Generate a radio map with fingerprint points on a grid."""
    grid_spacing = 0.5  # meters
    x_points = np.arange(grid_spacing, room["length_x"], grid_spacing)
    y_points = np.arange(grid_spacing, room["width_y"], grid_spacing)
    z = 1.0  # Fixed height for 2D radio map

    count = 0
    for x in x_points:
        for y in y_points:
            pos = np.array([x, y, z])
            rssi = generate_synthetic_rssi(anchors, pos, noise_std=2.0)
            fp.add_fingerprint(rssi, pos)
            count += 1

    # Reset trained state since we added new data
    st.session_state.fingerprinting._is_trained = False