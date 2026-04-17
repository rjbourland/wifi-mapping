"""Live Collection — Real-time CSI/RSSI data collection interface."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from gui.utils.theme import inject_theme, section_header, status_badge
from gui.utils.data_loader import (
    init_session_state,
    generate_synthetic_rssi,
    generate_synthetic_csi,
    rssi_history_to_dataframe,
)

st.set_page_config(page_title="Live Collection", page_icon="📡", layout="wide")
inject_theme()
init_session_state()

st.title("📡 Live Collection")
st.markdown(
    '<span style="color:#888; font-family:Fira Code,Consolas,monospace; font-size:0.85rem;">'
    "CSI/RSSI data collection • Signal monitoring • Hardware control"
    "</span>",
    unsafe_allow_html=True,
)

# --- Connection Status ---
section_header("Connection", "🔌")

col_conn1, col_conn2, col_conn3 = st.columns([1, 1, 2])

with col_conn1:
    source = st.selectbox("Data Source", ["Simulation", "ESP32 Serial", "AX210 UDP"], key="coll_source")

with col_conn2:
    if source == "ESP32 Serial":
        serial_port = st.text_input("Serial Port", value="COM3", key="coll_serial")
        baud = st.number_input("Baud Rate", value=921600, key="coll_baud")
    elif source == "AX210 UDP":
        udp_port = st.number_input("UDP Port", value=5500, key="coll_udp")
    else:
        st.markdown(status_badge("SIMULATION", "warning"), unsafe_allow_html=True)

with col_conn3:
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("▶ Start Collection", type="primary", use_container_width=True):
            st.session_state.collection_active = True
            st.toast("Collection started")
    with col_btn2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.collection_active = False
            st.toast("Collection stopped")
    with col_btn3:
        if st.button("📊 Generate Sample Data", use_container_width=True):
            _generate_sample_data()

# --- RSSI Time Series ---
section_header("RSSI Time Series", "📈")

if st.session_state.rssi_history:
    rssi_df = rssi_history_to_dataframe(st.session_state.rssi_history)

    fig_rssi = go.Figure()
    for anchor_id in rssi_df["anchor_id"].unique():
        anchor_data = rssi_df[rssi_df["anchor_id"] == anchor_id]
        fig_rssi.add_trace(go.Scatter(
            x=anchor_data["timestamp"],
            y=anchor_data["rssi"],
            mode="lines+markers",
            name=anchor_id,
            line=dict(width=2),
        ))

    fig_rssi.update_layout(
        xaxis_title="Time", yaxis_title="RSSI (dBm)",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", family="Fira Code"),
        height=350,
        legend=dict(font=dict(color="#888", size=9)),
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_rssi, use_container_width=True)
else:
    st.info("No RSSI data collected yet. Start collection or generate sample data.")

# --- CSI Amplitude Heatmap ---
section_header("CSI Amplitude Heatmap", "🌡")

col_heat1, col_heat2 = st.columns([3, 1])

with col_heat1:
    num_packets = st.slider("Display Packets", 10, 200, 50, key="coll_packets")
    csi_data = generate_synthetic_csi(num_packets, motion=st.session_state.collection_active)

    # Average across antennas for heatmap
    amplitude = np.abs(csi_data).mean(axis=1)  # (num_packets, num_subcarriers)

    fig_heat = px.imshow(
        amplitude.T,
        labels=dict(x="Packet #", y="Subcarrier", color="Amplitude"),
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig_heat.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", family="Fira Code"),
        height=350,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_heat2:
    st.markdown("**Collection Stats**")
    total_samples = len(st.session_state.rssi_history)
    st.metric("Total Samples", total_samples)
    st.metric("CSI Packets", num_packets)
    st.metric("Subcarriers", 52)

    active = st.session_state.collection_active
    st.markdown(
        status_badge("ACTIVE" if active else "IDLE", "online" if active else "offline"),
        unsafe_allow_html=True,
    )

    # Per-anchor RSSI stats
    if st.session_state.rssi_history:
        st.markdown("**RSSI Stats**")
        rssi_df = rssi_history_to_dataframe(st.session_state.rssi_history)
        for aid in rssi_df["anchor_id"].unique():
            anchor_rssi = rssi_df[rssi_df["anchor_id"] == aid]["rssi"]
            st.markdown(
                f'<div style="font-family:Fira Code,Consolas,monospace; font-size:0.75rem; color:#888;">'
                f"{aid}: mean={anchor_rssi.mean():.1f} std={anchor_rssi.std():.1f} dBm</div>",
                unsafe_allow_html=True,
            )


# --- Helper ---
def _generate_sample_data():
    """Generate sample RSSI and CSI data for testing the collection view."""
    anchors = st.session_state.anchors
    room = st.session_state.room_dimensions

    # Simulate RSSI from a point moving along a path
    for i in range(30):
        t = i / 30.0
        pos = np.array([
            room["length_x"] * t,
            room["width_y"] * (0.3 + 0.4 * np.sin(2 * np.pi * t)),
            1.0 + 0.5 * np.sin(4 * np.pi * t),
        ])
        rssi = generate_synthetic_rssi(anchors, pos, noise_std=3.0)
        for aid, val in rssi.items():
            st.session_state.rssi_history.append({
                "timestamp": datetime.now().isoformat(),
                "anchor_id": aid,
                "rssi": val,
            })

    st.toast(f"Generated 30 sample readings across {len(anchors)} anchors")