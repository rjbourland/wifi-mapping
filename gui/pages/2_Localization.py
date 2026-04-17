"""Localization — Floor plan view with position estimates and trilateration."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from gui.utils.theme import inject_theme, section_header
from gui.utils.data_loader import (
    init_session_state,
    generate_synthetic_rssi,
    generate_synthetic_csi,
    positions_to_dataframe,
    rssi_history_to_dataframe,
)

st.set_page_config(page_title="Localization", page_icon="📍", layout="wide")
inject_theme()
init_session_state()

st.title("📍 Localization")
st.markdown(
    '<span style="color:#888; font-family:Fira Code,Consolas,monospace; font-size:0.85rem;">'
    "RSSI trilateration • CSI fingerprinting • Kalman tracking"
    "</span>",
    unsafe_allow_html=True,
)

# --- Controls ---
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])

with col_ctrl1:
    method = st.selectbox("Method", ["trilateration", "fingerprinting"], key="loc_method")
    use_kalman = st.toggle("Kalman Filter", value=True, key="loc_kalman")

with col_ctrl2:
    noise_level = st.slider("RSSI Noise (dB)", 0.5, 10.0, 3.0, 0.5, key="loc_noise")
    path_loss_n = st.slider("Path-Loss Exponent", 2.0, 5.0, 3.0, 0.1, key="loc_pathloss")

with col_ctrl3:
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("▶ Simulate Single Estimate", use_container_width=True):
            _run_single_estimate()
    with col_btn2:
        if st.button("▶▶ Simulate Path (20 pts)", use_container_width=True):
            _run_path_simulation()
    with col_btn3:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.localized_positions = []
            st.session_state.rssi_history = []
            st.session_state.last_position = None
            st.rerun()

# --- Floor Plan ---
section_header("Floor Plan", "🗺")

room = st.session_state.room_dimensions
anchors = st.session_state.anchors

fig = go.Figure()

# Room outline
rx, ry = room["length_x"], room["width_y"]
fig.add_shape(type="rect", x0=0, y0=0, x1=rx, y1=ry,
              line=dict(color="#2a2a4e", width=2), fillcolor="#0e111722")

# Trilateration circles (if we have RSSI data and a position)
if st.session_state.last_position is not None and st.session_state.rssi_history:
    latest_rssi = {}
    for entry in reversed(st.session_state.rssi_history):
        if entry["anchor_id"] not in latest_rssi:
            latest_rssi[entry["anchor_id"]] = entry["rssi"]

    from src.utils.math_utils import path_loss_distance
    theta = np.linspace(0, 2 * np.pi, 100)
    for anchor in anchors:
        if anchor.anchor_id in latest_rssi:
            dist = path_loss_distance(latest_rssi[anchor.anchor_id], n=path_loss_n)
            x_circle = anchor.position[0] + dist * np.cos(theta)
            y_circle = anchor.position[1] + dist * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=x_circle, y=y_circle,
                mode="lines",
                line=dict(color="#00d4ff", width=1, dash="dot"),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ))

# Anchors
for anchor in anchors:
    color = {"ceiling": "#00d4ff", "mid": "#ffaa00", "floor": "#00ff88"}.get(anchor.height, "#888")
    fig.add_trace(go.Scatter(
        x=[anchor.position[0]], y=[anchor.position[1]],
        mode="markers+text",
        marker=dict(size=14, color=color, symbol="triangle-up"),
        text=[anchor.anchor_id],
        textposition="top center",
        textfont=dict(color=color, size=10, family="Fira Code"),
        name=anchor.anchor_id,
    ))

# Trajectory
if st.session_state.localized_positions:
    positions = np.array([p.position for p in st.session_state.localized_positions])
    fig.add_trace(go.Scatter(
        x=positions[:, 0], y=positions[:, 1],
        mode="lines+markers",
        line=dict(color="#6c63ff", width=2),
        marker=dict(size=4, color="#6c63ff"),
        name="Trajectory",
    ))

# Current position
if st.session_state.last_position is not None:
    pos = st.session_state.last_position
    fig.add_trace(go.Scatter(
        x=[pos[0]], y=[pos[1]],
        mode="markers",
        marker=dict(size=16, color="#ff4444", symbol="x", line=dict(width=3)),
        name="Current",
    ))

fig.update_layout(
    xaxis_title="X (m)", yaxis_title="Y (m)",
    xaxis=dict(range=[-0.5, rx + 0.5], gridcolor="#1a1a2e", zeroline=False),
    yaxis=dict(range=[-0.5, ry + 0.5], gridcolor="#1a1a2e", scaleanchor="x", scaleratio=1, zeroline=False),
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(color="#e0e0e0", family="Fira Code"),
    height=500, showlegend=True,
    legend=dict(font=dict(color="#888", size=9), orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=20, t=30, b=50),
)

st.plotly_chart(fig, use_container_width=True)

# --- Position Details ---
section_header("Position Details", "📐")

col_det1, col_det2 = st.columns([3, 2])

with col_det1:
    st.markdown("**Recent Position Estimates**")
    if st.session_state.localized_positions:
        df = positions_to_dataframe(st.session_state.localized_positions[-20:])
        st.dataframe(df, hide_index=True, use_container_width=True, height=300)
    else:
        st.info("No estimates yet. Run a simulation above.")

with col_det2:
    st.markdown("**Current Estimate**")
    if st.session_state.last_position is not None:
        pos = st.session_state.last_position
        st.markdown(
            f'<div style="background:#1a1a2e; border:1px solid #2a2a4e; border-radius:8px; padding:16px; '
            f'font-family:Fira Code,Consolas,monospace;">'
            f'<div style="color:#00d4ff; font-size:1.2rem;">X: {pos[0]:.3f} m</div>'
            f'<div style="color:#00ff88; font-size:1.2rem;">Y: {pos[1]:.3f} m</div>'
            f'<div style="color:#ffaa00; font-size:1.2rem;">Z: {pos[2]:.3f} m</div>'
            f'<div style="color:#888; font-size:0.8rem; margin-top:8px;">'
            f"Method: {st.session_state.last_method} | Kalman: {'ON' if use_kalman else 'OFF'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No position estimate available.")

    # RSSI from each anchor
    st.markdown("**RSSI per Anchor**")
    if st.session_state.rssi_history:
        rssi_df = rssi_history_to_dataframe(st.session_state.rssi_history)
        latest = rssi_df.drop_duplicates(subset=["anchor_id"], keep="last")
        for _, row in latest.iterrows():
            color = "#ff4444" if row["rssi"] < -70 else "#ffaa00" if row["rssi"] < -50 else "#00ff88"
            st.markdown(
                f'<span style="color:{color}; font-family:Fira Code,Consolas,monospace; font-size:0.85rem;">'
                f'{row["anchor_id"]}: {row["rssi"]:.1f} dBm</span>',
                unsafe_allow_html=True,
            )


def _run_single_estimate():
    room = st.session_state.room_dimensions
    solver = st.session_state.trilateration_solver
    solver.n = path_loss_n
    anchors = st.session_state.anchors

    # Random position in room
    true_pos = np.array([
        np.random.uniform(0.5, room["length_x"] - 0.5),
        np.random.uniform(0.5, room["width_y"] - 0.5),
        np.random.uniform(0.3, room["height_z"] - 0.3),
    ])

    rssi = generate_synthetic_rssi(anchors, true_pos, noise_std=noise_level, n=path_loss_n)

    result = solver.localize(rssi)
    if use_kalman:
        result.position = st.session_state.kalman_filter.update(result.position)

    st.session_state.localized_positions.append(result)
    st.session_state.last_position = result.position
    st.session_state.last_method = method

    for aid, val in rssi.items():
        st.session_state.rssi_history.append({
            "timestamp": datetime.now().isoformat(),
            "anchor_id": aid,
            "rssi": val,
        })


def _run_path_simulation():
    room = st.session_state.room_dimensions
    solver = st.session_state.trilateration_solver
    solver.n = path_loss_n
    anchors = st.session_state.anchors

    center = np.array([room["length_x"] / 2, room["width_y"] / 2, 1.2])
    radius = min(room["length_x"], room["width_y"]) / 3

    for i in range(20):
        angle = 2 * np.pi * i / 20
        true_pos = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.2 * np.sin(angle * 2),
        ])

        rssi = generate_synthetic_rssi(anchors, true_pos, noise_std=noise_level, n=path_loss_n)
        result = solver.localize(rssi)

        if use_kalman:
            result.position = st.session_state.kalman_filter.update(result.position)

        st.session_state.localized_positions.append(result)
        st.session_state.last_position = result.position
        st.session_state.last_method = method

        for aid, val in rssi.items():
            st.session_state.rssi_history.append({
                "timestamp": datetime.now().isoformat(),
                "anchor_id": aid,
                "rssi": val,
            })