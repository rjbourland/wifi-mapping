"""WiFi Mapping Dashboard — Home Page."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from gui.utils.theme import inject_theme, status_badge, section_header
from gui.utils.data_loader import (
    init_session_state,
    generate_synthetic_rssi,
    positions_to_dataframe,
)

st.set_page_config(
    page_title="WiFi Mapping",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📡 WiFi Mapping")
    st.markdown("---")

    mode = st.toggle("Simulation Mode", value=st.session_state.simulation_mode, key="sim_toggle")
    st.session_state.simulation_mode = mode

    if mode:
        st.markdown('<span style="color:#ffaa00; font-size:0.8rem;">⚡ Using synthetic data</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#00ff88; font-size:0.8rem;">🔌 Hardware connected</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Actions")

    if st.button("▶ Simulate Tracking", use_container_width=True):
        _run_simulation()

    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.localized_positions = []
        st.session_state.rssi_history = []
        st.session_state.motion_events = []
        st.session_state.last_position = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Anchors")
    for anchor in st.session_state.anchors:
        st.markdown(
            f"**{anchor.anchor_id}** "
            f'<span style="color:#888; font-size:0.75rem;">'
            f"({anchor.position[0]:.0f},{anchor.position[1]:.0f},{anchor.position[2]:.1f})m "
            f"| {anchor.hardware}"
            f"</span>",
            unsafe_allow_html=True,
        )


# --- Main Content ---
st.title("WiFi Signal Triangulation & 3D Indoor Mapping")
st.markdown(
    '<span style="color:#888; font-family:Fira Code,Consolas,monospace; font-size:0.9rem;">'
    "Real-time indoor positioning using WiFi CSI • RSSI trilateration • Device-free sensing"
    "</span>",
    unsafe_allow_html=True,
)

# --- KPI Metrics ---
section_header("System Status", "📊")

col1, col2, col3, col4, col5 = st.columns(5)

num_anchors = len(st.session_state.anchors)
last_pos = st.session_state.last_position
last_method = st.session_state.last_method or "none"
num_positions = len(st.session_state.localized_positions)
motion_active = any(e.get("is_motion", False) for e in st.session_state.motion_events[-1:]) if st.session_state.motion_events else False

with col1:
    st.metric("Anchors Online", f"{num_anchors}", help="Number of configured anchor nodes")
with col2:
    if last_pos is not None:
        st.metric("Last Position", f"({last_pos[0]:.2f}, {last_pos[1]:.2f}, {last_pos[2]:.2f})")
    else:
        st.metric("Last Position", "—")
with col3:
    st.metric("Method", last_method.upper() if last_method != "none" else "—")
with col4:
    st.metric("Estimates", f"{num_positions}")
with col5:
    motion_status = "ACTIVE" if motion_active else "CLEAR"
    motion_color = "#ff4444" if motion_active else "#00ff88"
    st.metric("Motion", motion_status)

# --- Anchor Status Cards ---
section_header("Anchor Nodes", "📍")

anchor_cols = st.columns(min(num_anchors, 4))
for i, anchor in enumerate(st.session_state.anchors):
    with anchor_cols[i % 4]:
        height_icon = {"ceiling": "⬆", "mid": "↔", "floor": "⬇"}.get(anchor.height, "?")
        st.markdown(
            f'<div style="background:#1a1a2e; border:1px solid #2a2a4e; border-radius:8px; padding:12px;">'
            f'<div style="color:#00d4ff; font-family:Fira Code,Consolas,monospace; font-size:0.9rem;">'
            f"{height_icon} {anchor.anchor_id}</div>"
            f'<div style="color:#888; font-size:0.75rem; font-family:Fira Code,Consolas,monospace;">'
            f"Position: ({anchor.position[0]:.1f}, {anchor.position[1]:.1f}, {anchor.position[2]:.1f})m<br>"
            f"Hardware: {anchor.hardware}<br>"
            f"Ch: {anchor.channel} | BW: {anchor.bandwidth}MHz</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# --- Room Overview ---
section_header("Room Overview", "🏠")

col_left, col_right = st.columns([2, 1])

with col_left:
    room = st.session_state.room_dimensions
    fig = go.Figure()

    # Draw room outline
    rx, ry = room["length_x"], room["width_y"]
    fig.add_shape(type="rect", x0=0, y0=0, x1=rx, y1=ry,
                  line=dict(color="#2a2a4e", width=2), fillcolor="#0e1117")

    # Draw anchors
    for anchor in st.session_state.anchors:
        color = {"ceiling": "#00d4ff", "mid": "#ffaa00", "floor": "#00ff88"}.get(anchor.height, "#888")
        fig.add_trace(go.Scatter(
            x=[anchor.position[0]], y=[anchor.position[1]],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol="triangle-up"),
            text=[anchor.anchor_id],
            textposition="top center",
            textfont=dict(color=color, size=10, family="Fira Code"),
            name=f"{anchor.anchor_id} ({anchor.height})",
        ))

    # Draw trajectory
    if st.session_state.localized_positions:
        positions = np.array([p.position for p in st.session_state.localized_positions[-50:]])
        fig.add_trace(go.Scatter(
            x=positions[:, 0], y=positions[:, 1],
            mode="lines+markers",
            line=dict(color="#6c63ff", width=2),
            marker=dict(size=4),
            name="Trajectory",
        ))

    # Draw current position
    if last_pos is not None:
        fig.add_trace(go.Scatter(
            x=[last_pos[0]], y=[last_pos[1]],
            mode="markers",
            marker=dict(size=16, color="#ff4444", symbol="x"),
            name="Current Position",
        ))

    fig.update_layout(
        xaxis_title="X (m)", yaxis_title="Y (m)",
        xaxis=dict(range=[-0.5, rx + 0.5], gridcolor="#1a1a2e"),
        yaxis=dict(range=[-0.5, ry + 0.5], gridcolor="#1a1a2e", scaleanchor="x", scaleratio=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", family="Fira Code"),
        height=400,
        showlegend=True,
        legend=dict(font=dict(color="#888", size=9)),
        margin=dict(l=40, r=20, t=20, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

with col_right:
    # Recent positions table
    st.markdown("**Recent Estimates**")
    if st.session_state.localized_positions:
        df = positions_to_dataframe(st.session_state.localized_positions[-10:])
        st.dataframe(
            df[["timestamp", "x", "y", "z", "method"]],
            hide_index=True,
            use_container_width=True,
            height=300,
        )
    else:
        st.info("No position estimates yet. Click 'Simulate Tracking' to generate data.")

    # Room info
    st.markdown("**Room Dimensions**")
    st.markdown(
        f'<div style="font-family:Fira Code,Consolas,monospace; font-size:0.8rem; color:#888;">'
        f"Length (X): {room['length_x']:.1f}m<br>"
        f"Width (Y): {room['width_y']:.1f}m<br>"
        f"Height (Z): {room['height_z']:.1f}m</div>",
        unsafe_allow_html=True,
    )


# --- Helper ---
def _run_simulation():
    """Run a simulated tracking session."""
    import time

    room = st.session_state.room_dimensions
    solver = st.session_state.trilateration_solver
    anchors = st.session_state.anchors

    # Simulate a circular path through the room
    center = np.array([room["length_x"] / 2, room["width_y"] / 2, 1.2])
    radius = min(room["length_x"], room["width_y"]) / 3

    for i in range(20):
        angle = 2 * np.pi * i / 20
        true_pos = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.2 * np.sin(angle * 2),  # Slight vertical oscillation
        ])

        rssi = generate_synthetic_rssi(anchors, true_pos, noise_std=2.0)
        result = solver.localize(rssi)
        st.session_state.localized_positions.append(result)
        st.session_state.last_position = result.position
        st.session_state.last_method = result.method

        # Add to RSSI history
        for aid, rssi_val in rssi.items():
            st.session_state.rssi_history.append({
                "timestamp": datetime.now().isoformat(),
                "anchor_id": aid,
                "rssi": rssi_val,
            })

        # Check motion
        csi = generate_synthetic_csi(50, motion=(i % 3 == 0))
        motion_result = st.session_state.motion_detector.detect(csi)
        st.session_state.motion_events.append(motion_result)

    st.session_state.point_cloud.add_points(
        np.array([p.position for p in st.session_state.localized_positions[-20:]])
    )