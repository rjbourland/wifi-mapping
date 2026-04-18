"""3D Map — Point cloud, floor plan, RSSI heatmap, and occupancy grid visualization."""

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
from src.mapping.floor_plan import FloorPlanMapper
from src.mapping.heatmap import HeatmapGenerator
from src.mapping.occupancy_grid import OccupancyGrid
from src.mapping.visualization import Visualizer
from src.mapping.adapters import positions_to_array

st.set_page_config(page_title="3D Map", page_icon="🧊", layout="wide")
inject_theme()
init_session_state()

st.title("🧊 3D Map")
st.markdown(
    '<span style="color:#888; font-family:Fira Code,Consolas,monospace; font-size:0.85rem;">'
    "Point cloud • Floor plan • RSSI heatmap • Occupancy grid"
    "</span>",
    unsafe_allow_html=True,
)

room = st.session_state.room_dimensions
anchors = st.session_state.anchors

# --- Tab Layout ---
tab_3d, tab_floor, tab_heat, tab_occ = st.tabs(["3D Point Cloud", "Floor Plan", "RSSI Heatmap", "Occupancy Grid"])

# ============================================================
# Tab 1: 3D Point Cloud (original content)
# ============================================================
with tab_3d:
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])

    with col_ctrl1:
        voxel_size = st.slider("Voxel Size (m)", 0.05, 0.5, 0.1, 0.05, key="map_voxel")
        st.session_state.point_cloud.voxel_size = voxel_size

    with col_ctrl2:
        if st.button("▶ Generate Path Points", use_container_width=True):
            _generate_path_points()
        if st.button("🗑 Clear Point Cloud", use_container_width=True):
            from src.mapping.point_cloud import PointCloudAccumulator
            st.session_state.point_cloud = PointCloudAccumulator(voxel_size)
            st.rerun()

    with col_ctrl3:
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("💾 Save PLY", use_container_width=True):
                pc = st.session_state.point_cloud
                if pc.num_points > 0:
                    pc.save_ply(Path(PROJECT_ROOT) / "data" / "point_cloud.ply")
                    st.toast("Point cloud saved as PLY")
        with col_exp2:
            if st.button("📂 Export CSV", use_container_width=True):
                pc = st.session_state.point_cloud
                if pc.num_points > 0:
                    np.savetxt("point_cloud.csv", pc.points, delimiter=",", header="x,y,z")
                    st.toast("Point cloud exported as CSV")

    section_header("3D Point Cloud", "🌐")

    pc = st.session_state.point_cloud

    if pc.num_points > 0:
        fig_3d = go.Figure()

        # Point cloud
        points = pc.points
        fig_3d.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker=dict(size=3, color=points[:, 2], colorscale="Viridis", opacity=0.7),
            name="Point Cloud",
        ))

        # Downsampled points
        downsampled = pc.downsample()
        if len(downsampled) > 0 and len(downsampled) < len(points):
            fig_3d.add_trace(go.Scatter3d(
                x=downsampled[:, 0], y=downsampled[:, 1], z=downsampled[:, 2],
                mode="markers",
                marker=dict(size=6, color="#ff4444", opacity=0.9),
                name="Downsampled",
            ))

        # Anchors
        for anchor in anchors:
            color = {"ceiling": "#00d4ff", "mid": "#ffaa00", "floor": "#00ff88"}.get(anchor.height, "#888")
            fig_3d.add_trace(go.Scatter3d(
                x=[anchor.position[0]], y=[anchor.position[1]], z=[anchor.position[2]],
                mode="markers+text",
                marker=dict(size=10, color=color, symbol="diamond"),
                text=[anchor.anchor_id],
                textposition="top center",
                textfont=dict(color=color, size=9),
                name=anchor.anchor_id,
            ))

        # Room bounds
        rx, ry, rz = room["length_x"], room["width_y"], room["height_z"]
        for start, end in [
            ([0,0,0],[rx,0,0]), ([0,0,0],[0,ry,0]), ([0,0,0],[0,0,rz]),
            ([rx,0,0],[rx,ry,0]), ([rx,0,0],[rx,0,rz]),
            ([0,ry,0],[rx,ry,0]), ([0,ry,0],[0,ry,rz]),
            ([0,0,rz],[rx,0,rz]), ([0,0,rz],[0,ry,rz]),
            ([rx,ry,0],[rx,ry,rz]), ([rx,0,rz],[rx,ry,rz]), ([0,ry,rz],[rx,ry,rz]),
        ]:
            fig_3d.add_trace(go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode="lines",
                line=dict(color="#2a2a4e", width=2),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig_3d.update_layout(
            scene=dict(
                xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                xaxis=dict(range=[0, rx], gridcolor="#1a1a2e"),
                yaxis=dict(range=[0, ry], gridcolor="#1a1a2e"),
                zaxis=dict(range=[0, rz], gridcolor="#1a1a2e"),
                bgcolor="#0e1117",
            ),
            paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0", family="Fira Code"),
            height=500,
            margin=dict(l=0, r=0, t=20, b=0),
        )

        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("No point cloud data yet. Generate path points or run localization to accumulate positions.")

    # Stats
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Total Points", pc.num_points)
    with col_s2:
        st.metric("Downsampled", len(pc.downsample()) if pc.num_points > 0 else 0)
    with col_s3:
        st.metric("Outlier-Filtered", len(pc.remove_outliers()) if pc.num_points > 0 else 0)
    with col_s4:
        if pc.num_points > 0:
            bounds_min = pc.points.min(axis=0)
            bounds_max = pc.points.max(axis=0)
            st.metric("Bounds", f"{bounds_max[0]-bounds_min[0]:.1f}x{bounds_max[1]-bounds_min[1]:.1f}x{bounds_max[2]-bounds_min[2]:.1f}m")
        else:
            st.metric("Bounds", "—")

# ============================================================
# Tab 2: Floor Plan
# ============================================================
with tab_floor:
    fp = FloorPlanMapper(bounds=(0, room["length_x"], 0, room["width_y"]))
    fp.add_room(0, 0, room["length_x"], room["width_y"], label="Room")

    for anchor in anchors:
        fp.add_ap(anchor.anchor_id, anchor.position[0], anchor.position[1])

    # Add position trail
    if st.session_state.position_trail_2d:
        trail = np.array(st.session_state.position_trail_2d)
        fp.set_positions(trail)
    elif st.session_state.localized_positions:
        positions = np.array([p.position for p in st.session_state.localized_positions])
        fp.set_positions(positions)

    # Optional heatmap overlay
    if st.session_state.rssi_history and len(st.session_state.position_trail_2d) > 5:
        try:
            hg = HeatmapGenerator(
                bounds=(0, room["length_x"], 0, room["width_y"]),
                resolution=0.2,
            )
            # Build per-anchor measurements from RSSI history
            rssi_by_anchor = {}
            for entry in st.session_state.rssi_history:
                aid = entry["anchor_id"]
                if aid not in rssi_by_anchor:
                    rssi_by_anchor[aid] = {"positions": [], "rssi": []}

            # Use position trail for heatmap measurements
            if st.session_state.position_trail_2d:
                trail = np.array(st.session_state.position_trail_2d)
                # Assign positions to anchors using path-loss model
                from src.utils.math_utils import path_loss_rssi
                for idx in range(len(trail)):
                    pos = trail[idx]
                    for anchor in anchors:
                        dist = np.linalg.norm(np.array([pos[0], pos[1], 0]) - anchor.position[:2])
                        rssi = path_loss_rssi(dist)
                        hg.add_measurements(
                            np.array([[pos[0], pos[1]]]),
                            np.array([rssi]),
                            bssid=anchor.anchor_id,
                        )
                fp.overlay_heatmap(hg)
                bssid_options = hg.bssids
                selected_bssid = st.selectbox("Heatmap AP", bssid_options, key="fp_heatmap_ap")
                fig = fp.to_plotly(heatmap_bssid=selected_bssid, dark_theme=True)
            else:
                fig = fp.to_plotly(dark_theme=True)
        except Exception:
            fig = fp.to_plotly(dark_theme=True)
    else:
        fig = fp.to_plotly(dark_theme=True)

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 3: RSSI Heatmap
# ============================================================
with tab_heat:
    if st.session_state.rssi_history and len(st.session_state.position_trail_2d) > 5:
        hg = HeatmapGenerator(
            bounds=(0, room["length_x"], 0, room["width_y"]),
            resolution=0.2,
        )

        from src.utils.math_utils import path_loss_rssi
        trail = np.array(st.session_state.position_trail_2d)

        for anchor in anchors:
            positions_list = []
            rssi_list = []
            for pos in trail:
                dist = np.linalg.norm(np.array([pos[0], pos[1], 0]) - anchor.position[:2])
                rssi = path_loss_rssi(dist)
                positions_list.append([pos[0], pos[1]])
                rssi_list.append(rssi)

            if positions_list:
                hg.add_measurements(
                    np.array(positions_list),
                    np.array(rssi_list),
                    bssid=anchor.anchor_id,
                )

        selected_ap = st.selectbox("Select Access Point", hg.bssids, key="heatmap_ap")
        fig_heat = hg.to_plotly(bssid=selected_ap, title=f"RSSI Heatmap — {selected_ap}")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Generate position data first by running localization. At least 5 position estimates are needed for a heatmap.")

# ============================================================
# Tab 4: Occupancy Grid
# ============================================================
with tab_occ:
    slice_height = st.slider("Z-Slice Height (m)", 0.0, float(room["height_z"]), 1.0, 0.1, key="map_zslice")

    if st.session_state.localized_positions or st.session_state.position_trail_2d:
        # Build occupancy grid from position data
        if st.session_state.get("occupancy_grid") is None:
            grid = OccupancyGrid(
                room_dimensions=room,
                voxel_size=0.1,
                occupancy_threshold=0.7,
                free_threshold=0.3,
            )
            grid.initialize()
            st.session_state.occupancy_grid = grid
        else:
            grid = st.session_state.occupancy_grid

        # Add positions to grid
        if st.session_state.position_trail_2d:
            positions = np.array(st.session_state.position_trail_2d)
            # Pad to 3D with z = slice_height
            positions_3d = np.column_stack([
                positions[:, 0],
                positions[:, 1],
                np.full(len(positions), slice_height),
            ])
            for pos in positions_3d:
                grid.update(pos)

        viz = Visualizer(room_dimensions=room)
        fig_occ = viz.plot_occupancy_slice(grid, z_slice=slice_height, method="plotly",
                                             title=f"Occupancy Grid (z = {slice_height:.1f}m)")
        if fig_occ is not None:
            st.plotly_chart(fig_occ, use_container_width=True)
        else:
            st.warning("Occupancy grid is empty — run localization first.")
    else:
        st.info("Generate position data first by running localization to populate the occupancy grid.")


def _generate_path_points():
    room = st.session_state.room_dimensions
    anchors = st.session_state.anchors

    center = np.array([room["length_x"] / 2, room["width_y"] / 2, 1.2])
    radius = min(room["length_x"], room["width_y"]) / 3

    points = []
    for i in range(30):
        angle = 2 * np.pi * i / 30
        pos = center + np.array([
            radius * np.cos(angle) + np.random.randn() * 0.1,
            radius * np.sin(angle) + np.random.randn() * 0.1,
            0.3 * np.sin(angle * 2) + np.random.randn() * 0.05,
        ])
        points.append(pos)

    st.session_state.point_cloud.add_points(np.array(points), method="simulated")
    st.toast(f"Generated {len(points)} path points")