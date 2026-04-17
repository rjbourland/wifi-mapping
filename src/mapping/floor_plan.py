"""Floor plan mapping: walls, AP markers, position trails, heatmap overlay.

Renders 2D floor plans with compositional layers — walls, access points,
position trails, and optional RSSI heatmap — in both matplotlib and Plotly.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FloorPlanMapper:
    """2D floor plan renderer with walls, AP markers, position trails,
    and optional RSSI heatmap overlay.

    Usage::

        fp = FloorPlanMapper(bounds=(0, 10, 0, 10))
        fp.add_room(0, 0, 10, 10, label="Demo Room")
        fp.add_ap("AP1", 0, 0)
        fp.set_positions(positions_xy)
        fp.to_matplotlib(filepath="output/floorplan.png")
        fp.to_plotly(filepath="output/floorplan.html")
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
        wall_color: str = "#2a2a4e",
        ap_color_map: Optional[dict[str, str]] = None,
    ):
        self.bounds = bounds
        self.wall_color = wall_color
        self.ap_color_map = ap_color_map or {}
        self._walls: list[tuple[float, float, float, float, str]] = []
        self._aps: list[dict] = []
        self._positions: Optional[np.ndarray] = None
        self._heatmap_generator = None

    def add_wall(
        self, x1: float, y1: float, x2: float, y2: float, label: str = "",
    ):
        """Add a wall segment from (x1, y1) to (x2, y2)."""
        self._walls.append((x1, y1, x2, y2, label))

    def add_walls(self, walls: list[tuple[float, float, float, float]]):
        """Add multiple wall segments at once."""
        for w in walls:
            self.add_wall(*w)

    def add_room(
        self, x_min: float, y_min: float, x_max: float, y_max: float, label: str = "",
    ):
        """Add four walls forming a rectangular room."""
        self.add_wall(x_min, y_min, x_max, y_min, label)
        self.add_wall(x_max, y_min, x_max, y_max, label)
        self.add_wall(x_max, y_max, x_min, y_max, label)
        self.add_wall(x_min, y_max, x_min, y_min, label)

    def add_ap(self, ap_id: str, x: float, y: float, color: Optional[str] = None):
        """Add an access point marker."""
        self._aps.append({"id": ap_id, "x": x, "y": y, "color": color})

    def add_anchors(self, anchors: list):
        """Add anchors from AnchorPosition objects."""
        for a in anchors:
            self.add_ap(a.anchor_id, a.position[0], a.position[1])

    def set_positions(self, positions: np.ndarray):
        """Set the position trail. Accepts (N, 2) or (N, 3) arrays."""
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        self._positions = positions[:, :2]

    def overlay_heatmap(self, heatmap_generator):
        """Overlay an RSSI heatmap from a HeatmapGenerator."""
        self._heatmap_generator = heatmap_generator

    def to_matplotlib(
        self,
        filepath: Optional[Path | str] = None,
        heatmap_bssid: Optional[str] = None,
        title: str = "Floor Plan",
        figsize: tuple[int, int] = (12, 10),
    ):
        """Render floor plan as matplotlib figure."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        # 1) Heatmap layer
        if self._heatmap_generator is not None and heatmap_bssid is not None:
            X, Y, Z = self._heatmap_generator.interpolate(heatmap_bssid)
            im = ax.pcolormesh(X, Y, Z, cmap="RdYlGn", shading="auto", alpha=0.6, zorder=1)
            plt.colorbar(im, ax=ax, label="RSSI (dBm)")

        # 2) Walls
        for x1, y1, x2, y2, label in self._walls:
            ax.plot([x1, x2], [y1, y2], color=self.wall_color, linewidth=2.5, zorder=3)

        # 3) AP markers
        for ap in self._aps:
            color = ap["color"] or self.ap_color_map.get(ap["id"], "#00d4ff")
            ax.scatter(ap["x"], ap["y"], c=color, s=200, marker="^", zorder=5)
            ax.annotate(ap["id"], (ap["x"], ap["y"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=color, zorder=5)

        # 4) Position trail
        if self._positions is not None and len(self._positions) > 0:
            pos = self._positions
            ax.plot(pos[:, 0], pos[:, 1], ".-", color="#6c63ff", alpha=0.6,
                    linewidth=1.5, markersize=4, label="Position trail", zorder=4)
            ax.scatter(pos[0, 0], pos[0, 1], c="green", s=80, marker="o", zorder=6, label="Start")
            ax.scatter(pos[-1, 0], pos[-1, 1], c="red", s=80, marker="x",
                        linewidths=2, zorder=6, label="End")

        ax.set_xlim(self.bounds[0] - 0.5, self.bounds[1] + 0.5)
        ax.set_ylim(self.bounds[2] - 0.5, self.bounds[3] + 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        if self._positions is not None and len(self._positions) > 0:
            ax.legend(loc="upper right", fontsize=8)

        if filepath is not None:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info("Saved floor plan to %s", filepath)

        return fig

    def to_plotly(
        self,
        filepath: Optional[Path | str] = None,
        heatmap_bssid: Optional[str] = None,
        title: str = "Floor Plan",
        dark_theme: bool = True,
    ):
        """Render floor plan as a Plotly figure."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # 1) Heatmap layer
        if self._heatmap_generator is not None and heatmap_bssid is not None:
            X, Y, Z = self._heatmap_generator.interpolate(heatmap_bssid)
            fig.add_trace(go.Heatmap(
                x=X[0, :], y=Y[:, 0], z=Z,
                colorscale="RdYlGn", opacity=0.6,
                showscale=True, colorbar=dict(title="RSSI (dBm)"),
                hovertemplate="X: %{x:.1f}m  Y: %{y:.1f}m<br>RSSI: %{z:.1f} dBm<extra></extra>",
            ))

        # 2) Walls
        wall_color = "#2a2a4e" if dark_theme else self.wall_color
        for x1, y1, x2, y2, label in self._walls:
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2],
                mode="lines", line=dict(color=wall_color, width=3),
                showlegend=False, hoverinfo="skip",
            ))

        # 3) AP markers
        for ap in self._aps:
            color = ap["color"] or self.ap_color_map.get(ap["id"], "#00d4ff")
            fig.add_trace(go.Scatter(
                x=[ap["x"]], y=[ap["y"]],
                mode="markers+text",
                marker=dict(size=14, color=color, symbol="triangle-up"),
                text=[ap["id"]], textposition="top center",
                textfont=dict(color=color, size=10),
                name=ap["id"],
            ))

        # 4) Position trail
        if self._positions is not None and len(self._positions) > 0:
            pos = self._positions
            fig.add_trace(go.Scatter(
                x=pos[:, 0], y=pos[:, 1],
                mode="lines+markers",
                line=dict(color="#6c63ff", width=2),
                marker=dict(size=4, color="#6c63ff"),
                name="Position trail",
            ))

        layout_kwargs = dict(
            title=title,
            xaxis_title="X (m)", yaxis_title="Y (m)",
            yaxis_scaleanchor="x", yaxis_scaleratio=1,
            xaxis=dict(range=[self.bounds[0] - 0.5, self.bounds[1] + 0.5]),
            yaxis=dict(range=[self.bounds[2] - 0.5, self.bounds[3] + 0.5]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        if dark_theme:
            layout_kwargs.update(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#e0e0e0", family="Fira Code"),
                xaxis_gridcolor="#1a1a2e", yaxis_gridcolor="#1a1a2e",
            )

        fig.update_layout(**layout_kwargs)

        if filepath is not None:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(filepath))
            logger.info("Saved floor plan HTML to %s", filepath)

        return fig