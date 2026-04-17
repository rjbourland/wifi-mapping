"""RSSI signal strength heatmap generation over a floor area.

Uses scipy interpolation to produce smooth signal strength surfaces
from sparse measurements, with both matplotlib and Plotly export.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


class HeatmapGenerator:
    """Generate interpolated RSSI signal strength heatmaps.

    Given measured (x, y) positions and their RSSI values for a specific
    access point, uses scipy interpolation to produce a smooth signal
    strength surface over the floor area.

    Usage::

        hg = HeatmapGenerator(bounds=(0, 10, 0, 10))
        hg.add_measurements(positions_xy, rssi_values, bssid="AP1")
        fig_png = hg.to_matplotlib("AP1", filepath="output/heatmap.png")
        fig_html = hg.to_plotly("AP1", filepath="output/heatmap.html")
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
        resolution: float = 0.2,
        method: str = "linear",
    ):
        """Initialize heatmap generator.

        Args:
            bounds: (x_min, x_max, y_min, y_max) floor area in meters.
            resolution: Grid cell size in meters.
            method: Interpolation method for scipy.griddata.
                    'linear', 'nearest', or 'cubic'.
        """
        self.bounds = bounds
        self.resolution = resolution
        self.method = method
        self._data: dict[str, dict[str, np.ndarray]] = {}

    def add_measurements(
        self,
        positions: np.ndarray,
        rssi_values: np.ndarray,
        bssid: str = "default",
    ):
        """Add RSSI measurements for an access point.

        Args:
            positions: (N, 2) or (N, 3) array of measurement positions.
                        If (N, 3), only the first two columns are used.
            rssi_values: (N,) array of RSSI in dBm.
            bssid: BSSID or label for this AP.
        """
        positions = np.atleast_2d(positions)
        if positions.shape[1] > 2:
            positions = positions[:, :2]

        rssi_values = np.atleast_1d(rssi_values).astype(float)
        if len(positions) != len(rssi_values):
            raise ValueError(
                f"positions ({len(positions)}) and rssi_values ({len(rssi_values)}) must match"
            )

        if bssid not in self._data:
            self._data[bssid] = {"positions": positions, "rssi": rssi_values}
        else:
            existing = self._data[bssid]
            self._data[bssid] = {
                "positions": np.vstack([existing["positions"], positions]),
                "rssi": np.concatenate([existing["rssi"], rssi_values]),
            }

        logger.info("Added %d measurements for %s (total: %d)",
                     len(positions), bssid, len(self._data[bssid]["rssi"]))

    @property
    def bssids(self) -> list[str]:
        """List of BSSIDs with recorded measurements."""
        return list(self._data.keys())

    def interpolate(
        self,
        bssid: str = "default",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate RSSI over the floor grid for a specific AP.

        Args:
            bssid: Which AP to interpolate.

        Returns:
            Tuple of (X, Y, Z) 2D arrays. Z is interpolated RSSI in dBm;
            NaN for unconstrained regions (outside the convex hull).

        Raises:
            ValueError: If no measurements exist for bssid.
        """
        if bssid not in self._data:
            raise ValueError(
                f"No measurements for BSSID '{bssid}'. Available: {self.bssids}"
            )

        x_min, x_max, y_min, y_max = self.bounds
        data = self._data[bssid]

        xi = np.arange(x_min, x_max + self.resolution, self.resolution)
        yi = np.arange(y_min, y_max + self.resolution, self.resolution)
        X, Y = np.meshgrid(xi, yi)

        points_flat = np.column_stack([X.ravel(), Y.ravel()])
        Z_flat = griddata(data["positions"], data["rssi"], points_flat, method=self.method)
        Z = Z_flat.reshape(X.shape)

        return X, Y, Z

    def to_matplotlib(
        self,
        bssid: str = "default",
        filepath: Optional[Path | str] = None,
        cmap: str = "RdYlGn",
        title: Optional[str] = None,
    ):
        """Export heatmap as a matplotlib figure.

        Args:
            bssid: AP to visualize.
            filepath: If provided, saves the figure to this path.
            cmap: Matplotlib colormap name.
            title: Plot title.

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        X, Y, Z = self.interpolate(bssid)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, label="RSSI (dBm)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title or f"RSSI Heatmap - {bssid}")
        ax.set_aspect("equal")

        if filepath is not None:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info("Saved heatmap PNG to %s", filepath)

        return fig

    def to_plotly(
        self,
        bssid: str = "default",
        filepath: Optional[Path | str] = None,
        colorscale: str = "RdYlGn",
        title: Optional[str] = None,
    ):
        """Export heatmap as a Plotly HTML figure.

        Args:
            bssid: AP to visualize.
            filepath: If provided, saves as HTML.
            colorscale: Plotly colorscale name.
            title: Plot title.

        Returns:
            plotly Figure object.
        """
        import plotly.graph_objects as go

        X, Y, Z = self.interpolate(bssid)

        z_min = float(np.nanmin(Z)) if not np.all(np.isnan(Z)) else -90
        z_max = float(np.nanmax(Z)) if not np.all(np.isnan(Z)) else -30

        fig = go.Figure(go.Heatmap(
            x=X[0, :],
            y=Y[:, 0],
            z=Z,
            colorscale=colorscale,
            colorbar=dict(title="RSSI (dBm)"),
            zmin=z_min, zmax=z_max,
            hovertemplate="X: %{x:.1f}m  Y: %{y:.1f}m<br>RSSI: %{z:.1f} dBm<extra></extra>",
        ))

        fig.update_layout(
            title=title or f"RSSI Heatmap - {bssid}",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
        )

        if filepath is not None:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(filepath))
            logger.info("Saved heatmap HTML to %s", filepath)

        return fig