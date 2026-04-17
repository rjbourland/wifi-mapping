"""CLI demo: generates synthetic RSSI data and produces heatmap + floor plan outputs.

Usage::

    python -m src.mapping.demo
"""

import logging
import sys
from pathlib import Path

import numpy as np

from .heatmap import HeatmapGenerator
from .floor_plan import FloorPlanMapper
from ..utils.math_utils import path_loss_rssi

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def run_demo():
    """Simulate a 10×10 m room with 3 APs and produce heatmap + floor plan files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # --- Room and AP configuration ---
    room_bounds = (0.0, 10.0, 0.0, 10.0)
    aps = [
        {"id": "AP1", "x": 1.0, "y": 1.0},
        {"id": "AP2", "x": 9.0, "y": 1.0},
        {"id": "AP3", "x": 5.0, "y": 9.0},
    ]

    # --- Circular path around center ---
    n_points = 40
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cx, cy = 5.0, 5.0
    radius = 3.0
    positions = np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])

    # --- Generate synthetic RSSI measurements ---
    hg = HeatmapGenerator(bounds=room_bounds, resolution=0.2, method="linear")
    for ap in aps:
        distances = np.sqrt((positions[:, 0] - ap["x"]) ** 2 + (positions[:, 1] - ap["y"]) ** 2)
        rssi = np.array([path_loss_rssi(d, n=2.5) for d in distances])
        rssi += rng.normal(0, 2.0, size=len(rssi))  # add noise
        hg.add_measurements(positions, rssi, bssid=ap["id"])

    # --- Heatmap outputs ---
    for ap in aps:
        bssid = ap["id"]
        hg.to_matplotlib(bssid, filepath=OUTPUT_DIR / f"demo_heatmap_{bssid}.png")
        hg.to_plotly(bssid, filepath=OUTPUT_DIR / f"demo_heatmap_{bssid}.html")

    # --- Floor plan outputs ---
    fp = FloorPlanMapper(bounds=room_bounds)
    fp.add_room(0, 0, 10, 10, label="Demo Room")
    for ap in aps:
        fp.add_ap(ap["id"], ap["x"], ap["y"])
    fp.set_positions(positions)
    fp.overlay_heatmap(hg)

    fp.to_matplotlib(filepath=OUTPUT_DIR / "demo_floorplan.png", heatmap_bssid="AP1")
    fp.to_plotly(filepath=OUTPUT_DIR / "demo_floorplan.html", heatmap_bssid="AP1")

    print(f"Demo outputs written to {OUTPUT_DIR}/")
    print(f"  Heatmaps: demo_heatmap_AP*.png / .html")
    print(f"  Floor plans: demo_floorplan.png / .html")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_demo()