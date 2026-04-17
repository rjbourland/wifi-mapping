"""Tests for src.mapping: adapters, heatmap, floor_plan, visualization."""

import numpy as np
import pytest
from datetime import datetime

from src.localization.trilateration import Position
from src.localization.kalman_filter import SmoothedPosition
from src.utils.data_formats import AnchorPosition, LocalizedPosition
from src.mapping.adapters import to_xyz, positions_to_array
from src.mapping.heatmap import HeatmapGenerator
from src.mapping.floor_plan import FloorPlanMapper


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class TestToXyz:
    def test_ndarray_3d(self):
        result = to_xyz(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_ndarray_2d_uses_default_z(self):
        result = to_xyz(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 0.0])

    def test_ndarray_2d_custom_z(self):
        result = to_xyz(np.array([1.0, 2.0]), z=1.5)
        np.testing.assert_array_equal(result, [1.0, 2.0, 1.5])

    def test_position_dataclass(self):
        pos = Position(x=3.0, y=4.0, estimated_error_meters=0.5,
                       timestamp=datetime.now(), ap_count=3)
        result = to_xyz(pos)
        np.testing.assert_array_equal(result, [3.0, 4.0, 0.0])

    def test_position_dataclass_custom_z(self):
        pos = Position(x=3.0, y=4.0, estimated_error_meters=0.5,
                       timestamp=datetime.now(), ap_count=3)
        result = to_xyz(pos, z=2.0)
        np.testing.assert_array_equal(result, [3.0, 4.0, 2.0])

    def test_smoothed_position_dataclass(self):
        sp = SmoothedPosition(x=1.0, y=2.0, velocity_x=0.1,
                               velocity_y=0.2, confidence=0.9)
        result = to_xyz(sp)
        np.testing.assert_array_equal(result, [1.0, 2.0, 0.0])

    def test_localized_position(self):
        lp = LocalizedPosition(
            timestamp=datetime.now(),
            position=np.array([5.0, 6.0, 7.0]),
            method="trilateration",
            anchors_used=["A1", "A2", "A3"],
        )
        result = to_xyz(lp)
        np.testing.assert_array_equal(result, [5.0, 6.0, 7.0])

    def test_unrecognized_type_raises(self):
        with pytest.raises(TypeError, match="Cannot promote"):
            to_xyz("not_a_position")


class TestPositionsToArray:
    def test_mixed_types(self):
        pos = Position(x=1.0, y=2.0, estimated_error_meters=0.5,
                       timestamp=datetime.now(), ap_count=3)
        arr = positions_to_array([pos, np.array([3.0, 4.0])], z=0.5)
        assert arr.shape == (2, 3)
        np.testing.assert_array_almost_equal(arr[0], [1.0, 2.0, 0.5])
        np.testing.assert_array_almost_equal(arr[1], [3.0, 4.0, 0.5])

    def test_empty_list(self):
        arr = positions_to_array([])
        assert arr.shape == (0, 3)


# ---------------------------------------------------------------------------
# HeatmapGenerator
# ---------------------------------------------------------------------------

class TestHeatmapGenerator:
    def setup_method(self):
        self.hg = HeatmapGenerator(bounds=(0, 10, 0, 10), resolution=0.5)
        rng = np.random.default_rng(0)
        n = 20
        self.positions = rng.uniform(1, 9, size=(n, 2))
        self.rssi = np.array([-50 - 3 * np.sqrt((x - 5) ** 2 + (y - 5) ** 2)
                               for x, y in self.positions])
        self.hg.add_measurements(self.positions, self.rssi, bssid="AP1")

    def test_add_measurements_single_bssid(self):
        assert "AP1" in self.hg.bssids
        assert len(self.hg._data["AP1"]["rssi"]) == 20

    def test_add_measurements_accumulates(self):
        more_pos = np.array([[2.0, 2.0], [8.0, 8.0]])
        more_rssi = np.array([-55.0, -60.0])
        self.hg.add_measurements(more_pos, more_rssi, bssid="AP1")
        assert len(self.hg._data["AP1"]["rssi"]) == 22

    def test_add_measurements_3d_truncation(self):
        pos_3d = np.column_stack([self.positions, np.zeros(20)])
        hg2 = HeatmapGenerator(bounds=(0, 10, 0, 10))
        hg2.add_measurements(pos_3d, self.rssi, bssid="AP3d")
        assert hg2._data["AP3d"]["positions"].shape[1] == 2

    def test_interpolate(self):
        X, Y, Z = self.hg.interpolate("AP1")
        assert X.shape == Z.shape
        assert Y.shape == Z.shape
        assert not np.all(np.isnan(Z))

    def test_interpolate_unknown_bssid_raises(self):
        with pytest.raises(ValueError, match="No measurements"):
            self.hg.interpolate("UNKNOWN")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            self.hg.add_measurements(np.zeros((5, 2)), np.zeros(3), bssid="bad")

    def test_to_matplotlib_returns_figure(self):
        fig = self.hg.to_matplotlib("AP1")
        assert fig is not None

    def test_to_plotly_returns_figure(self):
        fig = self.hg.to_plotly("AP1")
        assert fig is not None

    def test_to_matplotlib_saves_file(self, tmp_path):
        filepath = tmp_path / "heatmap.png"
        self.hg.to_matplotlib("AP1", filepath=filepath)
        assert filepath.exists()

    def test_to_plotly_saves_file(self, tmp_path):
        filepath = tmp_path / "heatmap.html"
        self.hg.to_plotly("AP1", filepath=filepath)
        assert filepath.exists()


# ---------------------------------------------------------------------------
# FloorPlanMapper
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless testing


class TestFloorPlanMapper:
    def setup_method(self):
        self.fp = FloorPlanMapper(bounds=(0, 10, 0, 10))

    def test_add_wall(self):
        self.fp.add_wall(0, 0, 10, 0)
        assert len(self.fp._walls) == 1

    def test_add_room_creates_four_walls(self):
        self.fp.add_room(0, 0, 10, 10, label="Room")
        assert len(self.fp._walls) == 4

    def test_add_walls_batch(self):
        walls = [(0, 0, 5, 0), (5, 0, 5, 5)]
        self.fp.add_walls(walls)
        assert len(self.fp._walls) == 2

    def test_add_ap(self):
        self.fp.add_ap("AP1", 2.0, 3.0)
        assert len(self.fp._aps) == 1
        assert self.fp._aps[0]["id"] == "AP1"

    def test_add_anchors(self):
        anchors = [
            AnchorPosition("A1", np.array([1.0, 2.0, 0.5]), "mid", "esp32_s3", "192.168.1.1", 6, 20),
            AnchorPosition("A2", np.array([8.0, 3.0, 0.5]), "mid", "esp32_s3", "192.168.1.2", 1, 20),
        ]
        self.fp.add_anchors(anchors)
        assert len(self.fp._aps) == 2

    def test_set_positions_2d(self):
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.fp.set_positions(positions)
        assert self.fp._positions.shape == (2, 2)

    def test_set_positions_3d_truncated(self):
        positions = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 0.5]])
        self.fp.set_positions(positions)
        assert self.fp._positions.shape == (2, 2)

    def test_to_matplotlib_returns_figure(self):
        self.fp.add_room(0, 0, 10, 10)
        self.fp.add_ap("AP1", 5, 5)
        self.fp.set_positions(np.array([[2, 3], [4, 5], [6, 7]]))
        fig = self.fp.to_matplotlib()
        assert fig is not None

    def test_to_plotly_returns_figure(self):
        self.fp.add_room(0, 0, 10, 10)
        self.fp.add_ap("AP1", 5, 5)
        fig = self.fp.to_plotly()
        assert fig is not None

    def test_overlay_heatmap_and_render(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(1, 9, size=(20, 2))
        rssi = np.array([-50 - 3 * np.sqrt((x - 5) ** 2 + (y - 5) ** 2)
                          for x, y in positions])
        hg = HeatmapGenerator(bounds=(0, 10, 0, 10), resolution=0.5)
        hg.add_measurements(positions, rssi, bssid="AP1")

        self.fp.add_room(0, 0, 10, 10)
        self.fp.add_ap("AP1", 5, 5)
        self.fp.set_positions(positions)
        self.fp.overlay_heatmap(hg)

        fig = self.fp.to_matplotlib(heatmap_bssid="AP1")
        assert fig is not None

        fig_plotly = self.fp.to_plotly(heatmap_bssid="AP1")
        assert fig_plotly is not None

    def test_to_matplotlib_saves_file(self, tmp_path):
        self.fp.add_room(0, 0, 10, 10)
        filepath = tmp_path / "floorplan.png"
        fig = self.fp.to_matplotlib(filepath=filepath)
        assert filepath.exists()

    def test_to_plotly_saves_file(self, tmp_path):
        self.fp.add_room(0, 0, 10, 10)
        filepath = tmp_path / "floorplan.html"
        fig = self.fp.to_plotly(filepath=filepath)
        assert filepath.exists()

    def test_dark_theme_plotly(self):
        self.fp.add_room(0, 0, 10, 10)
        fig = self.fp.to_plotly(dark_theme=True)
        assert fig.layout.plot_bgcolor == "#0e1117"