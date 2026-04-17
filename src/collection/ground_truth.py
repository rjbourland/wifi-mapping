"""Ground-truth position logging for calibration."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils.data_formats import GroundTruthPoint

LOG_DIR = Path("./data/ground_truth")


class GroundTruthLogger:
    """Logs ground-truth positions for calibration and accuracy evaluation.

    Records the true (x, y, z) position alongside metadata like LoS/NLoS
    conditions from each anchor. Used for fingerprinting training data
    and localization accuracy evaluation.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or LOG_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._points: list[GroundTruthPoint] = []

    def log_position(
        self,
        position: np.ndarray,
        label: str = "",
        los_from: Optional[list[str]] = None,
        nlos_from: Optional[list[str]] = None,
    ) -> GroundTruthPoint:
        """Record a ground-truth position measurement.

        Args:
            position: (x, y, z) coordinates in meters.
            label: Optional descriptive label (e.g., "desk_center").
            los_from: List of anchor IDs with line-of-sight.
            nlos_from: List of anchor IDs without line-of-sight.

        Returns:
            GroundTruthPoint object.
        """
        point = GroundTruthPoint(
            timestamp=datetime.now(),
            position=np.asarray(position, dtype=float),
            label=label,
            los_from=los_from or [],
            nlos_from=nlos_from or [],
        )
        self._points.append(point)
        return point

    def save_csv(self, filename: str = "ground_truth.csv"):
        """Save all logged points to a CSV file.

        Args:
            filename: Output filename (saved in output_dir).
        """
        filepath = self.output_dir / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "x", "y", "z", "label",
                "los_anchors", "nlos_anchors",
            ])
            for p in self._points:
                writer.writerow([
                    p.timestamp.isoformat(),
                    f"{p.position[0]:.4f}",
                    f"{p.position[1]:.4f}",
                    f"{p.position[2]:.4f}",
                    p.label,
                    ";".join(p.los_from),
                    ";".join(p.nlos_from),
                ])

    @classmethod
    def load_csv(cls, filepath: Path) -> list[GroundTruthPoint]:
        """Load ground-truth points from a CSV file.

        Args:
            filepath: Path to the CSV file.

        Returns:
            List of GroundTruthPoint objects.
        """
        points = []
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append(GroundTruthPoint(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    position=np.array([
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    ]),
                    label=row.get("label", ""),
                    los_from=row.get("los_anchors", "").split(";") if row.get("los_anchors") else [],
                    nlos_from=row.get("nlos_anchors", "").split(";") if row.get("nlos_anchors") else [],
                ))
        return points