"""k-NN fingerprinting-based indoor localization."""

import logging
from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..utils.config import load_config
from ..utils.data_formats import LocalizedPosition, RSSISample

logger = logging.getLogger(__name__)


class KNNFingerprinting:
    """k-Nearest Neighbors fingerprinting for indoor localization.

    Builds a radio map (fingerprint database) from training data,
    then uses k-NN to estimate position from new RSSI/CSI measurements.

    Training data consists of (RSSI_vector, position) pairs collected
    at known reference points throughout the environment.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize k-NN fingerprinting.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
        """
        if config is None:
            config = load_config("algorithm")

        fp_config = config.get("fingerprinting", {})
        self.k = fp_config.get("k", 5)
        self.distance_metric = fp_config.get("distance_metric", "euclidean")
        self.feature_type = fp_config.get("feature_type", "rssi")
        self.weighted = fp_config.get("weighted", True)

        self._fingerprint_db: Optional[np.ndarray] = None  # (N, num_anchors) RSSI matrix
        self._positions: Optional[np.ndarray] = None  # (N, 3) positions
        self._anchor_ids: list[str] = []
        self._knn: Optional[KNeighborsRegressor] = None
        self._is_trained = False

    def add_fingerprint(
        self, rssi_vector: dict[str, float], position: np.ndarray
    ):
        """Add a single fingerprint to the database.

        Args:
            rssi_vector: Dict mapping anchor_id -> RSSI (dBm).
            position: Ground-truth (x, y, z) position in meters.
        """
        if self._fingerprint_db is None:
            self._anchor_ids = sorted(rssi_vector.keys())
            self._fingerprint_db = np.zeros((0, len(self._anchor_ids)))
            self._positions = np.zeros((0, 3))

        rssi_row = np.array([rssi_vector.get(aid, -100.0) for aid in self._anchor_ids])
        self._fingerprint_db = np.vstack([self._fingerprint_db, rssi_row])
        self._positions = np.vstack([self._positions, position])
        self._is_trained = False

    def train(self):
        """Train the k-NN model on all collected fingerprints.

        Must be called after adding fingerprints and before localizing.
        """
        if self._fingerprint_db is None or len(self._fingerprint_db) == 0:
            raise RuntimeError("No fingerprints in database. Call add_fingerprint() first.")

        weights = "distance" if self.weighted else "uniform"
        self._knn = KNeighborsRegressor(
            n_neighbors=min(self.k, len(self._fingerprint_db)),
            metric=self.distance_metric,
            weights=weights,
        )
        self._knn.fit(self._fingerprint_db, self._positions)
        self._is_trained = True
        logger.info(f"Trained k-NN fingerprinting with {len(self._fingerprint_db)} fingerprints")

    def localize(self, rssi_vector: dict[str, float]) -> LocalizedPosition:
        """Estimate position from RSSI measurements using k-NN fingerprinting.

        Args:
            rssi_vector: Dict mapping anchor_id -> RSSI (dBm).

        Returns:
            LocalizedPosition with estimated (x, y, z) coordinates.
        """
        if not self._is_trained:
            self.train()

        rssi_row = np.array([rssi_vector.get(aid, -100.0) for aid in self._anchor_ids])
        rssi_row = rssi_row.reshape(1, -1)

        position = self._knn.predict(rssi_row)[0]
        distances, _ = self._knn.kneighbors(rssi_row)

        # Confidence based on distance to nearest neighbors
        avg_dist = np.mean(distances[0])
        confidence = 1.0 / (1.0 + avg_dist / 10.0)  # Scaled confidence

        return LocalizedPosition(
            timestamp=__import__("datetime").datetime.now(),
            position=position,
            method="fingerprinting",
            confidence=min(confidence, 1.0),
            anchors_used=list(rssi_vector.keys()),
        )

    def save_database(self, filepath: str = "fingerprint_db.npz"):
        """Save the fingerprint database to disk.

        Args:
            filepath: Path to save the database.
        """
        if self._fingerprint_db is None:
            raise RuntimeError("No fingerprint database to save.")

        np.savez(
            filepath,
            fingerprints=self._fingerprint_db,
            positions=self._positions,
            anchor_ids=np.array(self._anchor_ids),
        )
        logger.info(f"Saved fingerprint database to {filepath}")

    def load_database(self, filepath: str = "fingerprint_db.npz"):
        """Load a fingerprint database from disk.

        Args:
            filepath: Path to the saved database.
        """
        data = np.load(filepath, allow_pickle=True)
        self._fingerprint_db = data["fingerprints"]
        self._positions = data["positions"]
        self._anchor_ids = list(data["anchor_ids"])
        self._is_trained = False
        logger.info(f"Loaded {len(self._fingerprint_db)} fingerprints from {filepath}")