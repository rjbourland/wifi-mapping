"""CSI phase sanitization and correction.

Corrects Carrier Frequency Offset (CFO), Sampling Time Offset (STO),
and Sampling Frequency Offset (SFO) in CSI data, and filters guard/null subcarriers.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PhaseSanitizer:
    """Cleans raw CSI phase data by removing systematic offsets.

    Raw CSI phase is corrupted by:
    1. Carrier Frequency Offset (CFO) — phase rotation proportional to subcarrier index
    2. Sampling Time Offset (STO) — constant phase shift across all subcarriers
    3. Sampling Frequency Offset (SFO) — linear phase drift across subcarriers

    After sanitization, the phase should reflect only the channel response,
    enabling accurate AoA estimation and fingerprinting.
    """

    def __init__(self, bandwidth: int = 20, standard: str = "802.11n"):
        """Initialize phase sanitizer.

        Args:
            bandwidth: Channel bandwidth in MHz (20, 40, 80, or 160).
            standard: WiFi standard ('802.11n' or '802.11ax').
        """
        self.bandwidth = bandwidth
        self.standard = standard

        # Number of total and data subcarriers by bandwidth
        self._subcarrier_config = {
            20: {"total": 64 if standard == "802.11n" else 256, "data": 52 if standard == "802.11n" else 234},
            40: {"total": 128 if standard == "802.11n" else 512, "data": 114 if standard == "802.11n" else 468},
            80: {"total": 256, "data": 242},
            160: {"total": 512, "data": 484},
        }

    def sanitize(self, csi: np.ndarray, method: str = "conjugate") -> np.ndarray:
        """Sanitize CSI phase to remove systematic offsets.

        Args:
            csi: Complex CSI matrix, shape (num_antennas, num_subcarriers).
            method: Sanitization method:
                - 'conjugate': Multiply consecutive packets' conjugate (Widar-style)
                - 'difference': Subtract consecutive subcarrier phases (SpotFi-style)
                - 'linear': Remove linear fit from phase (simple)

        Returns:
            Sanitized complex CSI matrix.
        """
        phase = np.angle(csi)

        if method == "conjugate":
            # Widar-style: multiply CSI by its conjugate across packets
            # This removes the common phase offset but preserves relative phase
            return csi * np.conj(csi[:, 0:1])  # Normalize to first subcarrier

        elif method == "difference":
            # SpotFi-style: subtract phase of subcarrier k from subcarrier k-1
            # This removes CFO and STO but introduces noise
            phase_diff = np.diff(phase, axis=-1)
            amplitude = np.abs(csi)
            result_amp = np.sqrt(amplitude[:, :-1] ** 2 + amplitude[:, 1:] ** 2)
            result_phase = phase_diff
            return result_amp * np.exp(1j * result_phase)

        elif method == "linear":
            # Simple: fit and remove linear trend from phase
            sanitized = csi.copy()
            for ant in range(csi.shape[0]):
                x = np.arange(csi.shape[1])
                coeffs = np.polyfit(x, phase[ant, :], 1)
                sanitized[ant, :] *= np.exp(-1j * np.polyval(coeffs, x))
            return sanitized

        else:
            raise ValueError(f"Unknown sanitization method: {method}")

    def filter_subcarriers(
        self, csi: np.ndarray, include_guard: bool = False, include_null: bool = False
    ) -> np.ndarray:
        """Remove guard and null subcarriers, keeping only data subcarriers.

        Args:
            csi: Complex CSI matrix, shape (num_antennas, num_subcarriers).
            include_guard: Whether to include guard subcarriers.
            include_null: Whether to include null (DC, pilot) subcarriers.

        Returns:
            Filtered CSI matrix with only data subcarriers.
        """
        config = self._subcarrier_config.get(self.bandwidth)
        if config is None:
            raise ValueError(f"Unsupported bandwidth: {self.bandwidth}")

        total = csi.shape[-1]

        if total == config["total"]:
            # Full CSI — need to filter
            data_indices = self._get_data_subcarrier_indices(total, config["data"])
            return csi[:, data_indices]
        elif total == config["data"]:
            # Already filtered
            return csi
        else:
            logger.warning(
                f"Unexpected subcarrier count {total} for {self.bandwidth} MHz. "
                f"Expected {config['total']} or {config['data']}. Returning as-is."
            )
            return csi

    def remove_cfo(self, csi: np.ndarray) -> np.ndarray:
        """Remove Carrier Frequency Offset from CSI phase.

        CFO causes a linear phase rotation proportional to subcarrier index.
        This method estimates and removes it by fitting a line to the phase.

        Args:
            csi: Complex CSI matrix, shape (num_antennas, num_subcarriers).

        Returns:
            CFO-corrected CSI matrix.
        """
        phase = np.angle(csi)
        sanitized = csi.copy()

        for ant in range(csi.shape[0]):
            # Fit linear CFO (slope = 2*pi*delta_f*Ts)
            x = np.arange(csi.shape[1])
            coeffs = np.polyfit(x, phase[ant, :], 1)
            # Remove only the linear component (CFO)
            sanitized[ant, :] *= np.exp(-1j * coeffs[0] * x)

        return sanitized

    def _get_data_subcarrier_indices(self, total: int, num_data: int) -> np.ndarray:
        """Get indices of data subcarriers (excluding guard, DC, and pilot).

        Args:
            total: Total number of subcarriers.
            num_data: Number of data subcarriers.

        Returns:
            Array of data subcarrier indices.
        """
        # Standard 802.11 subcarrier mapping
        if total == 64 and num_data == 52:
            # 802.11n 20 MHz: 52 data subcarriers out of 64
            indices = np.concatenate([
                np.arange(-26, -22),   # Lower band
                np.arange(-20, -7),
                np.arange(-5, 0),
                np.arange(1, 6),
                np.arange(8, 21),
                np.arange(23, 27),     # Upper band
            ])
            return np.arange(total)[indices >= 0]  # Simplified

        # For other configs, return all indices as fallback
        return np.arange(total)