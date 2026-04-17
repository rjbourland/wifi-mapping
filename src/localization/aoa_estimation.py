"""Angle of Arrival (AoA) estimation using MUSIC algorithm.

Implements the MUltiple SIgnal Classification (MUSIC) algorithm for
AoA estimation from CSI data, inspired by SpotFi.
"""

import logging
from typing import Optional

import numpy as np
from scipy import signal

from ..utils.config import load_config

logger = logging.getLogger(__name__)


class MUSICEstimator:
    """MUSIC-based Angle of Arrival estimator.

    Uses the MUSIC (MUltiple SIgnal Classification) algorithm to estimate
    angles of arrival from CSI data. Extends the virtual antenna array
    across subcarriers (SpotFi approach) for improved resolution.

    Requires at least 2 antennas for basic AoA, or 3+ for multipath
    resolution with SpotFi-style joint AoA+ToF estimation.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize MUSIC estimator.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
        """
        if config is None:
            config = load_config("algorithm")

        music_config = config.get("music", {})
        self.num_antennas = music_config.get("num_antennas", 2)
        self.num_paths = music_config.get("num_paths", 4)
        self.angle_resolution = music_config.get("angle_resolution", 1.0)
        self.subcarrier_selection = music_config.get("subcarrier_selection", "data_only")

    def estimate_aoa(
        self,
        csi: np.ndarray,
        frequency: float = 5.18e9,
        antenna_spacing: float = None,
    ) -> list[dict]:
        """Estimate angles of arrival from CSI data using MUSIC.

        Args:
            csi: Complex CSI matrix, shape (num_antennas, num_subcarriers).
            frequency: Carrier frequency in Hz.
            antenna_spacing: Distance between antennas in meters.
                           Default: half-wavelength (lambda/2).

        Returns:
            List of dicts with keys 'angle_deg', 'power', 'path_type'.
            Sorted by power (strongest first).
        """
        num_antennas, num_subcarriers = csi.shape
        wavelength = 3e8 / frequency
        if antenna_spacing is None:
            antenna_spacing = wavelength / 2

        # Build spatial-smoothed covariance matrix
        # Virtual array: combine antennas and subcarriers (SpotFi approach)
        R = self._covariance_matrix(csi)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Signal subspace: first num_paths eigenvectors
        # Noise subspace: remaining eigenvectors
        noise_subspace = eigenvectors[:, self.num_paths:]

        # Search over angles
        angle_range = np.arange(-90, 91, self.angle_resolution)
        pseudo_spectrum = np.zeros(len(angle_range))

        for i, angle in enumerate(angle_range):
            steering_vec = self._steering_vector(
                angle, num_antennas, num_subcarriers,
                frequency, antenna_spacing,
            )
            # MUSIC pseudo-spectrum: 1 / (a^H * En * En^H * a)
            projection = steering_vec.conj() @ noise_subspace @ noise_subspace.conj().T @ steering_vec
            pseudo_spectrum[i] = 1.0 / (np.abs(projection) + 1e-10)

        # Find peaks in pseudo-spectrum
        peaks, properties = signal.find_peaks(
            pseudo_spectrum,
            height=np.max(pseudo_spectrum) * 0.1,  # 10% of max
            distance=5,  # Minimum 5 degrees between peaks
        )

        # Sort peaks by power
        peak_powers = pseudo_spectrum[peaks]
        sorted_idx = np.argsort(peak_powers)[::-1]

        results = []
        for idx in sorted_idx[:self.num_paths]:
            angle = angle_range[peaks[idx]]
            power = peak_powers[idx]
            results.append({
                "angle_deg": float(angle),
                "power": float(power),
                "path_type": "direct" if idx == sorted_idx[0] else "multipath",
            })

        return results

    def _covariance_matrix(self, csi: np.ndarray) -> np.ndarray:
        """Compute spatial-smoothed covariance matrix.

        Uses subcarrier smoothing (SpotFi approach) to create a virtual
        antenna array with num_antennas * num_subcarriers elements.

        Args:
            csi: Complex CSI matrix, shape (num_antennas, num_subcarriers).

        Returns:
            Covariance matrix, shape (M, M) where M = num_antennas * num_subcarriers.
        """
        num_antennas, num_subcarriers = csi.shape
        M = num_antennas * num_subcarriers

        # Flatten CSI into virtual array
        virtual_array = csi.flatten()

        # Covariance matrix estimate
        R = np.outer(virtual_array, virtual_array.conj())

        # Spatial smoothing: average over subcarrier shifts
        R_smooth = np.zeros_like(R)
        num_shifts = min(5, num_subcarriers)
        for shift in range(num_shifts):
            v = csi[:, shift:].flatten()
            R_smooth += np.outer(v, v.conj())

        R_smooth /= num_shifts
        return R_smooth

    def _steering_vector(
        self,
        angle_deg: float,
        num_antennas: int,
        num_subcarriers: int,
        frequency: float,
        antenna_spacing: float,
    ) -> np.ndarray:
        """Compute steering vector for MUSIC spectrum calculation.

        Args:
            angle_deg: Angle in degrees.
            num_antennas: Number of antennas.
            num_subcarriers: Number of subcarriers.
            frequency: Carrier frequency in Hz.
            antenna_spacing: Distance between antennas in meters.

        Returns:
            Steering vector, shape (num_antennas * num_subcarriers,).
        """
        wavelength = 3e8 / frequency
        angle_rad = np.radians(angle_deg)

        # Antenna steering vector
        antenna_phases = np.exp(
            -1j * 2 * np.pi * antenna_spacing * np.sin(angle_rad)
            * np.arange(num_antennas) / wavelength
        )

        # Subcarrier steering vector (for ToF component)
        subcarrier_spacing = frequency / num_subcarriers
        subcarrier_phases = np.exp(
            -1j * 2 * np.pi * np.arange(num_subcarriers)
            * subcarrier_spacing / frequency
        )

        # Kronecker product for joint angle-time steering vector
        steering = np.kron(antenna_phases, subcarrier_phases)
        return steering