"""Breathing rate detection from CSI periodicity."""

import logging
from typing import Optional

import numpy as np
from scipy import signal, fft

from ..utils.config import load_config

logger = logging.getLogger(__name__)


class BreathingDetector:
    """Detects breathing rate from periodic CSI amplitude variations.

    When a person breathes, their chest movement creates periodic
    variations in the WiFi channel. This detector extracts the
    dominant frequency in the 10-30 BPM range from CSI amplitude
    time series to estimate breathing rate.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize breathing detector.

        Args:
            config: Algorithm configuration dict (from algorithm.yaml).
        """
        if config is None:
            config = load_config("algorithm")

        br_config = config.get("breathing_detection", {})
        self.bpm_min = br_config.get("bpm_min", 10)
        self.bpm_max = br_config.get("bpm_max", 30)
        self.fft_window = br_config.get("fft_window", 1024)
        self.snr_threshold = br_config.get("snr_threshold", 3.0)

    def detect(
        self, csi_packets: np.ndarray, sample_rate: float = 50.0
    ) -> dict:
        """Detect breathing rate from CSI packets.

        Args:
            csi_packets: Complex CSI packets, shape (num_packets, num_antennas, num_subcarriers).
            sample_rate: CSI sampling rate in Hz.

        Returns:
            Dict with keys:
            - 'breathing_detected': Boolean
            - 'breathing_rate_bpm': Estimated breathing rate in breaths per minute
            - 'confidence': Detection confidence (0-1)
            - 'dominant_frequency_hz': Dominant frequency in Hz
            - 'snr': Signal-to-noise ratio of the detected peak
        """
        amplitude = np.abs(csi_packets)

        # Average across antennas and subcarriers
        amplitude_avg = np.mean(amplitude, axis=(1, 2))

        # Remove DC component
        amplitude_avg = amplitude_avg - np.mean(amplitude_avg)

        # Bandpass filter in the breathing frequency range
        nyquist = sample_rate / 2
        low_freq = self.bpm_min / 60.0  # Convert BPM to Hz
        high_freq = self.bpm_max / 60.0

        if low_freq < nyquist and high_freq < nyquist:
            sos = signal.butter(4, [low_freq, high_freq], btype="band", fs=sample_rate, output="sos")
            amplitude_filtered = signal.sosfiltfilt(sos, amplitude_avg)
        else:
            amplitude_filtered = amplitude_avg

        # Compute FFT
        n = len(amplitude_filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        spectrum = np.abs(np.fft.rfft(amplitude_filtered))

        # Find peaks in the breathing frequency range
        freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
        breathing_spectrum = spectrum[freq_mask]
        breathing_freqs = freqs[freq_mask]

        if len(breathing_spectrum) == 0:
            return {
                "breathing_detected": False,
                "breathing_rate_bpm": 0.0,
                "confidence": 0.0,
                "dominant_frequency_hz": 0.0,
                "snr": 0.0,
            }

        # Find dominant peak
        peak_idx = np.argmax(breathing_spectrum)
        dominant_freq = breathing_freqs[peak_idx]
        dominant_power = breathing_spectrum[peak_idx]

        # Compute SNR (peak power / mean noise power)
        noise_mask = ~freq_mask
        noise_power = np.mean(spectrum[noise_mask]) if np.any(noise_mask) else 1.0
        snr = dominant_power / (noise_power + 1e-10)

        breathing_detected = snr > self.snr_threshold
        breathing_rate_bpm = dominant_freq * 60.0  # Hz to BPM

        # Confidence based on SNR
        confidence = min(snr / (self.snr_threshold * 3), 1.0)

        return {
            "breathing_detected": breathing_detected,
            "breathing_rate_bpm": float(breathing_rate_bpm),
            "confidence": float(confidence),
            "dominant_frequency_hz": float(dominant_freq),
            "snr": float(snr),
        }