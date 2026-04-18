"""Tests for GaitClassifier.classify() and add_template()."""

import numpy as np
import pytest

from src.detection.gait_classifier import GaitClassifier


@pytest.fixture
def classifier():
    return GaitClassifier(config=None)


def _make_gait_csi(
    num_packets=200, sample_rate=50.0, step_freq=1.8, amplitude=0.1, noise_std=0.02
):
    """Generate synthetic CSI with a periodic Doppler pattern and noise.

    Args:
        num_packets: Number of CSI packets.
        sample_rate: Sampling rate in Hz.
        step_freq: Dominant step frequency (Hz).
        amplitude: Phase modulation amplitude.
        noise_std: Per-subcarrier noise std for Doppler variance.
    """
    rng = np.random.default_rng(42)
    t = np.arange(num_packets) / sample_rate
    csi = np.ones((num_packets, 2, 52), dtype=complex)
    for i in range(num_packets):
        base_phase = 2 * np.pi * step_freq * t[i] * amplitude
        noise = rng.normal(0, noise_std, size=(2, 52))
        csi[i] *= np.exp(1j * (base_phase + noise))
    return csi


class TestClassify:
    def test_classify_returns_expected_keys(self, classifier):
        csi = _make_gait_csi()
        result = classifier.classify(csi, sample_rate=50.0)
        assert "gait_type" in result
        assert "confidence" in result
        assert "step_frequency_hz" in result
        assert "step_frequency_bpm" in result
        assert "doppler_variance" in result
        assert "gait_period_s" in result

    def test_classify_confidence_range(self, classifier):
        csi = _make_gait_csi()
        result = classifier.classify(csi, sample_rate=50.0)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_matches_known_template(self, classifier):
        """Classify with templates adjusted to match extracted features."""
        csi = _make_gait_csi()
        features = classifier.extract_gait_features(csi, sample_rate=50.0)
        # Register a template matching the extracted features
        classifier.add_template("test_gait", csi, sample_rate=50.0)
        result = classifier.classify(csi, sample_rate=50.0)
        assert result["gait_type"] == "test_gait"
        assert result["confidence"] > 0.5

    def test_classify_stationary(self, classifier):
        """Constant CSI should classify as stationary (near-zero features)."""
        csi = np.ones((200, 2, 52), dtype=complex)
        result = classifier.classify(csi, sample_rate=50.0)
        assert result["gait_type"] == "stationary"
        assert result["confidence"] > 0.0

    def test_classify_no_templates(self):
        clf = GaitClassifier(config=None)
        clf._templates = {}
        csi = _make_gait_csi()
        result = clf.classify(csi, sample_rate=50.0)
        assert result["gait_type"] == "unknown"
        assert result["confidence"] == 0.0

    def test_classify_closest_template_wins(self, classifier):
        """Template with smallest weighted distance should win."""
        # Set templates far apart so classification is unambiguous
        classifier._templates = {
            "slow": {"step_frequency_hz": 0.5, "doppler_variance": 0.01},
            "fast": {"step_frequency_hz": 3.0, "doppler_variance": 0.8},
        }
        csi = _make_gait_csi()
        features = classifier.extract_gait_features(csi, sample_rate=50.0)
        # Manually compute distances to verify classify picks the closer one
        slow_dist = 2.0 * abs(features["step_frequency_hz"] - 0.5) / 3.0 + abs(
            features["doppler_variance"] - 0.01
        )
        fast_dist = 2.0 * abs(features["step_frequency_hz"] - 3.0) / 3.0 + abs(
            features["doppler_variance"] - 0.8
        )
        result = classifier.classify(csi, sample_rate=50.0)
        expected = "slow" if slow_dist < fast_dist else "fast"
        assert result["gait_type"] == expected

    def test_classify_features_in_result(self, classifier):
        """All extract_gait_features keys should appear in classify result."""
        csi = _make_gait_csi()
        features = classifier.extract_gait_features(csi, sample_rate=50.0)
        result = classifier.classify(csi, sample_rate=50.0)
        for key in features:
            assert key in result
            assert result[key] == features[key]


class TestAddTemplate:
    def test_add_template_updates_templates(self, classifier):
        csi = _make_gait_csi(step_freq=2.0)
        classifier.add_template("custom_gait", csi, sample_rate=50.0)
        assert "custom_gait" in classifier._templates
        tmpl = classifier._templates["custom_gait"]
        assert "step_frequency_hz" in tmpl
        assert "doppler_variance" in tmpl

    def test_classify_with_custom_template(self, classifier):
        csi = _make_gait_csi(step_freq=2.0)
        classifier.add_template("custom_gait", csi, sample_rate=50.0)
        result = classifier.classify(csi, sample_rate=50.0)
        assert result["gait_type"] == "custom_gait"

    def test_add_template_overwrites_default(self, classifier):
        original_freq = classifier._templates["walking"]["step_frequency_hz"]
        csi = _make_gait_csi(step_freq=2.0)
        classifier.add_template("walking", csi, sample_rate=50.0)
        new_freq = classifier._templates["walking"]["step_frequency_hz"]
        assert new_freq != pytest.approx(original_freq, abs=0.01)


class TestDefaultTemplates:
    def test_default_templates_exist(self, classifier):
        assert "walking" in classifier._templates
        assert "running" in classifier._templates
        assert "stationary" in classifier._templates

    def test_walking_template_values(self, classifier):
        t = classifier._templates["walking"]
        assert t["step_frequency_hz"] == 1.8
        assert t["doppler_variance"] == 0.15

    def test_running_template_values(self, classifier):
        t = classifier._templates["running"]
        assert t["step_frequency_hz"] == 2.8
        assert t["doppler_variance"] == 0.6

    def test_stationary_template_values(self, classifier):
        t = classifier._templates["stationary"]
        assert t["step_frequency_hz"] == 0.1
        assert t["doppler_variance"] == 0.01