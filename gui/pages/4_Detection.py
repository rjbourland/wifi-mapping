"""Detection — Motion, breathing, and gait analysis."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from gui.utils.theme import inject_theme, section_header, status_badge
from gui.utils.data_loader import init_session_state, generate_synthetic_csi

st.set_page_config(page_title="Detection", page_icon="👁", layout="wide")
inject_theme()
init_session_state()

st.title("👁 Device-Free Detection")
st.markdown(
    '<span style="color:#888; font-family:Fira Code,Consolas,monospace; font-size:0.85rem;">'
    "Motion detection • Breathing rate • Gait analysis — no camera required"
    "</span>",
    unsafe_allow_html=True,
)

# --- Controls ---
col_ctrl1, col_ctrl2 = st.columns([1, 3])
with col_ctrl1:
    if st.button("▶ Run Detection", type="primary", use_container_width=True):
        _run_detection()
    if st.button("▶ Simulate Motion", use_container_width=True):
        _simulate_motion(True)
    if st.button("▶ Simulate Static", use_container_width=True):
        _simulate_motion(False)

with col_ctrl2:
    # Motion event log
    if st.session_state.motion_events:
        last = st.session_state.motion_events[-1]
        motion_active = last.get("is_motion", False)
        score = last.get("motion_score", 0)
        color = "#ff4444" if motion_active else "#00ff88"
        st.markdown(
            f'<div style="background:#1a1a2e; border:1px solid {color}40; border-radius:8px; padding:12px;">'
            f'<span style="color:{color}; font-family:Fira Code,Consolas,monospace; font-size:1.2rem;">'
            f'{"MOTION DETECTED" if motion_active else "NO MOTION"}</span><br>'
            f'<span style="color:#888; font-size:0.8rem;">Score: {score:.2f} | '
            f'Variance: {last.get("total_variance", 0):.4f}</span></div>',
            unsafe_allow_html=True,
        )

# --- Motion Detection ---
section_header("Motion Detection", "🏃")

col_mot1, col_mot2 = st.columns([2, 1])

with col_mot1:
    # CSI Variance plot
    num_packets = 100
    motion_csi = generate_synthetic_csi(num_packets, motion=bool(st.session_state.motion_events and st.session_state.motion_events[-1].get("is_motion", False)))

    from src.processing.feature_extraction import compute_variance_features
    features = compute_variance_features(motion_csi, window_size=20)

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        y=features["total_variance"],
        mode="lines",
        line=dict(color="#00d4ff", width=2),
        name="Total Variance",
    ))
    fig_var.add_hline(y=0.5, line_dash="dot", line_color="#ff4444", annotation_text="Threshold")

    fig_var.update_layout(
        xaxis_title="Window", yaxis_title="Amplitude Variance",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", family="Fira Code"),
        height=300,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_var, use_container_width=True)

with col_mot2:
    st.markdown("**Detection Parameters**")
    variance_thresh = st.slider("Variance Threshold", 0.1, 2.0, 0.5, 0.1, key="det_var_thresh")
    window_size = st.slider("Window Size", 5, 50, 20, 5, key="det_window")
    min_duration = st.slider("Min Duration (s)", 0.1, 2.0, 0.5, 0.1, key="det_min_dur")

    st.markdown("---")
    st.markdown("**Motion Events**")
    num_events = sum(1 for e in st.session_state.motion_events if e.get("is_motion", False))
    st.metric("Motion Events", num_events)
    st.metric("Total Samples", len(st.session_state.motion_events))

# --- Breathing Detection ---
section_header("Breathing Rate Estimation", "🫁")

col_br1, col_br2 = st.columns([2, 1])

with col_br1:
    # Generate breathing-like CSI data
    sample_rate = 50.0
    bpm = st.slider("Simulated BPM", 10, 30, 16, key="br_bpm")
    t = np.arange(0, 10, 1.0 / sample_rate)
    breathing_signal = 1.0 + 0.3 * np.sin(2 * np.pi * bpm / 60.0 * t)
    breathing_csi = np.ones((len(t), 2, 52), dtype=complex)
    for i in range(len(t)):
        breathing_csi[i] *= breathing_signal[i]

    detector = st.session_state.breathing_detector
    result = detector.detect(breathing_csi, sample_rate=sample_rate)

    # FFT spectrum
    from src.detection.breathing_detector import BreathingDetector
    amp = np.abs(breathing_csi).mean(axis=(1, 2))
    amp = amp - np.mean(amp)
    from scipy import signal as sp_signal
    sos = sp_signal.butter(4, [10/60, 30/60], btype="band", fs=sample_rate, output="sos")
    amp_filtered = sp_signal.sosfiltfilt(sos, amp)
    freqs = np.fft.rfftfreq(len(amp_filtered), d=1.0/sample_rate)
    spectrum = np.abs(np.fft.rfft(amp_filtered))

    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(
        x=freqs * 60,  # Convert Hz to BPM
        y=spectrum,
        mode="lines",
        line=dict(color="#00ff88", width=2),
    ))
    if result["dominant_frequency_hz"] > 0:
        fig_fft.add_vline(x=result["breathing_rate_bpm"], line_dash="dot", line_color="#ff4444",
                          annotation_text=f'{result["breathing_rate_bpm"]:.1f} BPM')

    fig_fft.update_layout(
        xaxis_title="Breaths per Minute", yaxis_title="FFT Magnitude",
        xaxis_range=[5, 40],
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", family="Fira Code"),
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_fft, use_container_width=True)

with col_br2:
    st.markdown("**Breathing Result**")
    detected = result["breathing_detected"]
    st.markdown(
        f'<div style="background:#1a1a2e; border:1px solid #2a2a4e; border-radius:8px; padding:12px; '
        f'font-family:Fira Code,Consolas,monospace;">'
        f'<div style="color:{"#00ff88" if detected else "#ff4444"}; font-size:1.1rem;">'
        f'{"DETECTED" if detected else "NOT DETECTED"}</div>'
        f'<div style="color:#00d4ff; font-size:1.5rem; margin:8px 0;">{result["breathing_rate_bpm"]:.1f} BPM</div>'
        f'<div style="color:#888; font-size:0.75rem;">'
        f"SNR: {result['snr']:.1f} | Confidence: {result['confidence']:.2f}</div></div>",
        unsafe_allow_html=True,
    )

# --- Gait Analysis ---
section_header("Gait Analysis", "🚶")

gait_csi = generate_synthetic_csi(200, motion=True)
gait_classifier = st.session_state.gait_classifier
gait_result = gait_classifier.classify(gait_csi, sample_rate=50.0)

# Gait type badge
gait_type = gait_result["gait_type"]
gait_conf = gait_result["confidence"]
gait_colors = {"walking": "#00d4ff", "running": "#ff8800", "stationary": "#888888"}
badge_color = gait_colors.get(gait_type, "#00ff88")
st.markdown(
    f'<div style="background:#1a1a2e; border:1px solid {badge_color}40; border-radius:8px; padding:12px; '
    f'font-family:Fira Code,Consolas,monospace;">'
    f'<span style="color:{badge_color}; font-size:1.3rem; font-weight:bold;">'
    f'{gait_type.upper()}</span> '
    f'<span style="color:#888; font-size:0.85rem;">(confidence: {gait_conf:.1%})</span></div>',
    unsafe_allow_html=True,
)

col_g1, col_g2, col_g3, col_g4 = st.columns(4)
with col_g1:
    st.metric("Step Frequency", f"{gait_result['step_frequency_hz']:.2f} Hz")
with col_g2:
    st.metric("Steps/Min", f"{gait_result['step_frequency_bpm']:.0f}")
with col_g3:
    st.metric("Gait Period", f"{gait_result['gait_period_s']:.2f} s")
with col_g4:
    st.metric("Doppler Variance", f"{gait_result['doppler_variance']:.4f}")


def _run_detection():
    csi = generate_synthetic_csi(100, motion=False)
    result = st.session_state.motion_detector.detect(csi)
    st.session_state.motion_events.append(result)


def _simulate_motion(motion: bool):
    csi = generate_synthetic_csi(100, motion=motion)
    result = st.session_state.motion_detector.detect(csi)
    st.session_state.motion_events.append(result)