"""Dark theme CSS injection for WiFi Mapping dashboard."""

import streamlit as st


def inject_theme():
    """Inject custom dark theme CSS matching the project aesthetic.

    Colors:
    - Background: #0e1117
    - Sidebar: #1a1a2e
    - Anchors: #00d4ff (cyan)
    - Positions: #00ff88 (green)
    - Alerts: #ff4444 (red)
    - Accent: #6c63ff (purple)
    """
    st.markdown(
        """
        <style>
        /* Global background */
        .stApp {
            background-color: #0e1117 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1a1a2e !important;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stText {
            color: #e0e0e0 !important;
        }

        /* Main text */
        .stMarkdown, .stText, p, span, label {
            color: #e0e0e0 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #00d4ff !important;
            font-family: 'Fira Code', 'Consolas', monospace !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #888888 !important;
            font-family: 'Fira Code', 'Consolas', monospace !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'Fira Code', 'Consolas', monospace !important;
        }

        /* Metric container cards */
        [data-testid="stMetric"] {
            background-color: #1a1a2e !important;
            border: 1px solid #2a2a4e !important;
            border-radius: 8px !important;
            padding: 12px 16px !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #1a1a2e !important;
            color: #00d4ff !important;
            border: 1px solid #00d4ff !important;
            border-radius: 6px !important;
            font-family: 'Fira Code', 'Consolas', monospace !important;
        }
        .stButton > button:hover {
            background-color: #00d4ff !important;
            color: #0e1117 !important;
        }

        /* Dataframes */
        .stDataFrame {
            border: 1px solid #2a2a4e !important;
            border-radius: 8px !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #888888 !important;
            font-family: 'Fira Code', 'Consolas', monospace !important;
        }
        .stTabs [aria-selected="true"] {
            color: #00d4ff !important;
            border-bottom-color: #00d4ff !important;
        }

        /* Sliders and inputs */
        .stSlider, .stNumberInput, .stTextInput, .stSelectbox {
            color: #e0e0e0 !important;
        }

        /* Status indicators */
        .status-online {
            color: #00ff88 !important;
        }
        .status-offline {
            color: #ff4444 !important;
        }
        .status-warning {
            color: #ffaa00 !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            color: #e0e0e0 !important;
            border: 1px solid #2a2a4e !important;
            border-radius: 6px !important;
        }

        /* Progress bar */
        .stProgress > div > div {
            background-color: #00d4ff !important;
        }

        /* Page links in sidebar */
        [data-testid="stSidebarNav"] a {
            color: #e0e0e0 !important;
            font-family: 'Fira Code', 'Consolas', monospace !important;
        }
        [data-testid="stSidebarNav"] a:hover {
            color: #00d4ff !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1a1a2e;
        }
        ::-webkit-scrollbar-thumb {
            background: #2a2a4e;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3a3a6e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, status: str) -> str:
    """Return an HTML status badge.

    Args:
        label: Badge text.
        status: 'online', 'offline', or 'warning'.

    Returns:
        HTML string for the badge.
    """
    colors = {
        "online": ("#00ff88", "#0a2a1a"),
        "offline": ("#ff4444", "#2a0a0a"),
        "warning": ("#ffaa00", "#2a1a0a"),
    }
    fg, bg = colors.get(status, ("#888888", "#1a1a1a"))
    return (
        f'<span style="background-color:{bg}; color:{fg}; '
        f"padding:2px 10px; border-radius:12px; font-size:0.8rem; "
        f'font-family:Fira Code,Consolas,monospace; '
        f"border:1px solid {fg}40;\">{label}</span>"
    )


def section_header(title: str, icon: str = "") -> None:
    """Render a styled section header.

    Args:
        title: Section title text.
        icon: Optional emoji icon prefix.
    """
    prefix = f"{icon} " if icon else ""
    st.markdown(
        f'<h3 style="color:#00d4ff; font-family:Fira Code,Consolas,monospace; '
        f'border-bottom:1px solid #2a2a4e; padding-bottom:8px;">'
        f"{prefix}{title}</h3>",
        unsafe_allow_html=True,
    )