"""
Pharma Demand Forecasting & Intelligent Replenishment System
Streamlit Application — Day 8-10 Skeleton + Forecast Page

Run with: streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src.app_pages.home import show_home
from src.app_pages.forecast import show_forecast
from src.app_pages.replenishment import show_replenishment
from src.app_pages.inventory import show_inventory
from src.app_pages.simulator import show_simulator
from src.app_pages.alerts import show_alerts
from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pharma Demand Forecasting",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS for professional look
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Main title */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #7f8c8d !important;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 8px !important;
    }

    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 8px;
    }

    /* Tooltip helper */
    .tooltip-text {
        font-size: 0.85rem;
        color: #7f8c8d;
        font-style: italic;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Check data availability
# ---------------------------------------------------------------------------
def check_data_availability() -> dict:
    """Check if required data files exist."""
    checks = {
        "master_data": Path("data/raw/master_data.csv").exists(),
        "sales_data": Path("data/raw/sales_data.csv").exists(),
        "forecast_reports": len(list(Path("reports").glob("forecast_detail_*.csv"))) > 0,
    }
    return checks


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple:
    """Render sidebar with language selector and navigation.
    Returns (lang_code, selected_page)."""

    # Language selector
    lang = st.sidebar.radio(
        "Language / 语言",
        options=["English", "中文"],
        index=0,
        help="Select interface language",
    )
    lang_code = "en" if lang == "English" else "zh"

    st.sidebar.divider()

    # Navigation
    page = st.sidebar.radio(
        get_text("sidebar_nav", lang_code),
        options=[
            get_text("page_home", lang_code),
            get_text("page_forecast", lang_code),
            get_text("page_replenishment", lang_code),
            get_text("page_inventory", lang_code),
            get_text("page_simulator", lang_code),
            get_text("page_alerts", lang_code),
        ],
        index=0,
    )

    st.sidebar.divider()

    # Data status indicator
    checks = check_data_availability()
    with st.sidebar.expander("📂 Data Status", expanded=False):
        for key, ok in checks.items():
            icon = "✅" if ok else "❌"
            st.write(f"{icon} {key}")
        if not all(checks.values()):
            st.warning(
                "Some data files are missing. "
                "Please run the data generation scripts first."
            )

    return lang_code, page


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    lang_code, page = render_sidebar()

    # Route to page
    if page == get_text("page_home", lang_code):
        show_home(lang_code)
    elif page == get_text("page_forecast", lang_code):
        show_forecast(lang_code)
    elif page == get_text("page_replenishment", lang_code):
        show_replenishment(lang_code)
    elif page == get_text("page_inventory", lang_code):
        show_inventory(lang_code)
    elif page == get_text("page_simulator", lang_code):
        show_simulator(lang_code)
    elif page == get_text("page_alerts", lang_code):
        show_alerts(lang_code)


if __name__ == "__main__":
    main()
