"""Home page for Streamlit app."""

import streamlit as st
from src.utils.i18n import get_text


def show_home(lang: str = "en"):
    """Render the home page."""
    st.title(get_text("app_title", lang))
    st.caption(get_text("app_subtitle", lang))
    st.divider()

    # Welcome section
    st.header(f"👋 {get_text('home_welcome', lang)}")
    st.markdown(get_text("home_intro", lang))

    # How to use
    with st.expander(f"📖 {get_text('home_how_to_use', lang)}", expanded=True):
        st.markdown(get_text("home_guide", lang))

    # Data scope
    with st.expander(f"📊 {get_text('home_data_scope', lang)}"):
        st.markdown(get_text("home_data_detail", lang))

    # System architecture diagram (text-based)
    st.subheader("🛠️ System Architecture")
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    with arch_col1:
        st.info("**Data Layer**\n\n• 80 SKUs\n• 3 Pharmacy Types\n• 104 Weeks\n• External Signals")
    with arch_col2:
        st.info("**Algorithm Layer**\n\n• ETS (Smooth)\n• Croston/SBA (Intermittent)\n• XGBoost (Demo SKUs)\n• FEFO (R,S) Policy")
    with arch_col3:
        st.info("**Application Layer**\n\n• Demand Forecast\n• Replenishment\n• Inventory Health\n• Policy Simulator")
