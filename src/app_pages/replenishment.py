"""Replenishment recommendations page for Streamlit app."""

import pandas as pd
import streamlit as st

from src.data_loader import load_master_data, load_replenishment
from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def show_replenishment(lang: str = "en"):
    """Render the replenishment dashboard."""
    st.title(get_text("replenishment_title", lang))
    st.divider()

    df = load_replenishment()
    master = load_master_data()

    if df is None:
        st.warning(get_text("replenishment_no_data", lang))
        return

    # Merge with master for names
    df = df.merge(master[["sku_id", "generic_name_cn"]], on="sku_id", how="left")

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    filt_col1, filt_col2 = st.columns(2)

    with filt_col1:
        priority_options = ["All", "High", "Low"]
        selected_priority = st.selectbox(get_text("replenishment_filter_priority", lang), options=priority_options, index=0)

    with filt_col2:
        search_term = st.text_input("🔍 " + get_text("replenishment_search", lang), "")

    filtered = df.copy()
    if selected_priority != "All":
        filtered = filtered[filtered["priority"] == selected_priority]
    if search_term.strip():
        mask = (
            filtered["generic_name_cn"].astype(str).str.contains(search_term, case=False, na=False)
            | filtered["sku_id"].astype(str).str.contains(search_term, case=False, na=False)
        )
        filtered = filtered[mask]

    # ------------------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------------------
    st.subheader(get_text("inventory_kpi_title", lang))
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(get_text("inventory_total_skus", lang), len(filtered))
    kpi2.metric(get_text("replenishment_high_priority", lang), len(filtered[filtered["priority"] == "High"]))
    kpi3.metric(get_text("replenishment_low_priority", lang), len(filtered[filtered["priority"] == "Low"]))

    st.divider()

    # ------------------------------------------------------------------
    # Recommendations Table
    # ------------------------------------------------------------------
    st.subheader(get_text("replenishment_table_title", lang))

    if len(filtered) == 0:
        st.info(get_text("replenishment_no_match", lang))
    else:
        display = filtered[[
            "sku_id", "generic_name_cn", "current_inventory", "avg_monthly_demand",
            "safety_stock", "reorder_point", "suggested_order_qty", "order_value_cny",
            "priority", "lead_time_days",
        ]].copy()

        st.dataframe(display, use_container_width=True, hide_index=True)
