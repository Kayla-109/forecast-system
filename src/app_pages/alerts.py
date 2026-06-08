"""Alerts center page for Streamlit app."""

import pandas as pd
import streamlit as st

from src.data_loader import load_replenishment, load_master_data
from src.utils.i18n import get_text


def show_alerts(lang: str = "en"):
    """Render the alerts center dashboard."""
    st.title(get_text("alerts_title", lang))
    st.divider()

    df = load_replenishment()
    master = load_master_data()

    if df is None:
        st.warning(get_text("inventory_no_data", lang))
        return

    df = df.merge(master[["sku_id", "generic_name_cn"]], on="sku_id", how="left")

    # ------------------------------------------------------------------
    # Identify alerts
    # ------------------------------------------------------------------
    stockout_alerts = df[df["current_inventory"] < df["safety_stock"]].copy()
    expiry_alerts = df[df["current_inventory"] > 0].copy()  # Simplified

    # ------------------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------------------
    k1, k2 = st.columns(2)
    k1.metric(get_text("alerts_total_stockout", lang), len(stockout_alerts))
    k2.metric(get_text("alerts_total_expiry", lang), len(expiry_alerts))

    st.divider()

    # ------------------------------------------------------------------
    # Stockout Alerts
    # ------------------------------------------------------------------
    st.subheader(get_text("alerts_stockout_title", lang))

    if len(stockout_alerts) == 0:
        st.success(get_text("alerts_no_alerts", lang))
    else:
        stockout_alerts = stockout_alerts.sort_values("current_inventory")
        display = stockout_alerts[["sku_id", "generic_name_cn", "current_inventory", "safety_stock",
                                   "suggested_order_qty", "priority"]].head(20)
        st.dataframe(display, use_container_width=True, hide_index=True)
        if len(stockout_alerts) > 20:
            st.caption(get_text("alerts_more", lang).format(n=len(stockout_alerts) - 20))

    st.divider()

    # ------------------------------------------------------------------
    # Low Inventory Alerts
    # ------------------------------------------------------------------
    st.subheader(get_text("alerts_expiry_title", lang))

    if len(expiry_alerts) == 0:
        st.success(get_text("alerts_no_alerts", lang))
    else:
        expiry_alerts = expiry_alerts[expiry_alerts["current_inventory"] < expiry_alerts["reorder_point"]]
        display = expiry_alerts[["sku_id", "generic_name_cn", "current_inventory", "reorder_point",
                                 "suggested_order_qty", "priority"]].head(20)
        st.dataframe(display, use_container_width=True, hide_index=True)
