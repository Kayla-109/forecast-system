"""Alerts center page for Streamlit app."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_alerts_data():
    path = Path("reports/replenishment_recommendations.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Risk card renderer
# ---------------------------------------------------------------------------
def render_alert_card(row: pd.Series, lang: str, alert_type: str):
    """Render a single alert as a styled card."""
    if alert_type == "stockout":
        color = "#e74c3c"
        icon = "⏰"
        subtitle = f"{get_text('alerts_days_supply', lang)}: **{row['days_supply']} days**"
        action = "立即补货 / Restock immediately"
    else:  # expiry
        color = "#f39c12"
        icon = "📅"
        expiry_months = round(row['expiry_weeks'] / 4.33, 1)
        subtitle = f"{get_text('alerts_expiry_weeks', lang)}: **{row['expiry_weeks']}** (~{expiry_months} months)"
        action = "促销清仓或调拨 / Promote or transfer"

    st.markdown(f"""
    <div style="
        border-left: 5px solid {color};
        background-color: #fafafa;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 1.1rem; font-weight: 600;">{icon} {row['name_cn']} <span style="color: #7f8c8d; font-size: 0.9rem;">({row['sku_id']})</span></span>
            <span style="background-color: {color}; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">{row['pharmacy_type'].capitalize()}</span>
        </div>
        <div style="margin-top: 6px; color: #555;">
            {subtitle} &nbsp;|&nbsp; {get_text('alerts_risk_reason', lang)}: {row['risk_reason']}
        </div>
        <div style="margin-top: 6px; font-size: 0.9rem; color: {color}; font-weight: 500;">
            💡 {get_text('alerts_action_needed', lang)}: {action}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def show_alerts(lang: str = "en"):
    """Render the alerts center dashboard."""
    st.title(get_text("alerts_title", lang))
    st.divider()

    df = load_alerts_data()
    if df is None:
        st.warning(get_text("inventory_no_data", lang))
        return

    # ------------------------------------------------------------------
    # Filter by pharmacy type
    # ------------------------------------------------------------------
    pharmacy_types = ["All", "hospital", "chain", "independent"]
    pharmacy_labels = {
        "All": "All",
        "hospital": "Hospital",
        "chain": "Chain",
        "independent": "Independent",
    }
    selected_pharmacy = st.selectbox(
        get_text("alerts_filter_pharmacy", lang),
        options=pharmacy_types,
        index=0,
        format_func=lambda x: pharmacy_labels.get(x, x),
    )

    # Apply pharmacy filter
    filtered = df.copy()
    if selected_pharmacy != "All":
        filtered = filtered[filtered["pharmacy_type"] == selected_pharmacy]

    # ------------------------------------------------------------------
    # Identify alerts
    # ------------------------------------------------------------------
    stockout_alerts = filtered[filtered["days_supply"] < 5].copy()
    expiry_alerts = filtered[filtered["expiry_weeks"] < 24].copy()

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
        st.success(get_text("alerts_no_alerts", lang) if len(expiry_alerts) == 0 else "No stockout alerts.")
    else:
        # Sort by days_supply ascending (most urgent first)
        stockout_alerts = stockout_alerts.sort_values("days_supply")
        for _, row in stockout_alerts.head(20).iterrows():
            render_alert_card(row, lang, "stockout")
        if len(stockout_alerts) > 20:
            st.caption(f"... and {len(stockout_alerts) - 20} more stockout alerts.")

    st.divider()

    # ------------------------------------------------------------------
    # Expiry Alerts
    # ------------------------------------------------------------------
    st.subheader(get_text("alerts_expiry_title", lang))

    if len(expiry_alerts) == 0:
        st.success(get_text("alerts_no_alerts", lang) if len(stockout_alerts) == 0 else "No expiry alerts.")
    else:
        # Sort by expiry_weeks ascending (most urgent first)
        expiry_alerts = expiry_alerts.sort_values("expiry_weeks")
        for _, row in expiry_alerts.head(20).iterrows():
            render_alert_card(row, lang, "expiry")
        if len(expiry_alerts) > 20:
            st.caption(f"... and {len(expiry_alerts) - 20} more expiry alerts.")
