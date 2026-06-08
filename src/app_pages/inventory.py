"""Inventory health dashboard for Streamlit app."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_master_data, load_historical_demand, load_replenishment
from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def show_inventory(lang: str = "en"):
    """Render the inventory health dashboard."""
    st.title(get_text("inventory_title", lang))
    st.divider()

    master = load_master_data()
    demand = load_historical_demand()
    replenishment = load_replenishment()

    if master is None or demand is None:
        st.warning(get_text("inventory_no_data", lang))
        return

    # ------------------------------------------------------------------
    # Compute ABC / XYZ from historical demand
    # ------------------------------------------------------------------
    # Total demand per SKU
    sku_demand = demand.groupby("sku_id").agg({
        "demand_total": "sum",
        "demand_total": ["sum", "std", "mean"],
    }).reset_index()
    sku_demand.columns = ["sku_id", "total_demand", "std_demand", "mean_demand"]
    sku_demand["cv"] = sku_demand["std_demand"] / sku_demand["mean_demand"]
    sku_demand = sku_demand.sort_values("total_demand", ascending=False)
    sku_demand["cum_pct"] = sku_demand["total_demand"].cumsum() / sku_demand["total_demand"].sum() * 100

    def abc_label(pct):
        if pct <= 80:
            return "A"
        elif pct <= 95:
            return "B"
        return "C"

    def xyz_label(cv):
        if cv < 0.5:
            return "X"
        elif cv <= 1.0:
            return "Y"
        return "Z"

    sku_demand["abc_class"] = sku_demand["cum_pct"].apply(abc_label)
    sku_demand["xyz_class"] = sku_demand["cv"].apply(xyz_label)

    # Merge with master
    sku_demand = sku_demand.merge(master[["sku_id", "generic_name_cn"]], on="sku_id", how="left")

    # ------------------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------------------
    st.subheader(get_text("inventory_kpi_title", lang))

    if replenishment is not None:
        red_count = len(replenishment[replenishment["priority"] == "High"])
        green_count = len(replenishment[replenishment["priority"] == "Low"])
    else:
        red_count = green_count = 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(get_text("inventory_total_skus", lang), len(sku_demand))
    k2.metric(get_text("inventory_a_class", lang), len(sku_demand[sku_demand["abc_class"] == "A"]))
    k3.metric(get_text("inventory_x_class", lang), len(sku_demand[sku_demand["xyz_class"] == "X"]))
    k4.metric(get_text("inventory_z_class", lang), len(sku_demand[sku_demand["xyz_class"] == "Z"]))

    st.divider()

    # ------------------------------------------------------------------
    # ABC / XYZ Interactive Scatter Plot
    # ------------------------------------------------------------------
    st.subheader(get_text("inventory_abc_xyz_title", lang))
    st.caption(get_text("inventory_abc_tooltip", lang))

    abc_map = {"A": 1, "B": 2, "C": 3}
    xyz_map = {"X": 1, "Y": 2, "Z": 3}
    sku_demand["abc_num"] = sku_demand["abc_class"].map(abc_map)
    sku_demand["xyz_num"] = sku_demand["xyz_class"].map(xyz_map)

    np.random.seed(42)
    sku_demand["abc_jitter"] = sku_demand["abc_num"] + np.random.normal(0, 0.08, len(sku_demand))
    sku_demand["xyz_jitter"] = sku_demand["xyz_num"] + np.random.normal(0, 0.08, len(sku_demand))

    color_map = {"A": "#e74c3c", "B": "#f39c12", "C": "#2ecc71"}

    fig = px.scatter(
        sku_demand,
        x="abc_jitter",
        y="xyz_jitter",
        color="abc_class",
        color_discrete_map=color_map,
        hover_data={"generic_name_cn": True, "abc_class": True, "xyz_class": True, "total_demand": True},
        size="total_demand",
        size_max=25,
        opacity=0.7,
    )

    fig.update_layout(
        xaxis=dict(title="ABC Class", tickvals=[1, 2, 3], ticktext=["A (Top 80%)", "B (80-95%)", "C (Bottom 5%)"], range=[0.5, 3.5]),
        yaxis=dict(title="XYZ Class", tickvals=[1, 2, 3], ticktext=["X (CV<0.5)", "Y (CV 0.5-1.0)", "Z (CV>1.0)"], range=[0.5, 3.5]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=500,
    )

    fig.add_hline(y=1.5, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_hline(y=2.5, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_vline(x=1.5, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_vline(x=2.5, line_dash="dot", line_color="gray", opacity=0.3)

    st.plotly_chart(fig, use_container_width=True)
