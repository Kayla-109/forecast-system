"""Inventory health dashboard for Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_inventory_data():
    path = Path("reports/replenishment_recommendations.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def show_inventory(lang: str = "en"):
    """Render the inventory health dashboard."""
    st.title(get_text("inventory_title", lang))
    st.divider()

    df = load_inventory_data()
    if df is None:
        st.warning(get_text("inventory_no_data", lang))
        return

    # ------------------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------------------
    st.subheader(get_text("inventory_kpi_title", lang))

    total = len(df)
    red_count = len(df[df["risk_level"] == "Red"])
    yellow_count = len(df[df["risk_level"] == "Yellow"])
    green_count = len(df[df["risk_level"] == "Green"])
    expiry_count = len(df[df["expiry_weeks"] < 24])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(get_text("inventory_total_skus", lang), total)
    k2.metric(get_text("inventory_red_count", lang), red_count, delta_color="inverse")
    k3.metric(get_text("inventory_yellow_count", lang), yellow_count)
    k4.metric(get_text("inventory_green_count", lang), green_count)
    k5.metric(get_text("inventory_expiry_risk_count", lang), expiry_count, delta_color="inverse")

    st.divider()

    # ------------------------------------------------------------------
    # ABC / XYZ Interactive Scatter Plot
    # ------------------------------------------------------------------
    st.subheader(get_text("inventory_abc_xyz_title", lang))
    st.caption(get_text("inventory_abc_tooltip", lang))

    # Prepare data: one row per SKU (aggregate across pharmacy types)
    sku_df = df.groupby(["sku_id", "name_cn"]).agg({
        "abc_class": "first",
        "xyz_class": "first",
        "current_stock": "sum",
        "risk_level": lambda x: "Red" if "Red" in x.values else ("Yellow" if "Yellow" in x.values else "Green"),
        "vbp_status": "first",
    }).reset_index()

    # Numeric mappings for axes
    abc_map = {"A": 1, "B": 2, "C": 3}
    xyz_map = {"X": 1, "Y": 2, "Z": 3}
    sku_df["abc_num"] = sku_df["abc_class"].map(abc_map)
    sku_df["xyz_num"] = sku_df["xyz_class"].map(xyz_map)

    # Add jitter for visual separation
    np.random.seed(42)
    sku_df["abc_jitter"] = sku_df["abc_num"] + np.random.normal(0, 0.08, len(sku_df))
    sku_df["xyz_jitter"] = sku_df["xyz_num"] + np.random.normal(0, 0.08, len(sku_df))

    # Color by risk level
    color_map = {"Red": "#e74c3c", "Yellow": "#f39c12", "Green": "#2ecc71"}

    fig = px.scatter(
        sku_df,
        x="abc_jitter",
        y="xyz_jitter",
        color="risk_level",
        color_discrete_map=color_map,
        hover_data={
            "name_cn": True,
            "abc_class": True,
            "xyz_class": True,
            "current_stock": True,
            "vbp_status": True,
            "abc_jitter": False,
            "xyz_jitter": False,
        },
        size="current_stock",
        size_max=25,
        opacity=0.8,
    )

    fig.update_layout(
        xaxis=dict(
            title="ABC Class",
            tickvals=[1, 2, 3],
            ticktext=["A (Top 80%)", "B (80-95%)", "C (Bottom 5%)"],
            range=[0.5, 3.5],
        ),
        yaxis=dict(
            title="XYZ Class",
            tickvals=[1, 2, 3],
            ticktext=["X (CV<0.5)", "Y (CV 0.5-1.0)", "Z (CV>1.0)"],
            range=[0.5, 3.5],
        ),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=500,
    )

    # Add quadrant background shapes
    fig.add_hline(y=1.5, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_hline(y=2.5, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_vline(x=1.5, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_vline(x=2.5, line_dash="dot", line_color="gray", opacity=0.3)

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Expiry Risk Heatmap (Static PNG from Day 7)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader(get_text("inventory_expiry_title", lang))
    st.caption(get_text("inventory_expiry_tooltip", lang))

    expiry_path = Path("outputs/figures/expiry_risk_heatmap.png")
    if expiry_path.exists():
        st.image(str(expiry_path), use_container_width=True)
    else:
        st.info("Expiry risk heatmap not found. Run Day 7 scripts to generate.")

    # ------------------------------------------------------------------
    # Inventory Dashboard Overview (optional second static figure)
    # ------------------------------------------------------------------
    dashboard_path = Path("outputs/figures/inventory_dashboard.png")
    if dashboard_path.exists():
        st.divider()
        st.subheader("Inventory Overview")
        st.image(str(dashboard_path), use_container_width=True)
