"""VBP Policy Simulator page for Streamlit app."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_master_for_simulator():
    path = Path("data/raw/master_data.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def show_simulator(lang: str = "en"):
    """Render the VBP policy simulator dashboard."""
    st.title(get_text("simulator_title", lang))
    st.divider()

    master = load_master_for_simulator()
    if master is None:
        st.warning(get_text("simulator_no_data", lang))
        return

    # ------------------------------------------------------------------
    # Batch selector
    # ------------------------------------------------------------------
    batch = st.selectbox(
        get_text("simulator_select_batch", lang),
        options=[get_text("simulator_batch_10", lang)],
        index=0,
    )

    st.info(
        "💡 " + ("This simulation models the demand shift caused by National Volume-Based Procurement (VBP). "
                 "Selected drugs see hospital demand surge (+50-60%) while retail demand drops (-15-20%). "
                 "Non-selected alternatives face hospital demand collapse (-90%) but retail spillover (+50-70%)."
                 if lang == "en" else
                 "本模拟展示国家药品集采（VBP）引发的需求转移效应。中选品种医院需求激增（+50-60%），"
                 "零售需求下降（-15-20%）；非中选替代品医院需求崩塌（-90%），但零售渠道承接溢出需求（+50-70%）。"))

    # ------------------------------------------------------------------
    # Filter to affected SKUs only
    # ------------------------------------------------------------------
    affected = master[master["vbp_status"].isin(["selected", "non_selected"])].copy()

    if len(affected) == 0:
        st.warning("No affected SKUs found in master data.")
        return

    # ------------------------------------------------------------------
    # Compute before / after demand per SKU
    # ------------------------------------------------------------------
    channels = ["hospital", "chain", "independent"]
    channel_labels = {
        "hospital": get_text("simulator_channel_hospital", lang),
        "chain": get_text("simulator_channel_chain", lang),
        "independent": get_text("simulator_channel_independent", lang),
    }

    # Calculate totals
    for ch in channels:
        affected[f"before_{ch}"] = affected[f"base_demand_{ch}"]
        affected[f"after_{ch}"] = affected[f"base_demand_{ch}"] * (1 + affected[f"vbp_shock_{ch}"])

    affected["before_total"] = affected[[f"before_{c}" for c in channels]].sum(axis=1)
    affected["after_total"] = affected[[f"after_{c}" for c in channels]].sum(axis=1)
    affected["revenue_before"] = affected["before_total"] * affected["price_rmb"]
    affected["revenue_after"] = affected["after_total"] * affected["price_rmb"]

    # ------------------------------------------------------------------
    # Aggregate by therapeutic area
    # ------------------------------------------------------------------
    agg = affected.groupby("therapeutic_area").agg({
        "before_hospital": "sum", "after_hospital": "sum",
        "before_chain": "sum", "after_chain": "sum",
        "before_independent": "sum", "after_independent": "sum",
        "revenue_before": "sum", "revenue_after": "sum",
        "sku_id": "count",
    }).reset_index()
    agg = agg.rename(columns={"sku_id": "sku_count"})

    # ------------------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------------------
    st.subheader(get_text("simulator_kpi_title", lang))

    total_before_hosp = agg["before_hospital"].sum()
    total_after_hosp = agg["after_hospital"].sum()
    total_before_retail = agg["before_chain"].sum() + agg["before_independent"].sum()
    total_after_retail = agg["after_chain"].sum() + agg["after_independent"].sum()
    total_revenue_before = agg["revenue_before"].sum()
    total_revenue_after = agg["revenue_after"].sum()

    hosp_change = (total_after_hosp - total_before_hosp) / total_before_hosp * 100
    retail_change = (total_after_retail - total_before_retail) / total_before_retail * 100
    revenue_change = total_revenue_after - total_revenue_before

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        get_text("simulator_hospital_demand", lang),
        f"{hosp_change:+.1f}%",
        delta_color="normal",
    )
    k2.metric(
        get_text("simulator_retail_demand", lang),
        f"{retail_change:+.1f}%",
        delta_color="normal",
    )
    k3.metric(
        get_text("simulator_revenue_impact", lang),
        f"¥{revenue_change:,.0f}k",
        delta_color="normal",
    )
    k4.metric(
        get_text("simulator_affected_skus", lang),
        len(affected),
    )

    st.divider()

    # ------------------------------------------------------------------
    # Grouped Bar Chart: Before vs After by Therapeutic Area
    # ------------------------------------------------------------------
    st.subheader(get_text("simulator_impact_chart_title", lang))

    # Prepare long-form data for Plotly
    chart_data = []
    for _, row in agg.iterrows():
        area = row["therapeutic_area"]
        # Before
        chart_data.append({
            "therapeutic_area": area,
            "period": get_text("simulator_before", lang),
            "hospital": row["before_hospital"],
            "chain": row["before_chain"],
            "independent": row["before_independent"],
        })
        # After
        chart_data.append({
            "therapeutic_area": area,
            "period": get_text("simulator_after", lang),
            "hospital": row["after_hospital"],
            "chain": row["after_chain"],
            "independent": row["after_independent"],
        })

    chart_df = pd.DataFrame(chart_data)

    fig = go.Figure()
    colors = {"hospital": "#3498db", "chain": "#2ecc71", "independent": "#f39c12"}

    for ch in channels:
        for period in [get_text("simulator_before", lang), get_text("simulator_after", lang)]:
            sub = chart_df[chart_df["period"] == period]
            fig.add_trace(go.Bar(
                name=f"{channel_labels[ch]} — {period}",
                x=sub["therapeutic_area"],
                y=sub[ch],
                marker_color=colors[ch],
                opacity=0.9 if period == get_text("simulator_after", lang) else 0.5,
                legendgroup=ch,
                showlegend=True,
                offsetgroup=f"{ch}_{period}",
            ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Therapeutic Area",
        yaxis_title="Weekly Demand (units)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Affected SKU Detail Table
    # ------------------------------------------------------------------
    st.divider()
    st.subheader(get_text("simulator_table_title", lang))

    # Create display table
    display = affected[[
        "sku_id", "name_cn", "name_en", "therapeutic_area", "vbp_status",
        "price_rmb", "before_total", "after_total",
    ]].copy()

    display["demand_change_pct"] = ((display["after_total"] - display["before_total"]) / display["before_total"] * 100).round(1)
    display["revenue_change"] = ((display["after_total"] - display["before_total"]) * display["price_rmb"]).round(0)

    # Translate vbp_status
    status_map = {
        "selected": get_text("simulator_selected", lang),
        "non_selected": get_text("simulator_nonselected", lang),
    }
    display["vbp_status_label"] = display["vbp_status"].map(status_map)

    display = display.rename(columns={
        "name_cn": get_text("alerts_sku", lang),
        "name_en": "English Name",
        "therapeutic_area": "Area",
        "vbp_status_label": "VBP Status",
        "price_rmb": "Price (RMB)",
        "before_total": get_text("simulator_before", lang) + " Demand",
        "after_total": get_text("simulator_after", lang) + " Demand",
        "demand_change_pct": "Demand Change %",
        "revenue_change": "Revenue Impact (RMB)",
    })

    st.dataframe(display, use_container_width=True, hide_index=True)
