"""Forecasting dashboard for Streamlit app — Early-bird daily data."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    load_master_data,
    load_sku_profiles,
    get_sku_history,
    get_sku_forecast,
    load_xgboost_feature_importance,
)
from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def show_forecast(lang: str = "en"):
    """Render the forecasting dashboard with daily data."""
    st.title(get_text("forecast_title", lang))
    st.divider()

    # Load master data
    master = load_master_data()
    profiles = load_sku_profiles()

    if master is None or len(master) == 0:
        st.warning(get_text("forecast_no_data", lang))
        return

    # ------------------------------------------------------------------
    # SKU selector
    # ------------------------------------------------------------------
    sku_options = master.apply(
        lambda r: f"{r['sku_id']} — {r['generic_name_cn']} ({r['generic_name']})", axis=1
    ).tolist()
    sku_map = dict(zip(sku_options, master["sku_id"].tolist()))

    col1, col2 = st.columns(2)
    with col1:
        selected_sku_label = st.selectbox(
            get_text("forecast_select_sku", lang),
            options=sku_options,
            index=0,
        )
        selected_sku = sku_map[selected_sku_label]

    with col2:
        pharmacy_types = ["total", "hospital", "chain", "independent"]
        type_labels = {
            "total": get_text("forecast_channel_total", lang),
            "hospital": get_text("forecast_channel_hospital", lang),
            "chain": get_text("forecast_channel_chain", lang),
            "independent": get_text("forecast_channel_independent", lang),
        }
        selected_pharmacy = st.selectbox(
            get_text("forecast_select_pharmacy", lang),
            options=pharmacy_types,
            index=0,
            format_func=lambda x: type_labels.get(x, x),
        )

    # Get demand_class for selected SKU
    profile = profiles[profiles["sku_id"] == selected_sku]
    if len(profile) == 0:
        st.warning(get_text("forecast_sku_not_found", lang))
        return
    demand_class = profile.iloc[0]["demand_class"]

    # Map demand_class to model name
    model_name_map = {
        "fast": "ETS",
        "seasonal": "Prophet",
        "long_tail": "Croston/SBA",
        "policy_shocked": "XGBoost",
    }
    model_name = model_name_map.get(demand_class, "Unknown")

    # ------------------------------------------------------------------
    # Load historical + forecast data
    # ------------------------------------------------------------------
    with st.spinner(get_text("forecast_loading", lang)):
        hist = get_sku_history(selected_sku)
        forecast = get_sku_forecast(selected_sku, demand_class)

    if hist is None or len(hist) == 0:
        st.warning(get_text("forecast_no_hist_data", lang))
        return

    # Column mapping for historical data
    demand_col_map = {
        "total": "demand_total",
        "hospital": "demand_hospital",
        "chain": "demand_chain",
        "independent": "demand_independent",
    }
    hist_col = demand_col_map.get(selected_pharmacy, "demand_total")

    # Column mapping for forecast data
    forecast_col_map = {
        "total": "forecast_total",
        "hospital": "forecast_hospital",
        "chain": "forecast_chain",
        "independent": "forecast_independent",
    }
    fc_col = forecast_col_map.get(selected_pharmacy, "forecast_total")

    # ------------------------------------------------------------------
    # Plotly chart: Historical + Forecast
    # ------------------------------------------------------------------
    fig = go.Figure()

    # Historical data (all history)
    fig.add_trace(go.Scatter(
        x=hist["date"],
        y=hist[hist_col],
        mode="lines",
        name=get_text("forecast_historical", lang),
        line=dict(color="#2c3e50", width=1),
        hovertemplate="%{x|%Y-%m-%d}<br>" + get_text("forecast_historical", lang) + ": %{y}<extra></extra>",
    ))

    # Forecast
    if forecast is not None and len(forecast) > 0 and fc_col in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast["date"],
            y=forecast[fc_col],
            mode="lines",
            name=get_text("forecast_forecast_label", lang),
            line=dict(color="#e74c3c", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>" + get_text("forecast_forecast_label", lang) + ": %{y:.1f}<extra></extra>",
        ))

    # Default view: last 90 days history + all forecast
    last_hist_date = hist["date"].max()
    default_start = last_hist_date - pd.Timedelta(days=90)

    fig.update_layout(
        title=f"{selected_sku} — {model_name} — {type_labels[selected_pharmacy]}",
        xaxis_title=get_text("forecast_date", lang),
        yaxis_title=get_text("forecast_daily_demand", lang),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(range=[default_start, forecast["date"].max()] if forecast is not None and len(forecast) > 0 else [default_start, last_hist_date]),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Croston/SBA explanation
    if demand_class == "long_tail":
        st.info(
            "ℹ️ Croston/SBA produces a flat (horizontal) forecast for intermittent-demand SKUs. "
            "This is expected behavior — the model estimates the average demand rate between non-zero periods, "
            "rather than predicting daily fluctuations."
            if lang == "en" else
            "ℹ️ Croston/SBA 对间歇性需求 SKU 生成扁平（水平）预测。这是正常行为——"
            "模型估算非零需求期间的平均需求率，而非预测每日波动。"
        )

    # ------------------------------------------------------------------
    # Model info + metrics
    # ------------------------------------------------------------------
    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric(get_text("forecast_model", lang), model_name)
    m2.metric(get_text("forecast_demand_class", lang), demand_class)
    m3.metric(get_text("forecast_history_days", lang), len(hist))

    # ------------------------------------------------------------------
    # Feature importance (XGBoost only)
    # ------------------------------------------------------------------
    if demand_class == "policy_shocked":
        st.divider()
        st.subheader(get_text("forecast_feature_importance", lang))

        fi = load_xgboost_feature_importance()
        if fi is not None and len(fi) > 0:
            fi = fi.sort_values("importance", ascending=True).tail(15)
            fig_fi = go.Figure(go.Bar(
                x=fi["importance"],
                y=fi["feature"],
                orientation="h",
                marker_color="#3498db",
            ))
            fig_fi.update_layout(
                title=get_text("forecast_features_title", lang),
                xaxis_title=get_text("forecast_importance", lang),
                yaxis_title=get_text("forecast_feature", lang),
                margin=dict(l=40, r=40, t=60, b=40),
                height=400,
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info(get_text("forecast_fi_not_available", lang))
    else:
        st.divider()
        st.info(get_text("forecast_fi_not_xgboost", lang).format(model=model_name))
