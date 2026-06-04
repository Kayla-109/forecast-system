"""Forecasting dashboard for Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_master_data():
    path = Path("data/raw/master_data.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_sales_data():
    path = Path("data/raw/sales_data.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_forecast_detail(sku_id: str, pharmacy_type: str):
    path = Path(f"reports/forecast_detail_{sku_id}_{pharmacy_type}.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_external_signals():
    path = Path("data/raw/external_signals.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def compute_mase(actual: np.ndarray, predicted: np.ndarray) -> float:
    mae = np.mean(np.abs(actual - predicted))
    naive = np.mean(np.abs(actual[1:] - actual[:-1])) if len(actual) > 1 else mae
    return float(mae / naive) if naive > 0 else float("inf")


def compute_trend_direction(forecast_series: pd.Series) -> str:
    """Compare last 4 weeks vs previous 4 weeks."""
    if len(forecast_series) < 8:
        return "flat"
    recent = forecast_series.iloc[-4:].mean()
    previous = forecast_series.iloc[-8:-4].mean()
    if recent > previous * 1.05:
        return "up"
    elif recent < previous * 0.95:
        return "down"
    return "flat"


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def show_forecast(lang: str = "en"):
    """Render the forecasting dashboard."""
    st.title(get_text("forecast_title", lang))
    st.divider()

    # Load data
    master = load_master_data()
    sales = load_sales_data()

    if master is None or sales is None:
        st.warning(get_text("forecast_no_data", lang))
        return

    # Ensure is_demo_sku exists
    if "is_demo_sku" not in master.columns:
        master["is_demo_sku"] = 0

    # ------------------------------------------------------------------
    # Sidebar controls
    # ------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        # SKU selector: show Chinese name + English name + SKU ID
        sku_options = master.apply(
            lambda r: f"{r['name_cn']} ({r['name_en']}) — {r['sku_id']}", axis=1
        ).tolist()
        sku_map = dict(zip(sku_options, master["sku_id"].tolist()))
        selected_sku_label = st.selectbox(
            get_text("forecast_select_sku", lang),
            options=sku_options,
            index=0,
        )
        selected_sku = sku_map[selected_sku_label]

    with col2:
        pharmacy_types = ["hospital", "chain", "independent"]
        selected_pharmacy = st.selectbox(
            get_text("forecast_select_pharmacy", lang),
            options=pharmacy_types,
            index=1,  # default to chain
            format_func=lambda x: x.capitalize(),
        )

    # Check if demo SKU
    is_demo = master[master["sku_id"] == selected_sku]["is_demo_sku"].values[0] == 1

    # ------------------------------------------------------------------
    # Load forecast detail
    # ------------------------------------------------------------------
    forecast_df = load_forecast_detail(selected_sku, selected_pharmacy)

    if forecast_df is None or len(forecast_df) == 0:
        st.warning(get_text("forecast_no_data", lang))
        return

    # Load historical sales for the same SKU + pharmacy
    hist = sales[
        (sales["sku_id"] == selected_sku) & (sales["pharmacy_type"] == selected_pharmacy)
    ].sort_values("date")

    # ------------------------------------------------------------------
    # Plotly chart: Historical + Forecast
    # ------------------------------------------------------------------
    fig = go.Figure()

    # Historical sales (last 52 weeks for context)
    hist_plot = hist.tail(52) if len(hist) > 52 else hist
    fig.add_trace(go.Scatter(
        x=hist_plot["date"],
        y=hist_plot["units_sold"],
        mode="lines",
        name=get_text("forecast_actual", lang),
        line=dict(color="black", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y}<extra></extra>",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["forecast"],
        mode="lines",
        name=get_text("forecast_predicted", lang),
        line=dict(color="#3498db", width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.1f}<extra></extra>",
    ))

    # Confidence interval (if available)
    if "lower_90" in forecast_df.columns and "upper_90" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["upper_90"],
            mode="lines",
            line=dict(width=0),
            showlegend=True,
            name=get_text("forecast_ci", lang),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["lower_90"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(52, 152, 219, 0.2)",
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=get_text("forecast_chart_title", lang),
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Metrics cards
    # ------------------------------------------------------------------
    st.subheader(get_text("forecast_metrics", lang))

    actual = forecast_df["units_sold"].values
    predicted = forecast_df["forecast"].values

    mape = compute_mape(actual, predicted)
    smape_v = compute_smape(actual, predicted)
    mase_v = compute_mase(actual, predicted)
    trend = compute_trend_direction(forecast_df["forecast"])

    trend_label = {
        "up": get_text("forecast_trend_up", lang),
        "down": get_text("forecast_trend_down", lang),
        "flat": get_text("forecast_trend_flat", lang),
    }.get(trend, get_text("forecast_trend_flat", lang))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(get_text("forecast_mape", lang), f"{mape:.1f}%")
    m2.metric(get_text("forecast_smape", lang), f"{smape_v:.1f}%")
    m3.metric(get_text("forecast_mase", lang), f"{mase_v:.2f}")
    m4.metric(get_text("forecast_trend", lang), trend_label)

    # ------------------------------------------------------------------
    # Feature importance (Demo SKU only)
    # ------------------------------------------------------------------
    if is_demo:
        st.divider()
        st.subheader(get_text("forecast_feature_importance", lang))
        st.caption(get_text("forecast_feature_importance_hint", lang))

        fi_path = Path("outputs/figures/feature_importance.png")
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)
        else:
            st.info(get_text("forecast_feature_importance_missing", lang))
    else:
        st.divider()
        st.info(get_text("forecast_feature_importance_missing", lang))
