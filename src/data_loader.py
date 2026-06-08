"""
Data Loader for Early-bird Module 4 Case 2
===========================================
Unified interface to load new daily data, SKU profiles, and pre-computed
forecast results (ETS, Prophet, Croston, XGBoost).

All loaders use @st.cache_data to avoid repeated I/O.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EARLY_BIRD_DIR = Path("Early-bird_Module4_case2")
DATA_DIR = EARLY_BIRD_DIR / "data"
RESULTS_DIR = EARLY_BIRD_DIR / "results"


# ---------------------------------------------------------------------------
# Master / SKU Profile Data (small, cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_master_data() -> pd.DataFrame:
    """Load products + sku_profiles merged master data."""
    products = pd.read_csv(DATA_DIR / "products.csv")
    profiles = pd.read_csv(DATA_DIR / "sku_profiles.csv")
    master = products.merge(profiles, on="sku_id", how="left")
    return master


@st.cache_data
def load_sku_profiles() -> pd.DataFrame:
    """Load sku_profiles for demand_class lookup."""
    return pd.read_csv(DATA_DIR / "sku_profiles.csv")


# ---------------------------------------------------------------------------
# Historical Demand (large, filter by SKU when needed)
# ---------------------------------------------------------------------------
@st.cache_data
def load_historical_demand() -> pd.DataFrame:
    """Load full daily demand history."""
    df = pd.read_csv(DATA_DIR / "demand_daily.csv", parse_dates=["date"])
    return df


@st.cache_data
def get_sku_history(sku_id: str) -> pd.DataFrame:
    """Get historical demand for a single SKU."""
    df = load_historical_demand()
    return df[df["sku_id"] == sku_id].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forecast Results (pre-computed by Early-bird)
# ---------------------------------------------------------------------------
@st.cache_data
def load_forecast_ets() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "ets_predictions.csv", parse_dates=["date"])


@st.cache_data
def load_forecast_prophet() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "prophet_predictions.csv", parse_dates=["date"])


@st.cache_data
def load_forecast_croston() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "croston_predictions.csv", parse_dates=["date"])


@st.cache_data
def load_forecast_xgboost() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "xgboost_predictions.csv", parse_dates=["date"])


# ---------------------------------------------------------------------------
# Unified Forecast Router
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "fast": "ets",
    "seasonal": "prophet",
    "long_tail": "croston",
    "policy_shocked": "xgboost",
}

FORECAST_LOADERS = {
    "ets": load_forecast_ets,
    "prophet": load_forecast_prophet,
    "croston": load_forecast_croston,
    "xgboost": load_forecast_xgboost,
}


@st.cache_data
def get_sku_forecast(sku_id: str, demand_class: str) -> pd.DataFrame:
    """
    Return pre-computed forecast for a SKU based on its demand_class.
    Columns are normalized to: date, forecast_total, forecast_hospital,
    forecast_chain, forecast_independent
    """
    model_name = MODEL_MAP.get(demand_class, "ets")
    loader = FORECAST_LOADERS[model_name]
    df = loader()
    df = df[df["sku_id"] == sku_id].copy()

    # Normalize column names
    if model_name == "croston":
        # Croston has both croston_ and sba_ prefixes; use SBA
        df = df.rename(columns={
            "sba_forecast total": "forecast_total",
            "sba_forecast hospital": "forecast_hospital",
            "sba_forecast chain": "forecast_chain",
            "sba_forecast independent": "forecast_independent",
        })
    elif model_name == "xgboost":
        df = df.rename(columns={
            "total": "forecast_total",
            "hospital": "forecast_hospital",
            "chain": "forecast_chain",
            "independent": "forecast_independent",
        })
    # ETS and Prophet already have forecast_ prefix

    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature Importance (XGBoost only)
# ---------------------------------------------------------------------------
@st.cache_data
def load_xgboost_feature_importance() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "xgboost_feature_importance.csv")


# ---------------------------------------------------------------------------
# Replenishment Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_replenishment() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "replenishment.csv")


# ---------------------------------------------------------------------------
# VBP Impact Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_vbp_impact() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "vbp_impact.csv")


# ---------------------------------------------------------------------------
# External Signals
# ---------------------------------------------------------------------------
@st.cache_data
def load_external_signals() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "external_signals.csv", parse_dates=["date"])


# ---------------------------------------------------------------------------
# Inventory Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_inventory() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "inventory.csv", parse_dates=["date"])


@st.cache_data
def get_sku_inventory(sku_id: str) -> pd.DataFrame:
    df = load_inventory()
    return df[df["sku_id"] == sku_id].sort_values("date").reset_index(drop=True)
