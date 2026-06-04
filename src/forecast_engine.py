"""
Demand Forecasting Engine — Day 5
=================================
Segmented forecasting pipeline:
  • Smooth (ADI < 1.32)     → ETS (Exponential Smoothing)
  • Intermittent (ADI ≥ 1.32) → Croston / SBA
  • Demo SKUs                → XGBoost + external regressors

Features:
  – MASE, MAPE, RMSE, SMAPE evaluation
  – Prediction intervals (ETS native, XGBoost residual-based)
  – Future external-signal generation (ILI%, holidays, VBP)
  – VBP-channel feature engineering
  – Bottom-up hierarchical reconciliation (ATC-1→ATC-4)
  – Model persistence (joblib)
  – Auto-generated Markdown report
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Chinese font (comment out if running on systems without SimHei)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastConfig:
    data_dir: Path = Path("data/raw")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    chart_dir: Path = Path("docs/validation")
    horizon: int = 12  # weeks ahead
    test_weeks: int = 12
    ets_seasonal_periods: int = 52
    croston_alpha: float = 0.1
    croston_method: str = "SBA"  # "Croston" or "SBA"
    random_seed: int = 42
    demo_skus: List[str] = field(default_factory=lambda: ["SKU_0020", "SKU_0033"])


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------
class ForecastEngine:
    def __init__(self, cfg: ForecastConfig | None = None):
        self.cfg = cfg or ForecastConfig()
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.report_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.chart_dir.mkdir(parents=True, exist_ok=True)

        # Containers
        self.sales: pd.DataFrame | None = None
        self.master: pd.DataFrame | None = None
        self.signals: pd.DataFrame | None = None
        self.results: pd.DataFrame | None = None
        self.models: Dict[str, any] = {}
        self.future_signals: pd.DataFrame | None = None

    # =====================================================================
    # Data Loading
    # =====================================================================
    def load_data(self) -> None:
        self.sales = pd.read_csv(self.cfg.data_dir / "sales_data.csv", parse_dates=["date"])
        self.master = pd.read_csv(self.cfg.data_dir / "master_data.csv")
        self.signals = pd.read_csv(self.cfg.data_dir / "external_signals.csv", parse_dates=["date"])
        print(f"[Load] Sales: {len(self.sales):,} rows | Master: {len(self.master)} SKUs")

    # =====================================================================
    # ADI Computation & Classification
    # =====================================================================
    @staticmethod
    def compute_adi(series: np.ndarray) -> float:
        positive = np.where(series > 0)[0]
        if len(positive) <= 1:
            return float(len(series))
        return float(np.mean(np.diff(positive)))

    @staticmethod
    def classify_demand(adi: float) -> str:
        if adi < 1.32:
            return "smooth"
        elif adi < 5.0:
            return "intermittent"
        return "lumpy"

    def compute_all_adi(self) -> pd.DataFrame:
        adi_records = []
        for (sku_id, ptype), g in self.sales.groupby(["sku_id", "pharmacy_type"]):
            adi = self.compute_adi(g["units_sold"].values)
            adi_records.append({
                "sku_id": sku_id, "pharmacy_type": ptype, "adi": adi,
                "demand_type_actual": self.classify_demand(adi),
            })
        return pd.DataFrame(adi_records)

    # =====================================================================
    # Feature Engineering (for XGBoost)
    # =====================================================================
    def build_features(self, df: pd.DataFrame, sku_info: pd.Series) -> pd.DataFrame:
        df = df.copy().sort_values("date")
        f = pd.DataFrame({"date": df["date"]})

        # Lag features
        for lag in [1, 2, 4]:
            f[f"lag_{lag}"] = df["units_sold"].shift(lag)

        # Rolling statistics
        f["rolling_mean_4"] = df["units_sold"].shift(1).rolling(4, min_periods=1).mean()
        f["rolling_mean_12"] = df["units_sold"].shift(1).rolling(12, min_periods=1).mean()
        f["rolling_std_4"] = df["units_sold"].shift(1).rolling(4, min_periods=1).std().fillna(0)

        # External signals
        f["ili_pct"] = df["ili_pct"]
        f["vbp_10_active"] = df["vbp_10_active"]
        f["is_holiday"] = df["is_holiday"]
        f["week_of_year"] = df["week_of_year"]

        # Month dummies
        month = df["date"].dt.month
        for m in range(1, 13):
            f[f"month_{m}"] = (month == m).astype(int)

        # VBP status one-hot
        f["vbp_selected"] = 1 if sku_info["vbp_status"] == "selected" else 0
        f["vbp_nonselected"] = 1 if sku_info["vbp_status"] == "non_selected" else 0

        # Pharmacy type one-hot
        ptype = sku_info.get("pharmacy_type", "chain")
        for pt in ["hospital", "chain", "independent"]:
            f[f"ptype_{pt}"] = 1 if ptype == pt else 0

        # VBP × Pharmacy interaction
        f["vbp_selected_x_pharmacy"] = f["vbp_selected"] * f[f"ptype_{ptype}"]

        # ATC-4 competition feature: count of selected drugs in same ATC-4
        atc4 = sku_info["atc4"]
        n_selected_atc4 = self.master[
            (self.master["atc4"] == atc4) & (self.master["vbp_status"] == "selected")
        ].shape[0]
        f["n_selected_atc4"] = n_selected_atc4

        # Target
        f["target"] = df["units_sold"]

        return f.reset_index(drop=True)

    # =====================================================================
    # Future External Signal Generation
    # =====================================================================
    def generate_future_signals(self, n_weeks: int | None = None) -> pd.DataFrame:
        n = n_weeks or self.cfg.horizon
        last_date = self.signals["date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n, freq="W-MON")
        week_of_year = future_dates.isocalendar().week.values

        # ILI extrapolation
        phase = 2 * np.pi * week_of_year / 52
        seasonal_ili = 3.0 * np.sin(phase - np.pi / 3) + 1.0 * np.sin(2 * phase + np.pi / 4) + 2.5
        last_trend = -0.3  # post-COVID dampening level
        trend = np.linspace(last_trend, last_trend - 0.05, n)
        noise = np.random.normal(0, 0.35, n)
        ili_pct = np.clip(seasonal_ili + trend + noise, 0.5, 12.0)

        # Holidays
        df = pd.DataFrame({"date": future_dates, "month": future_dates.month, "day": future_dates.day})
        spring = ((df["month"] == 1) & (df["day"] >= 20)) | ((df["month"] == 2) & (df["day"] <= 10))
        golden = (df["month"] == 10) & (df["day"] <= 7)
        labor = (df["month"] == 5) & (df["day"] <= 5)

        self.future_signals = pd.DataFrame({
            "date": future_dates,
            "week_of_year": week_of_year,
            "ili_pct": np.round(ili_pct, 2),
            "is_spring_festival": spring.astype(int),
            "is_golden_week": golden.astype(int),
            "is_labor_day": labor.astype(int),
            "is_holiday": (spring | golden | labor).astype(int),
            "vbp_10_active": 1,  # always active after 2025-04-01
        })
        return self.future_signals

    # =====================================================================
    # Forecast Models
    # =====================================================================
    def fit_ets(self, train_ts: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (forecast, lower, upper) for test period."""
        try:
            model = ExponentialSmoothing(
                train_ts, seasonal_periods=self.cfg.ets_seasonal_periods,
                trend="add", seasonal="add", damped_trend=True,
            ).fit(optimized=True)
            pred = model.get_forecast(steps=self.cfg.test_weeks)
            forecast = pred.predicted_mean
            conf = pred.conf_int(alpha=0.10)  # 90% CI
            lower, upper = conf.iloc[:, 0], conf.iloc[:, 1]
            self.models[f"ets_{train_ts.name}"] = model
            return forecast.values, lower.values, upper.values
        except Exception as e:
            print(f"  [ETS Warning] Fallback to naive for {train_ts.name}: {e}")
            last = train_ts.iloc[-1]
            return np.full(self.cfg.test_weeks, last), np.full(self.cfg.test_weeks, last*0.8), np.full(self.cfg.test_weeks, last*1.2)

    def fit_croston(self, train_ts: pd.Series, method: str | None = None) -> np.ndarray:
        """Croston/SBA multi-step forecast (flat forecast assumed)."""
        method = method or self.cfg.croston_method
        series = train_ts.values
        positive_idx = np.where(series > 0)[0]

        if len(positive_idx) == 0:
            return np.zeros(self.cfg.test_weeks)
        if len(positive_idx) == 1:
            return np.full(self.cfg.test_weeks, series[positive_idx[0]] / len(series))

        intervals = np.diff(positive_idx)
        sizes = series[positive_idx]
        alpha = self.cfg.croston_alpha

        s_interval = float(intervals[0])
        s_size = float(sizes[0])
        for i in range(1, len(intervals)):
            s_interval = alpha * intervals[i] + (1 - alpha) * s_interval
            s_size = alpha * sizes[i] + (1 - alpha) * s_size

        # Partial interval since last demand
        last_partial = len(series) - 1 - positive_idx[-1]
        s_interval = alpha * last_partial + (1 - alpha) * s_interval

        forecast = s_size / max(s_interval, 1e-6)
        if method == "SBA":
            forecast *= (1 - alpha / 2)

        return np.full(self.cfg.test_weeks, max(forecast, 0))

    def fit_xgboost(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    feature_cols: List[str], sku_id: str, pharmacy_type: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """XGBoost with residual-based prediction intervals."""
        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_test = test_df[feature_cols]

        model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=self.cfg.random_seed,
            objective="reg:squarederror", n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        # Residual-based 90% PI
        train_preds = model.predict(X_train)
        residuals = y_train - train_preds
        std_resid = np.std(residuals)
        lower = preds - 1.645 * std_resid
        upper = preds + 1.645 * std_resid

        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        self.models[f"xgb_{sku_id}_{pharmacy_type}"] = model
        return preds, np.vstack([lower, upper]).T, importance

    # =====================================================================
    # Evaluation Metrics
    # =====================================================================
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    @staticmethod
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        denom = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denom != 0
        return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)

    @staticmethod
    def mase(actual: np.ndarray, predicted: np.ndarray, train_actual: np.ndarray) -> float:
        mae = mean_absolute_error(actual, predicted)
        naive_errors = np.abs(train_actual[1:] - train_actual[:-1])
        mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else mae
        return float(mae / mae_naive) if mae_naive > 0 else float("inf")

    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(actual, predicted)))

    # =====================================================================
    # Full Pipeline
    # =====================================================================
    def run(self) -> pd.DataFrame:
        print("=" * 60)
        print("Demand Forecasting Engine — Day 5")
        print("=" * 60)
        self.load_data()

        # Compute ADI for classification
        adi_df = self.compute_all_adi()

        # Generate future signals
        self.generate_future_signals()

        # Pre-compute ATC-4 competition
        atc4_selected = self.master.groupby("atc4").apply(
            lambda g: (g["vbp_status"] == "selected").sum()
        ).to_dict()

        results: List[Dict] = []
        sku_pharmacy_groups = self.sales.groupby(["sku_id", "pharmacy_type"])
        total = len(sku_pharmacy_groups)

        for idx, ((sku_id, ptype), group) in enumerate(sku_pharmacy_groups, 1):
            group = group.sort_values("date").reset_index(drop=True)
            sku_info = self.master[self.master["sku_id"] == sku_id].iloc[0].to_dict()
            sku_info["pharmacy_type"] = ptype

            # Determine model
            adi_row = adi_df[(adi_df["sku_id"] == sku_id) & (adi_df["pharmacy_type"] == ptype)]
            adi = adi_row["adi"].values[0] if len(adi_row) else 99.0
            is_demo = sku_id in self.cfg.demo_skus
            designated_type = sku_info["demand_type"]

            if is_demo:
                model_name = "XGBoost"
            elif adi < 1.32:
                model_name = "ETS"
            else:
                model_name = self.cfg.croston_method

            # Split train/test
            train = group.iloc[:-self.cfg.test_weeks]
            test = group.iloc[-self.cfg.test_weeks:]
            actual = test["units_sold"].values
            train_ts = train["units_sold"]

            # Forecast
            if model_name == "ETS":
                preds, lower, upper = self.fit_ets(train_ts)
                preds = np.clip(preds, 0, None)
                lower = np.clip(lower, 0, None)
                importance = {}
            elif model_name in ("Croston", "SBA"):
                preds = self.fit_croston(train_ts, method=model_name)
                lower = preds * 0.8
                upper = preds * 1.2
                importance = {}
            else:  # XGBoost
                train_feat = self.build_features(train, sku_info)
                test_feat = self.build_features(test, sku_info)
                feature_cols = [c for c in train_feat.columns if c not in ("date", "target")]
                preds, bounds, importance = self.fit_xgboost(train_feat, test_feat, feature_cols, sku_id, ptype)
                lower, upper = bounds[:, 0], bounds[:, 1]
                preds = np.clip(preds, 0, None)

            # Evaluate
            mape_v = self.mape(actual, preds)
            rmse_v = self.rmse(actual, preds)
            smape_v = self.smape(actual, preds)
            mase_v = self.mase(actual, preds, train_ts.values)

            results.append({
                "sku_id": sku_id,
                "pharmacy_type": ptype,
                "name_cn": sku_info["name_cn"],
                "demand_type_designated": designated_type,
                "demand_type_actual": self.classify_demand(adi),
                "adi": round(adi, 2),
                "model": model_name,
                "is_demo": is_demo,
                "mape": round(mape_v, 2),
                "rmse": round(rmse_v, 2),
                "smape": round(smape_v, 2),
                "mase": round(mase_v, 2),
                "feature_importance": json.dumps(importance) if importance else "{}",
            })

            # Store detailed forecast for reporting
            test = test.copy()
            test["forecast"] = preds
            test["lower_90"] = lower
            test["upper_90"] = upper
            test.to_csv(
                self.cfg.report_dir / f"forecast_detail_{sku_id}_{ptype}.csv",
                index=False, encoding="utf-8-sig"
            )

            if idx % 20 == 0 or idx == total:
                print(f"  Progress: {idx}/{total} SKUs processed")

        self.results = pd.DataFrame(results)
        print(f"[Pipeline] Completed forecasting for {len(self.results)} SKU-pharmacy combinations.")
        return self.results

    # =====================================================================
    # Hierarchical Reconciliation (Bottom-Up)
    # =====================================================================
    def reconcile_bottom_up(self) -> pd.DataFrame:
        """Aggregate SKU forecasts up ATC hierarchy."""
        if self.results is None:
            raise RuntimeError("Run .run() first.")

        # Load detailed forecasts and aggregate
        all_forecasts = []
        for f in self.cfg.report_dir.glob("forecast_detail_*.csv"):
            df = pd.read_csv(f, parse_dates=["date"])
            sku_id = df["sku_id"].iloc[0]
            ptype = df["pharmacy_type"].iloc[0]
            info = self.master[self.master["sku_id"] == sku_id].iloc[0]
            df["atc1"] = info["atc1"]
            df["atc2"] = info["atc2"]
            df["atc3"] = info["atc3"]
            df["atc4"] = info["atc4"]
            all_forecasts.append(df)

        all_df = pd.concat(all_forecasts, ignore_index=True)

        reconciled = {}
        for level in ["atc1", "atc2", "atc3", "atc4"]:
            agg = all_df.groupby(["date", level]).agg({
                "units_sold": "sum", "forecast": "sum", "lower_90": "sum", "upper_90": "sum"
            }).reset_index()
            reconciled[level] = agg

        return all_df, reconciled

    # =====================================================================
    # Model Persistence
    # =====================================================================
    def save_models(self) -> None:
        for name, model in self.models.items():
            path = self.cfg.model_dir / f"{name}.joblib"
            joblib.dump(model, path)
        print(f"[Save] {len(self.models)} models saved to {self.cfg.model_dir}")

    # =====================================================================
    # Auto-Generated Report
    # =====================================================================
    def generate_report(self) -> None:
        if self.results is None:
            raise RuntimeError("Run .run() first.")

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# Demand Forecasting Report",
            f"\n**Generated:** {now}  ",
            f"**Horizon:** {self.cfg.horizon} weeks  ",
            f"**Test Period:** Last {self.cfg.test_weeks} weeks of historical data  ",
            "\n---\n",
            "## Executive Summary\n",
        ]

        # Summary by model
        summary = self.results.groupby("model").agg({
            "mape": "mean", "rmse": "mean", "smape": "mean", "mase": "mean", "sku_id": "count"
        }).rename(columns={"sku_id": "count"}).round(2)
        lines.append("### Average Performance by Model\n")
        lines.append(summary.to_markdown())
        lines.append("\n")

        # Best / Worst SKUs
        lines.append("## Best Performing SKUs (Lowest MAPE)\n")
        best = self.results.nsmallest(5, "mape")[["sku_id", "name_cn", "pharmacy_type", "model", "mape", "mase"]]
        lines.append(best.to_markdown(index=False))
        lines.append("\n")

        lines.append("## Worst Performing SKUs (Highest MAPE)\n")
        worst = self.results.nlargest(5, "mape")[["sku_id", "name_cn", "pharmacy_type", "model", "mape", "mase"]]
        lines.append(worst.to_markdown(index=False))
        lines.append("\n")

        # Demo SKU detail
        lines.append("## Demo SKU Performance\n")
        demo = self.results[self.results["is_demo"] == 1][[
            "sku_id", "name_cn", "pharmacy_type", "model", "mape", "rmse", "smape", "mase"
        ]]
        lines.append(demo.to_markdown(index=False))
        lines.append("\n")

        # Feature importance for demo SKUs
        lines.append("## XGBoost Feature Importance (Demo SKUs)\n")
        for _, row in demo.iterrows():
            if row["model"] == "XGBoost":
                imp = json.loads(row.get("feature_importance", "{}"))
                if imp:
                    lines.append(f"\n### {row['sku_id']} — {row['name_cn']} ({row['pharmacy_type']})\n")
                    imp_df = pd.DataFrame([{"feature": k, "importance": round(v, 4)} for k, v in imp.items()])
                    lines.append(imp_df.to_markdown(index=False))
                    lines.append("\n")

        # ADI distribution
        lines.append("## ADI Classification Check\n")
        adi_check = self.results.groupby("demand_type_actual").agg({
            "sku_id": "count", "mape": "mean", "mase": "mean"
        }).round(2)
        lines.append(adi_check.to_markdown())
        lines.append("\n")

        # Write
        report_path = self.cfg.report_dir / "forecast_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Report] Saved to {report_path}")

    # =====================================================================
    # Charting
    # =====================================================================
    def plot_demo_forecasts(self) -> None:
        """Generate forecast-vs-actual charts for Demo SKUs."""
        if self.results is None:
            return

        fig, axes = plt.subplots(len(self.cfg.demo_skus), 1, figsize=(14, 4 * len(self.cfg.demo_skus)))
        if len(self.cfg.demo_skus) == 1:
            axes = [axes]

        for ax, sku_id in zip(axes, self.cfg.demo_skus):
            # Load chain pharmacy forecast
            f_path = self.cfg.report_dir / f"forecast_detail_{sku_id}_chain.csv"
            if not f_path.exists():
                continue
            df = pd.read_csv(f_path, parse_dates=["date"])
            name = self.master.loc[self.master["sku_id"] == sku_id, "name_cn"].values[0]

            ax.plot(df["date"], df["units_sold"], label="Actual", color="black", linewidth=1.5)
            ax.plot(df["date"], df["forecast"], label="Forecast", color="steelblue", linewidth=1.5)
            ax.fill_between(df["date"], df["lower_90"], df["upper_90"], alpha=0.2, color="steelblue", label="90% PI")
            ax.axvline(df["date"].iloc[-self.cfg.test_weeks], color="red", linestyle="--", alpha=0.5, label="Train/Test Split")
            ax.set_title(f"{sku_id} | {name} | Chain Pharmacy", fontweight="bold")
            ax.set_ylabel("Units Sold")
            ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        out_path = self.cfg.chart_dir / "08_demo_forecast_vs_actual.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved {out_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Pharma Demand Forecasting Engine — Day 5")
    print("=" * 60)

    cfg = ForecastConfig()
    engine = ForecastEngine(cfg)

    # Run full pipeline
    results = engine.run()

    # Reconciliation
    engine.reconcile_bottom_up()

    # Save models
    engine.save_models()

    # Generate report
    engine.generate_report()

    # Charts
    engine.plot_demo_forecasts()

    print()
    print("=" * 60)
    print("Day 5 complete! Check reports/forecast_report.md and models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
