"""
Day 7 — SKU-Level Demo Preparation & Model Tuning
===================================================
1. XGBoost hyperparameter tuning for Demo SKUs (Metformin, Oseltamivir)
2. 12-week walk-forward backtest with confidence bands
3. Feature importance bar chart
4. Inventory health diagnosis (ABC/XYZ + expiry risk heatmap + dashboard)
5. Summary report: day7_summary.md
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Chinese font support
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Day7Config:
    data_dir: Path = Path("data/raw")
    report_dir: Path = Path("reports")
    figure_dir: Path = Path("outputs/figures")
    forecast_dir: Path = Path("reports")
    random_seed: int = 42
    demo_skus: List[str] = None  # set in __post_init__

    def __post_init__(self):
        object.__setattr__(self, "demo_skus", ["SKU_0020", "SKU_0033"])
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature builder (inline to avoid heavy imports)
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, sku_info: pd.Series, master_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    f = pd.DataFrame({"date": df["date"]})
    for lag in [1, 2, 4]:
        f[f"lag_{lag}"] = df["units_sold"].shift(lag)
    f["rolling_mean_4"] = df["units_sold"].shift(1).rolling(4, min_periods=1).mean()
    f["rolling_mean_12"] = df["units_sold"].shift(1).rolling(12, min_periods=1).mean()
    f["rolling_std_4"] = df["units_sold"].shift(1).rolling(4, min_periods=1).std().fillna(0)
    f["ili_pct"] = df.get("ili_pct", 0)
    f["vbp_10_active"] = df.get("vbp_10_active", 0)
    f["is_holiday"] = df.get("is_holiday", 0)
    f["week_of_year"] = pd.to_datetime(df["date"]).dt.isocalendar().week.values
    month = pd.to_datetime(df["date"]).dt.month
    for m in range(1, 13):
        f[f"month_{m}"] = (month == m).astype(int)
    f["vbp_selected"] = 1 if sku_info.get("vbp_status") == "selected" else 0
    f["vbp_nonselected"] = 1 if sku_info.get("vbp_status") == "non_selected" else 0
    ptype = sku_info.get("pharmacy_type", "chain")
    for pt in ["hospital", "chain", "independent"]:
        f[f"ptype_{pt}"] = 1 if ptype == pt else 0
    f["vbp_selected_x_pharmacy"] = f["vbp_selected"] * f[f"ptype_{ptype}"]
    atc4 = sku_info.get("atc4", "")
    n_sel = master_df[(master_df["atc4"] == atc4) & (master_df["vbp_status"] == "selected")].shape[0]
    f["n_selected_atc4"] = n_sel
    f["target"] = df["units_sold"]
    return f


# ---------------------------------------------------------------------------
# Main Demo Class
# ---------------------------------------------------------------------------
class Day7Demo:
    def __init__(self, cfg: Day7Config | None = None):
        self.cfg = cfg or Day7Config()
        np.random.seed(self.cfg.random_seed)

        self.sales: pd.DataFrame | None = None
        self.master: pd.DataFrame | None = None
        self.signals: pd.DataFrame | None = None
        self.tuning_results: Dict[str, Dict] = {}
        self.walkforward_df: pd.DataFrame | None = None
        self.best_models: Dict[str, XGBRegressor] = {}
        self.inventory_health: pd.DataFrame | None = None

    # =====================================================================
    # Data Loading
    # =====================================================================
    def load_data(self) -> None:
        self.sales = pd.read_csv(self.cfg.data_dir / "sales_data.csv", parse_dates=["date"])
        self.master = pd.read_csv(self.cfg.data_dir / "master_data.csv")
        self.signals = pd.read_csv(self.cfg.data_dir / "external_signals.csv", parse_dates=["date"])
        if "cold_chain" not in self.master.columns:
            self.master["cold_chain"] = 0
        print(f"[Load] Sales: {len(self.sales):,} rows | Master: {len(self.master)} SKUs")

    # =====================================================================
    # 1. XGBoost Hyperparameter Tuning for Demo SKUs
    # =====================================================================
    def tune_xgboost(self) -> Dict[str, Dict]:
        print("\n[1/5] XGBoost Hyperparameter Tuning for Demo SKUs...")
        results = {}

        for sku_id in self.cfg.demo_skus:
            sku_info = self.master[self.master["sku_id"] == sku_id].iloc[0].to_dict()
            best_mape = float("inf")
            best_params = None
            best_model = None
            best_importance = None

            for ptype in ["hospital", "chain", "independent"]:
                group = self.sales[(self.sales["sku_id"] == sku_id) & (self.sales["pharmacy_type"] == ptype)]
                group = group.sort_values("date").reset_index(drop=True)
                if len(group) < 30:
                    continue

                # Merge external signals
                group = group.merge(self.signals[["date", "ili_pct", "vbp_10_active", "is_holiday", "week_of_year"]], on="date", how="left")
                sku_info["pharmacy_type"] = ptype

                feat_df = build_features(group, sku_info, self.master)
                feature_cols = [c for c in feat_df.columns if c not in ("date", "target")]

                # Train/val split: first 80% train, last 20% validation
                split = int(len(feat_df) * 0.8)
                train_df, val_df = feat_df.iloc[:split], feat_df.iloc[split:]
                if len(val_df) < 5:
                    continue

                X_train, y_train = train_df[feature_cols], train_df["target"]
                X_val, y_val = val_df[feature_cols], val_df["target"]

                # Small grid search
                candidates = [
                    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
                    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8},
                    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9},
                    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9},
                ]

                for params in candidates:
                    model = XGBRegressor(
                        **params, random_state=self.cfg.random_seed,
                        objective="reg:squarederror", n_jobs=-1,
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    mape = self._mape(y_val.values, preds)
                    if mape < best_mape:
                        best_mape = mape
                        best_params = params.copy()
                        best_params["pharmacy_type"] = ptype
                        best_model = model
                        best_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

            results[sku_id] = {
                "sku_id": sku_id,
                "name_cn": sku_info["name_cn"],
                "best_params": best_params,
                "best_mape": round(best_mape, 2),
                "feature_importance": dict(sorted(best_importance.items(), key=lambda x: x[1], reverse=True)[:10]) if best_importance else {},
            }
            self.best_models[sku_id] = best_model
            print(f"  {sku_id} ({sku_info['name_cn']}): best MAPE={best_mape:.2f}%, params={best_params}")

        self.tuning_results = results
        return results

    @staticmethod
    def _mape(actual, predicted):
        mask = actual != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    # =====================================================================
    # 2. Feature Importance Bar Chart
    # =====================================================================
    def plot_feature_importance(self) -> None:
        if not self.tuning_results:
            return

        n_skus = len(self.tuning_results)
        fig, axes = plt.subplots(1, n_skus, figsize=(7 * n_skus, 5))
        if n_skus == 1:
            axes = [axes]

        for ax, (sku_id, res) in zip(axes, self.tuning_results.items()):
            imp = res.get("feature_importance", {})
            if not imp:
                continue
            df_imp = pd.DataFrame([{"feature": k, "importance": v} for k, v in imp.items()])
            df_imp = df_imp.sort_values("importance", ascending=True)
            ax.barh(df_imp["feature"], df_imp["importance"], color="steelblue")
            ax.set_title(f"{sku_id} — {res['name_cn']}", fontweight="bold")
            ax.set_xlabel("Importance")

        plt.tight_layout()
        out_path = self.cfg.figure_dir / "feature_importance.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved {out_path}")

    # =====================================================================
    # 3. 12-Week Walk-Forward Backtest with Confidence Bands
    # =====================================================================
    def walk_forward_backtest(self, forecast_horizon: int = 4, n_windows: int = 3) -> pd.DataFrame:
        """
        Expanding-window walk-forward:
        - Train on all data up to window start
        - Predict next `forecast_horizon` weeks
        - Shift window by forecast_horizon, repeat n_windows
        - Covers last 12 weeks of data
        """
        print(f"\n[2/5] Walk-Forward Backtest ({n_windows}×{forecast_horizon} weeks)...")
        all_records = []
        all_residuals = []

        for sku_id in self.cfg.demo_skus:
            sku_info = self.master[self.master["sku_id"] == sku_id].iloc[0].to_dict()
            for ptype in ["chain"]:  # Focus on chain pharmacy for demo visual
                group = self.sales[(self.sales["sku_id"] == sku_id) & (self.sales["pharmacy_type"] == ptype)]
                group = group.sort_values("date").reset_index(drop=True)
                group = group.merge(self.signals[["date", "ili_pct", "vbp_10_active", "is_holiday", "week_of_year"]], on="date", how="left")

                n_total = len(group)
                # Last 12 weeks = test region
                test_start = n_total - (forecast_horizon * n_windows)

                for w in range(n_windows):
                    train_end = test_start + w * forecast_horizon
                    test_end = train_end + forecast_horizon

                    train = group.iloc[:train_end]
                    test = group.iloc[train_end:test_end]
                    if len(test) == 0:
                        continue

                    sku_info["pharmacy_type"] = ptype
                    train_feat = build_features(train, sku_info, self.master)
                    test_feat = build_features(test, sku_info, self.master)
                    feature_cols = [c for c in train_feat.columns if c not in ("date", "target")]

                    # Use best params from tuning
                    best_params = self.tuning_results.get(sku_id, {}).get("best_params", {})
                    model = XGBRegressor(
                        n_estimators=best_params.get("n_estimators", 200),
                        max_depth=best_params.get("max_depth", 5),
                        learning_rate=best_params.get("learning_rate", 0.1),
                        subsample=best_params.get("subsample", 0.8),
                        random_state=self.cfg.random_seed,
                        objective="reg:squarederror",
                        n_jobs=-1,
                    )
                    model.fit(train_feat[feature_cols], train_feat["target"])
                    preds = model.predict(test_feat[feature_cols])

                    actuals = test_feat["target"].values
                    residuals = actuals - preds
                    all_residuals.extend(residuals.tolist())

                    for i in range(len(test)):
                        all_records.append({
                            "sku_id": sku_id,
                            "name_cn": sku_info["name_cn"],
                            "pharmacy_type": ptype,
                            "date": test_feat["date"].iloc[i],
                            "actual": actuals[i],
                            "forecast": preds[i],
                        })

        df = pd.DataFrame(all_records)
        residual_std = np.std(all_residuals) if all_residuals else 1.0
        df["upper_90"] = df["forecast"] + 1.645 * residual_std
        df["lower_90"] = df["forecast"] - 1.645 * residual_std
        df["upper_90"] = df["upper_90"].clip(lower=0)
        df["lower_90"] = df["lower_90"].clip(lower=0)

        self.walkforward_df = df
        print(f"  Records: {len(df)}, Residual std: {residual_std:.2f}")
        return df

    def plot_walkforward(self) -> None:
        if self.walkforward_df is None or len(self.walkforward_df) == 0:
            return

        sku_ids = self.walkforward_df["sku_id"].unique()
        fig, axes = plt.subplots(len(sku_ids), 1, figsize=(14, 4 * len(sku_ids)), sharex=True)
        if len(sku_ids) == 1:
            axes = [axes]

        for ax, sku_id in zip(axes, sku_ids):
            sub = self.walkforward_df[self.walkforward_df["sku_id"] == sku_id].sort_values("date")
            name = sub["name_cn"].iloc[0]

            ax.plot(sub["date"], sub["actual"], label="Actual", color="black", linewidth=1.5)
            ax.plot(sub["date"], sub["forecast"], label="Forecast", color="steelblue", linewidth=1.5)
            ax.fill_between(sub["date"], sub["lower_90"], sub["upper_90"],
                            alpha=0.25, color="steelblue", label="90% Confidence Band")
            ax.set_title(f"{sku_id} | {name} | 12-Week Walk-Forward Backtest", fontweight="bold")
            ax.set_ylabel("Units Sold")
            ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        out_path = self.cfg.figure_dir / "walkforward_12week.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved {out_path}")

    # =====================================================================
    # 4. Inventory Health Diagnosis
    # =====================================================================
    def inventory_health_diagnosis(self) -> pd.DataFrame:
        print("\n[3/5] Inventory Health Diagnosis...")
        # Load replenishment recommendations for current stock / expiry data
        rec_path = self.cfg.report_dir / "replenishment_recommendations.csv"
        if not rec_path.exists():
            print("  [Warning] Run replenishment_engine first for inventory data.")
            return pd.DataFrame()

        recs = pd.read_csv(rec_path)

        # Expiry risk score: (stock_weeks_of_supply) / remaining_weeks
        # But we need recent_avg and remaining_weeks
        # Build from sales + master
        health_records = []
        for _, row in recs.iterrows():
            sku_id = row["sku_id"]
            ptype = row["pharmacy_type"]
            stock = row["current_stock"]
            recent_avg = row.get("recent_avg_demand", 10)
            remaining_w = row.get("expiry_weeks", 999)

            if remaining_w <= 0:
                score = 1.0
            else:
                weeks_supply = stock / max(recent_avg, 0.1)
                score = min(1.0, weeks_supply / remaining_w)

            if score > 0.7:
                expiry_risk = "Red"
            elif score >= 0.3:
                expiry_risk = "Yellow"
            else:
                expiry_risk = "Green"

            health_records.append({
                "sku_id": sku_id,
                "name_cn": row["name_cn"],
                "pharmacy_type": ptype,
                "abc_class": row["abc_class"],
                "xyz_class": row["xyz_class"],
                "current_stock": stock,
                "days_supply": row.get("days_supply", 0),
                "expiry_weeks": remaining_w,
                "expiry_score": round(score, 3),
                "expiry_risk": expiry_risk,
                "risk_level": row["risk_level"],
            })

        self.inventory_health = pd.DataFrame(health_records)
        print(f"  Diagnosed {len(self.inventory_health)} SKU-pharmacy combinations.")
        return self.inventory_health

    def plot_expiry_heatmap(self) -> None:
        if self.inventory_health is None or len(self.inventory_health) == 0:
            return

        df = self.inventory_health.copy()
        pivot = df.pivot_table(index="sku_id", columns="pharmacy_type",
                               values="expiry_score", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.3)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r",
                    vmin=0, vmax=1, cbar_kws={"label": "Expiry Risk Score"}, ax=ax)
        ax.set_title("Expiry Risk Heatmap (Score: 0=Green, 1=Red)", fontweight="bold")
        plt.tight_layout()
        out_path = self.cfg.figure_dir / "expiry_risk_heatmap.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved {out_path}")

    def plot_dashboard(self) -> None:
        if self.inventory_health is None:
            return

        df = self.inventory_health
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Inventory Health Dashboard", fontsize=16, fontweight="bold")

        # 1. ABC distribution
        abc_counts = df["abc_class"].value_counts().reindex(["A", "B", "C"], fill_value=0)
        axes[0, 0].pie(abc_counts, labels=[f"{k} ({v})" for k, v in abc_counts.items()],
                       colors=["#e74c3c", "#f39c12", "#2ecc71"], autopct="%1.1f%%")
        axes[0, 0].set_title("ABC Distribution (by Revenue)")

        # 2. XYZ distribution
        xyz_counts = df["xyz_class"].value_counts().reindex(["X", "Y", "Z"], fill_value=0)
        axes[0, 1].pie(xyz_counts, labels=[f"{k} ({v})" for k, v in xyz_counts.items()],
                       colors=["#2ecc71", "#f39c12", "#e74c3c"], autopct="%1.1f%%")
        axes[0, 1].set_title("XYZ Distribution (by Volatility)")

        # 3. Expiry risk distribution
        exp_counts = df["expiry_risk"].value_counts().reindex(["Red", "Yellow", "Green"], fill_value=0)
        axes[1, 0].bar(exp_counts.index, exp_counts.values, color=["#e74c3c", "#f39c12", "#2ecc71"])
        axes[1, 0].set_title("Expiry Risk Distribution")
        axes[1, 0].set_ylabel("Count")

        # 4. Key metrics text
        red_risk = (df["risk_level"] == "Red").sum()
        yellow_risk = (df["risk_level"] == "Yellow").sum()
        green_risk = (df["risk_level"] == "Green").sum()
        axes[1, 1].text(0.1, 0.8, "Key Metrics", fontsize=14, fontweight="bold", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.65, f"Total SKUs: {df['sku_id'].nunique()}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.55, f"Red Risk: {red_risk}", color="red", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.45, f"Yellow Risk: {yellow_risk}", color="orange", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.35, f"Green Risk: {green_risk}", color="green", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f"Avg Expiry Score: {df['expiry_score'].mean():.2f}", transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Summary KPIs")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = self.cfg.figure_dir / "inventory_dashboard.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved {out_path}")

    # =====================================================================
    # 5. Summary Report
    # =====================================================================
    def generate_summary_report(self) -> None:
        print("\n[4/5] Generating day7_summary.md...")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# Day 7 Summary Report — SKU-Level Demo Preparation",
            f"\n**Generated:** {now}",
            "\n---\n",
            "## 1. XGBoost Hyperparameter Tuning\n",
        ]

        for sku_id, res in self.tuning_results.items():
            lines.append(f"### {sku_id} — {res['name_cn']}")
            params = res.get("best_params", {})
            lines.append(f"- **Best Params:** n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')}, learning_rate={params.get('learning_rate')}, subsample={params.get('subsample')}")
            lines.append(f"- **Best Validation MAPE:** {res['best_mape']:.2f}%")
            lines.append(f"- **Pharmacy Type Tuned:** {params.get('pharmacy_type', 'N/A')}")
            lines.append("")

        # Walk-forward metrics
        lines.append("\n## 2. 12-Week Walk-Forward Backtest\n")
        if self.walkforward_df is not None and len(self.walkforward_df) > 0:
            for sku_id in self.walkforward_df["sku_id"].unique():
                sub = self.walkforward_df[self.walkforward_df["sku_id"] == sku_id]
                actual = sub["actual"].values
                forecast = sub["forecast"].values
                mape = self._mape(actual, forecast)
                smape = self._smape(actual, forecast)
                mase = self._mase(actual, forecast)
                lines.append(f"### {sku_id} — {sub['name_cn'].iloc[0]}")
                lines.append(f"- **MAPE:** {mape:.2f}%")
                lines.append(f"- **SMAPE:** {smape:.2f}%")
                lines.append(f"- **MASE:** {mase:.2f}")
                lines.append(f"- **Records:** {len(sub)}")
                lines.append("")

        # Inventory health
        lines.append("\n## 3. Inventory Health Diagnosis\n")
        if self.inventory_health is not None:
            exp = self.inventory_health["expiry_risk"].value_counts().reindex(["Red", "Yellow", "Green"], fill_value=0)
            lines.append(f"- **Red expiry risk:** {exp.get('Red', 0)} SKUs")
            lines.append(f"- **Yellow expiry risk:** {exp.get('Yellow', 0)} SKUs")
            lines.append(f"- **Green expiry risk:** {exp.get('Green', 0)} SKUs")
            lines.append(f"- **Average expiry score:** {self.inventory_health['expiry_score'].mean():.3f}")
            lines.append("")

        lines.append("\n## 4. Conclusion\n")
        lines.append("- Demo SKU XGBoost models tuned with small grid search.")
        lines.append("- Walk-forward backtest covers last 12 weeks with 90% confidence bands.")
        lines.append("- Inventory health dashboard highlights expiry risks and ABC/XYZ segmentation.")
        lines.append("")

        path = self.cfg.report_dir / "day7_summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Report] Saved to {path}")

    @staticmethod
    def _smape(actual, predicted):
        denom = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denom != 0
        return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)

    @staticmethod
    def _mase(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        naive = np.mean(np.abs(actual[1:] - actual[:-1])) if len(actual) > 1 else mae
        return float(mae / naive) if naive > 0 else float("inf")

    # =====================================================================
    # Main Runner
    # =====================================================================
    def run(self):
        print("=" * 60)
        print("Day 7 — SKU-Level Demo Preparation & Model Tuning")
        print("=" * 60)

        self.load_data()
        self.tune_xgboost()
        self.plot_feature_importance()
        self.walk_forward_backtest()
        self.plot_walkforward()
        self.inventory_health_diagnosis()
        self.plot_expiry_heatmap()
        self.plot_dashboard()
        self.generate_summary_report()

        print()
        print("=" * 60)
        print("Day 7 complete! Check outputs/figures/ and reports/day7_summary.md")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    cfg = Day7Config()
    demo = Day7Demo(cfg)
    demo.run()


if __name__ == "__main__":
    main()
