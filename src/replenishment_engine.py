"""
Intelligent Replenishment Engine — Day 6
=========================================
Translates demand forecasts into actionable restock recommendations.

Features:
  • Periodic review (R, S) policy with pharmacy-type customization
  • Differentiated safety stock: classic normal (smooth) vs Croston-variance (intermittent)
    vs prediction-interval width (demo SKUs)
  • FEFO expiry-aware constraints
  • Risk triage: Red / Yellow / Green
  • ABC/XYZ inventory health matrix
  • Inter-pharmacy transfer recommendations
  • Policy simulation: engine vs fixed-threshold baseline
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
class ReplenishmentConfig:
    data_dir: Path = Path("data/raw")
    report_dir: Path = Path("reports")
    chart_dir: Path = Path("docs/validation")
    forecast_dir: Path = Path("reports")
    lead_time_weeks: int = 1  # L = 1 week (configurable)
    random_seed: int = 42

    # Service level Z
    z_values: Dict[str, float] = None  # set in __post_init__
    # Review period R (weeks)
    r_values: Dict[str, int] = None    # set in __post_init__

    def __post_init__(self):
        object.__setattr__(self, "z_values", {"hospital": 2.33, "chain": 1.65, "independent": 1.65})
        object.__setattr__(self, "r_values", {"hospital": 1, "chain": 1, "independent": 2})


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------
class ReplenishmentEngine:
    def __init__(self, cfg: ReplenishmentConfig | None = None):
        self.cfg = cfg or ReplenishmentConfig()
        self.cfg.report_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.chart_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.sales: pd.DataFrame | None = None
        self.master: pd.DataFrame | None = None
        self.forecasts: Dict[str, pd.DataFrame] = {}
        self.inventory: pd.DataFrame | None = None
        self.recommendations: pd.DataFrame | None = None
        self.transfer_recs: pd.DataFrame | None = None
        self.simulation_results: pd.DataFrame | None = None

        np.random.seed(self.cfg.random_seed)

    # =====================================================================
    # Data Loading
    # =====================================================================
    def load_data(self) -> None:
        self.sales = pd.read_csv(self.cfg.data_dir / "sales_data.csv", parse_dates=["date"])
        self.master = pd.read_csv(self.cfg.data_dir / "master_data.csv")

        # Load forecast details for demo SKUs (to extract prediction intervals)
        for f in self.cfg.forecast_dir.glob("forecast_detail_*.csv"):
            df = pd.read_csv(f, parse_dates=["date"])
            key = f"{df['sku_id'].iloc[0]}_{df['pharmacy_type'].iloc[0]}"
            self.forecasts[key] = df

        # Compatibility: Day-4 data may lack cold_chain column
        if "cold_chain" not in self.master.columns:
            self.master["cold_chain"] = 0

        print(f"[Load] Sales: {len(self.sales):,} rows | Master: {len(self.master)} SKUs | Forecasts: {len(self.forecasts)}")

    # =====================================================================
    # Simulate Current Inventory
    # =====================================================================
    def simulate_current_inventory(self) -> pd.DataFrame:
        """Generate realistic current stock levels based on recent sales."""
        records = []
        for (sku_id, ptype), g in self.sales.groupby(["sku_id", "pharmacy_type"]):
            g = g.sort_values("date")
            sku_info = self.master[self.master["sku_id"] == sku_id].iloc[0]

            recent_avg = g["units_sold"].tail(4).mean()
            demand_cv = sku_info[f"cv_{ptype}"]
            r = self._get_review_period(ptype, sku_info["cold_chain"])
            l = self.cfg.lead_time_weeks

            # Target coverage = (R+L) × 1.2, with random perturbation
            coverage = (r + l) * 1.2
            noise = np.random.uniform(0.3, 1.8)  # some under, some over
            current_stock = max(0, int(recent_avg * coverage * noise))

            # Simulate batch age: 0 ~ expiry_months (in weeks)
            expiry_weeks = sku_info["expiry_months"] * 4
            weeks_in_stock = int(np.random.uniform(0, expiry_weeks * 0.7))
            remaining_weeks = max(0, expiry_weeks - weeks_in_stock)

            records.append({
                "sku_id": sku_id,
                "pharmacy_type": ptype,
                "name_cn": sku_info["name_cn"],
                "demand_type": sku_info["demand_type"],
                "cold_chain": sku_info["cold_chain"],
                "current_stock": current_stock,
                "remaining_weeks": remaining_weeks,
                "expiry_months": sku_info["expiry_months"],
                "price_rmb": sku_info["price_rmb"],
                "vbp_status": sku_info["vbp_status"],
                "is_demo_sku": sku_info["is_demo_sku"],
                "recent_avg_demand": recent_avg,
                "demand_cv": demand_cv,
                "atc4": sku_info["atc4"],
            })

        self.inventory = pd.DataFrame(records)
        print(f"[Inventory] Simulated current stock for {len(self.inventory)} SKU-pharmacy combinations.")
        return self.inventory

    def _get_review_period(self, pharmacy_type: str, cold_chain: int) -> int:
        if cold_chain == 1:
            return 1
        return self.cfg.r_values.get(pharmacy_type, 1)

    def _get_service_level_z(self, pharmacy_type: str) -> float:
        return self.cfg.z_values.get(pharmacy_type, 1.65)

    # =====================================================================
    # ABC / XYZ Classification
    # =====================================================================
    def classify_abc_xyz(self) -> pd.DataFrame:
        """ABC by cumulative sales (80/15/5), XYZ by demand CV."""
        if self.inventory is None:
            self.simulate_current_inventory()

        # Compute total historical sales per SKU (across all pharmacy types)
        total_sales = self.sales.groupby("sku_id")["units_sold"].sum().reset_index()
        total_sales = total_sales.rename(columns={"units_sold": "total_sales"})
        total_sales = total_sales.sort_values("total_sales", ascending=False)
        total_sales["cum_pct"] = total_sales["total_sales"].cumsum() / total_sales["total_sales"].sum() * 100

        def abc_label(pct):
            if pct <= 80:
                return "A"
            elif pct <= 95:
                return "B"
            return "C"

        total_sales["abc_class"] = total_sales["cum_pct"].apply(abc_label)

        # XYZ by CV (use max CV across pharmacy types as proxy)
        cv_df = self.inventory.groupby("sku_id")["demand_cv"].max().reset_index()

        def xyz_label(cv):
            if cv < 0.5:
                return "X"
            elif cv <= 1.0:
                return "Y"
            return "Z"

        cv_df["xyz_class"] = cv_df["demand_cv"].apply(xyz_label)

        abc_xyz = total_sales.merge(cv_df, on="sku_id", how="left")
        abc_xyz = abc_xyz.merge(self.master[["sku_id", "name_cn"]], on="sku_id", how="left")

        # Merge back to inventory
        self.inventory = self.inventory.merge(
            abc_xyz[["sku_id", "abc_class", "xyz_class"]], on="sku_id", how="left"
        )
        print("[ABC/XYZ] Classification complete.")
        return self.inventory

    # =====================================================================
    # Safety Stock
    # =====================================================================
    def calculate_safety_stock(self, sku_id: str, ptype: str,
                               forecast_df: pd.DataFrame | None) -> Tuple[float, str]:
        """Return (safety_stock, method_used)."""
        sku_info = self.master[self.master["sku_id"] == sku_id].iloc[0]
        z = self._get_service_level_z(ptype)
        l = self.cfg.lead_time_weeks
        key = f"{sku_id}_{ptype}"

        # Demo SKU: try prediction interval width first
        if sku_info["is_demo_sku"] == 1 and forecast_df is not None and "upper_90" in forecast_df.columns:
            pi_width = (forecast_df["upper_90"] - forecast_df["lower_90"]).mean()
            ss = pi_width / 2 * z / 1.645  # scale to target Z
            return max(ss, 0), "PI-width"

        # Intermittent / Lumpy: Croston-based variance approximation
        adi = self._compute_adi_for_sku(sku_id, ptype)
        if adi >= 1.32:
            # Approximate variance from Croston/SBA forecast error proxy
            cv = sku_info[f"cv_{ptype}"]
            avg_demand = sku_info[f"base_demand_{ptype}"]
            # Negative-binomial variance ≈ mean × (1 + dispersion)
            # Simplified: use gamma distribution approximation for lead-time demand
            sigma_lt = avg_demand * cv * np.sqrt(l)
            ss = z * sigma_lt
            return max(ss, 0), "Croston-proxy"

        # Smooth: classic normal lead-time demand
        cv = sku_info[f"cv_{ptype}"]
        avg_demand = sku_info[f"base_demand_{ptype}"]
        sigma_lt = avg_demand * cv * np.sqrt(l)
        ss = z * sigma_lt
        return max(ss, 0), "Classic-normal"

    def _compute_adi_for_sku(self, sku_id: str, ptype: str) -> float:
        sub = self.sales[(self.sales["sku_id"] == sku_id) & (self.sales["pharmacy_type"] == ptype)]["units_sold"]
        positive = np.where(sub.values > 0)[0]
        if len(positive) <= 1:
            return float(len(sub))
        return float(np.mean(np.diff(positive)))

    # =====================================================================
    # Main Replenishment Logic
    # =====================================================================
    def generate_recommendations(self) -> pd.DataFrame:
        if self.inventory is None:
            self.simulate_current_inventory()
        if "abc_class" not in self.inventory.columns:
            self.classify_abc_xyz()

        recs = []
        for _, inv in self.inventory.iterrows():
            sku_id = inv["sku_id"]
            ptype = inv["pharmacy_type"]
            key = f"{sku_id}_{ptype}"

            # Load forecast if available
            forecast_df = self.forecasts.get(key)

            # Parameters
            r = self._get_review_period(ptype, inv["cold_chain"])
            l = self.cfg.lead_time_weeks
            coverage_weeks = r + l

            # Forecast demand over coverage period
            if forecast_df is not None and len(forecast_df) >= coverage_weeks:
                forecast_demand = forecast_df["forecast"].iloc[:coverage_weeks].sum()
                forecast_series = forecast_df["forecast"].iloc[:coverage_weeks]
            else:
                # Fallback: use recent average × coverage
                forecast_demand = inv["recent_avg_demand"] * coverage_weeks
                forecast_series = pd.Series([inv["recent_avg_demand"]] * coverage_weeks)

            # Safety stock
            ss, ss_method = self.calculate_safety_stock(sku_id, ptype, forecast_df)

            # Order-up-to level S
            s_level = forecast_demand + ss

            # Recommended order quantity
            current = inv["current_stock"]
            rec_qty = max(0, int(s_level - current))

            # Days of supply (approx: weeks × 7)
            daily_avg = inv["recent_avg_demand"] / 7
            days_supply = int(current / max(daily_avg, 0.1))

            # FEFO expiry constraint
            expiry_flag = False
            if inv["remaining_weeks"] < 12:  # < 3 months
                expiry_flag = True
                if current > forecast_demand * 1.5:
                    rec_qty = 0  # overstock + near expiry = don't order
            elif inv["remaining_weeks"] < 24:  # < 6 months
                expiry_flag = True
                if current > forecast_demand:
                    rec_qty = max(0, int(rec_qty * 0.3))  # severely reduce order

            # Risk level
            risk, risk_reason = self._determine_risk(days_supply, inv["remaining_weeks"],
                                                      current, s_level, rec_qty, inv["demand_type"])

            # Recommended order date (next review cycle)
            order_date = (datetime.now() + timedelta(weeks=r)).strftime("%Y-%m-%d")

            # Reason generation
            reasons = []
            if days_supply < 5:
                reasons.append("库存告急")
            if rec_qty > 0 and forecast_series.iloc[-1] > forecast_series.iloc[0] * 1.2:
                reasons.append("预测需求上升")
            if inv["vbp_status"] == "selected":
                reasons.append("VBP中选品种需保障供应")
            if inv["remaining_weeks"] < 24:
                reasons.append("效期临近，谨慎补货")
            if current > s_level * 1.5:
                reasons.append("库存积压")
            if not reasons:
                reasons.append("常规补货")

            recs.append({
                "sku_id": sku_id,
                "name_cn": inv["name_cn"],
                "pharmacy_type": ptype,
                "current_stock": current,
                "days_supply": days_supply,
                "safety_stock": round(ss, 1),
                "ss_method": ss_method,
                "target_level_S": round(s_level, 1),
                "recommended_qty": rec_qty,
                "order_date": order_date,
                "risk_level": risk,
                "risk_reason": risk_reason,
                "reason": "；".join(reasons),
                "abc_class": inv["abc_class"],
                "xyz_class": inv["xyz_class"],
                "vbp_status": inv["vbp_status"],
                "expiry_weeks": inv["remaining_weeks"],
                "cold_chain": inv["cold_chain"],
            })

        self.recommendations = pd.DataFrame(recs)
        print(f"[Recommendations] Generated {len(self.recommendations)} lines.")
        return self.recommendations

    def _determine_risk(self, days_supply: int, remaining_weeks: int,
                        current: int, s_level: float, rec_qty: int, d_type: str) -> Tuple[str, str]:
        if days_supply < 5:
            return "Red", "可售天数不足5天"
        if remaining_weeks < 12 and current > 0:
            return "Red", "效期不足3个月且仍有库存"
        if current > s_level * 1.5:
            return "Yellow", "库存超过目标水位150%"
        if remaining_weeks < 24 and rec_qty > 0:
            return "Yellow", "效期不足6个月"
        if days_supply > 60 and d_type == "intermittent":
            return "Yellow", "慢动SKU积压"
        return "Green", "库存健康"

    # =====================================================================
    # Transfer Recommendations
    # =====================================================================
    def generate_transfer_recommendations(self) -> pd.DataFrame:
        if self.recommendations is None:
            self.generate_recommendations()

        transfers = []
        # For each SKU, find overstocked and understocked pharmacies
        for sku_id in self.recommendations["sku_id"].unique():
            sku_recs = self.recommendations[self.recommendations["sku_id"] == sku_id]
            over = sku_recs[sku_recs["risk_reason"].str.contains("超过目标")]
            under = sku_recs[sku_recs["risk_level"] == "Red"]

            for _, o in over.iterrows():
                for _, u in under.iterrows():
                    if o["pharmacy_type"] == u["pharmacy_type"]:
                        continue
                    excess = int(o["current_stock"] - o["target_level_S"] * 1.2)
                    deficit = int(u["target_level_S"] - u["current_stock"])
                    qty = max(0, min(excess, deficit))
                    if qty > 0:
                        transfers.append({
                            "sku_id": sku_id,
                            "name_cn": o["name_cn"],
                            "from_pharmacy": o["pharmacy_type"],
                            "to_pharmacy": u["pharmacy_type"],
                            "transfer_qty": qty,
                            "from_stock": o["current_stock"],
                            "to_stock": u["current_stock"],
                            "reason": f"{o['pharmacy_type']}过剩({excess})→{u['pharmacy_type']}缺货({deficit})"
                        })

        self.transfer_recs = pd.DataFrame(transfers)
        print(f"[Transfer] {len(self.transfer_recs)} transfer recommendations.")
        return self.transfer_recs

    # =====================================================================
    # Policy Simulation (Engine vs Fixed Threshold)
    # =====================================================================
    def run_policy_simulation(self, sim_weeks: int = 13) -> pd.DataFrame:
        """Simulate 90 days (~13 weeks) comparing engine vs fixed threshold."""
        results = []

        for (sku_id, ptype), group in self.sales.groupby(["sku_id", "pharmacy_type"]):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) < sim_weeks + 4:
                continue

            # Use last sim_weeks as simulation horizon
            sim_demand = group["units_sold"].tail(sim_weeks).values
            hist_demand = group["units_sold"].tail(sim_weeks + 4).head(4)
            avg_demand = hist_demand.mean()

            # Fixed threshold policy: reorder when stock < avg_demand, up to avg_demand×2
            fixed_threshold = avg_demand
            fixed_order_up_to = avg_demand * 2

            # Engine policy parameters
            inv_row = self.inventory[
                (self.inventory["sku_id"] == sku_id) & (self.inventory["pharmacy_type"] == ptype)
            ]
            if len(inv_row) == 0:
                continue
            inv_row = inv_row.iloc[0]
            r = self._get_review_period(ptype, inv_row["cold_chain"])
            l = self.cfg.lead_time_weeks
            key = f"{sku_id}_{ptype}"
            forecast_df = self.forecasts.get(key)

            # Simulate both policies
            initial_stock = int(inv_row["current_stock"])
            sim_fixed = self._simulate_one_policy(sim_demand, fixed_threshold, fixed_order_up_to, r, l, initial_stock)
            sim_engine = self._simulate_one_policy_engine(sim_demand, sku_id, ptype, forecast_df, r, l, initial_stock)

            results.append({
                "sku_id": sku_id,
                "pharmacy_type": ptype,
                "name_cn": inv_row["name_cn"],
                "fixed_stockouts": sim_fixed["stockouts"],
                "fixed_avg_inventory": round(sim_fixed["avg_inv"], 1),
                "engine_stockouts": sim_engine["stockouts"],
                "engine_avg_inventory": round(sim_engine["avg_inv"], 1),
                "demand_total": int(sim_demand.sum()),
            })

        self.simulation_results = pd.DataFrame(results)
        print(f"[Simulation] Completed for {len(self.simulation_results)} SKU-pharmacy pairs.")
        return self.simulation_results

    def _simulate_one_policy(self, demand: np.ndarray, threshold: float, order_up_to: float,
                             review_period: int, lead_time: int, initial_stock: int) -> Dict:
        stock = initial_stock  # same starting point as engine
        inv_levels = [stock]
        stockouts = 0
        pending_orders = []  # (arrival_week, qty)

        for week, d in enumerate(demand):
            # Receive pending orders that arrive this week
            arrived = [q for w, q in pending_orders if w == week]
            stock += sum(arrived)
            pending_orders = [(w, q) for w, q in pending_orders if w > week]

            # Consume demand
            if stock >= d:
                stock -= d
            else:
                stockouts += 1
                stock = 0

            # Review and order
            if week % review_period == 0:
                if stock < threshold:
                    qty = int(order_up_to - stock)
                    if qty > 0:
                        pending_orders.append((week + lead_time, qty))

            inv_levels.append(stock)

        return {"stockouts": stockouts, "avg_inv": np.mean(inv_levels)}

    def _simulate_one_policy_engine(self, demand: np.ndarray, sku_id: str, ptype: str,
                                    forecast_df: pd.DataFrame | None, review_period: int,
                                    lead_time: int, initial_stock: int) -> Dict:
        stock = initial_stock  # same starting point as fixed policy
        inv_levels = [stock]
        stockouts = 0
        pending_orders = []
        coverage = review_period + lead_time

        # Re-fetch inv_row for recent_avg_demand fallback
        inv_rows = self.inventory[
            (self.inventory["sku_id"] == sku_id) & (self.inventory["pharmacy_type"] == ptype)
        ]
        recent_avg = inv_rows.iloc[0]["recent_avg_demand"] if len(inv_rows) > 0 else 10.0

        for week, d in enumerate(demand):
            # Receive pending orders that arrive this week
            arrived = [q for w, q in pending_orders if w == week]
            stock += sum(arrived)
            pending_orders = [(w, q) for w, q in pending_orders if w > week]

            if stock >= d:
                stock -= d
            else:
                stockouts += 1
                stock = 0

            if week % review_period == 0:
                # Forecast coverage period
                if forecast_df is not None and week + coverage <= len(forecast_df):
                    fc = forecast_df["forecast"].iloc[week:week+coverage].sum()
                else:
                    fc = recent_avg * coverage

                ss, _ = self.calculate_safety_stock(sku_id, ptype, forecast_df)
                s_level = fc + ss
                qty = int(max(0, s_level - stock))
                if qty > 0:
                    pending_orders.append((week + lead_time, qty))

            inv_levels.append(stock)

        return {"stockouts": stockouts, "avg_inv": np.mean(inv_levels)}

    # =====================================================================
    # Persistence
    # =====================================================================
    def save_reports(self) -> None:
        if self.recommendations is not None:
            self.recommendations.to_csv(
                self.cfg.report_dir / "replenishment_recommendations.csv",
                index=False, encoding="utf-8-sig"
            )
        if self.transfer_recs is not None:
            self.transfer_recs.to_csv(
                self.cfg.report_dir / "transfer_recommendations.csv",
                index=False, encoding="utf-8-sig"
            )
        if self.simulation_results is not None:
            self.simulation_results.to_csv(
                self.cfg.report_dir / "policy_simulation_results.csv",
                index=False, encoding="utf-8-sig"
            )
        print(f"[Save] Reports written to {self.cfg.report_dir}")

    # =====================================================================
    # Charts
    # =====================================================================
    def plot_inventory_health_matrix(self) -> None:
        if self.inventory is None or "abc_class" not in self.inventory.columns:
            return

        df = self.inventory.copy()
        df["xyz_numeric"] = df["xyz_class"].map({"X": 1, "Y": 2, "Z": 3})
        df["abc_numeric"] = df["abc_class"].map({"A": 1, "B": 2, "C": 3})

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = {"A": "#e74c3c", "B": "#f39c12", "C": "#2ecc71"}
        for abc, grp in df.groupby("abc_class"):
            ax.scatter(grp["abc_numeric"] + np.random.normal(0, 0.05, len(grp)),
                       grp["xyz_numeric"] + np.random.normal(0, 0.05, len(grp)),
                       c=colors[abc], label=f"ABC-{abc}", alpha=0.6, s=60)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["A (Top 80%)", "B (80-95%)", "C (Bottom 5%)"])
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["X (CV<0.5)", "Y (CV 0.5-1.0)", "Z (CV>1.0)"])
        ax.set_title("Inventory Health Matrix (ABC / XYZ)", fontsize=13, fontweight="bold")
        ax.legend(title="ABC Class")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.cfg.chart_dir / "09_inventory_health_matrix.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 09_inventory_health_matrix.png")

    def plot_replenishment_priority(self) -> None:
        if self.recommendations is None:
            return

        risk_counts = self.recommendations["risk_level"].value_counts().reindex(["Red", "Yellow", "Green"], fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        axes[0].bar(risk_counts.index, risk_counts.values, color=colors)
        axes[0].set_title("Replenishment Risk Distribution", fontweight="bold")
        axes[0].set_ylabel("SKU Count")

        # Top 10 recommended quantities
        top10 = self.recommendations.nlargest(10, "recommended_qty")[["name_cn", "pharmacy_type", "recommended_qty"]]
        axes[1].barh(range(len(top10)), top10["recommended_qty"], color="steelblue")
        axes[1].set_yticks(range(len(top10)))
        axes[1].set_yticklabels([f"{r['name_cn']} ({r['pharmacy_type']})" for _, r in top10.iterrows()], fontsize=8)
        axes[1].set_title("Top 10 Recommended Order Quantities", fontweight="bold")
        axes[1].set_xlabel("Units")

        plt.tight_layout()
        plt.savefig(self.cfg.chart_dir / "10_replenishment_priority.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 10_replenishment_priority.png")

    def plot_policy_simulation(self) -> None:
        if self.simulation_results is None:
            return

        sim = self.simulation_results
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        avg_fixed_stockouts = sim["fixed_stockouts"].mean()
        avg_engine_stockouts = sim["engine_stockouts"].mean()
        avg_fixed_inv = sim["fixed_avg_inventory"].mean()
        avg_engine_inv = sim["engine_avg_inventory"].mean()

        axes[0].bar(["Fixed Threshold", "Engine (R,S)"],
                    [avg_fixed_stockouts, avg_engine_stockouts],
                    color=["#e74c3c", "#3498db"])
        axes[0].set_title("Average Stockouts (90-day simulation)", fontweight="bold")
        axes[0].set_ylabel("Stockout Count")

        axes[1].bar(["Fixed Threshold", "Engine (R,S)"],
                    [avg_fixed_inv, avg_engine_inv],
                    color=["#e74c3c", "#3498db"])
        axes[1].set_title("Average Inventory Level", fontweight="bold")
        axes[1].set_ylabel("Units")

        plt.tight_layout()
        plt.savefig(self.cfg.chart_dir / "11_policy_simulation_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 11_policy_simulation_comparison.png")

    # =====================================================================
    # Auto Report
    # =====================================================================
    def generate_report(self) -> None:
        if self.recommendations is None:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# Intelligent Replenishment Report",
            f"\n**Generated:** {now}  ",
            f"**Lead Time:** {self.cfg.lead_time_weeks} week(s)  ",
            "\n---\n",
            "## Executive Summary\n",
        ]

        # Risk summary
        risk_sum = self.recommendations["risk_level"].value_counts().reindex(["Red", "Yellow", "Green"], fill_value=0)
        lines.append(f"- **Red (urgent):** {risk_sum.get('Red', 0)} SKUs")
        lines.append(f"- **Yellow (caution):** {risk_sum.get('Yellow', 0)} SKUs")
        lines.append(f"- **Green (healthy):** {risk_sum.get('Green', 0)} SKUs\n")

        # ABC/XYZ summary
        lines.append("## ABC / XYZ Distribution\n")
        abc_xyz = self.recommendations.groupby(["abc_class", "xyz_class"]).size().unstack(fill_value=0)
        lines.append(abc_xyz.to_markdown())
        lines.append("\n")

        # Top urgent replenishments
        lines.append("## Top 10 Urgent Replenishments (Red Risk)\n")
        urgent = self.recommendations[self.recommendations["risk_level"] == "Red"].nlargest(10, "recommended_qty")
        lines.append(urgent[["sku_id", "name_cn", "pharmacy_type", "current_stock", "days_supply", "recommended_qty", "reason"]].to_markdown(index=False))
        lines.append("\n")

        # Transfer recommendations
        if self.transfer_recs is not None and len(self.transfer_recs) > 0:
            lines.append("## Inter-Pharmacy Transfer Suggestions\n")
            lines.append(self.transfer_recs.head(10).to_markdown(index=False))
            lines.append("\n")

        # Simulation summary
        if self.simulation_results is not None:
            lines.append("## Policy Simulation (Engine vs Fixed Threshold, 90 days)\n")
            sim = self.simulation_results
            lines.append(f"- Fixed threshold avg stockouts: **{sim['fixed_stockouts'].mean():.2f}**")
            lines.append(f"- Engine (R,S) avg stockouts: **{sim['engine_stockouts'].mean():.2f}**")
            lines.append(f"- Fixed threshold avg inventory: **{sim['fixed_avg_inventory'].mean():.1f}**")
            lines.append(f"- Engine (R,S) avg inventory: **{sim['engine_avg_inventory'].mean():.1f}**\n")

        path = self.cfg.report_dir / "replenishment_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Report] Saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Intelligent Replenishment Engine — Day 6")
    print("=" * 60)

    cfg = ReplenishmentConfig()
    engine = ReplenishmentEngine(cfg)

    engine.load_data()
    engine.simulate_current_inventory()
    engine.classify_abc_xyz()
    engine.generate_recommendations()
    engine.generate_transfer_recommendations()
    engine.run_policy_simulation(sim_weeks=13)
    engine.save_reports()
    engine.plot_inventory_health_matrix()
    engine.plot_replenishment_priority()
    engine.plot_policy_simulation()
    engine.generate_report()

    print()
    print("=" * 60)
    print("Day 6 complete! Check reports/ and docs/validation/")
    print("=" * 60)


if __name__ == "__main__":
    main()
