"""
Pharmaceutical Synthetic Data Generator
========================================
Generates three-layer synthetic data for the demand forecasting prototype:
  Layer 1: SKU Master Data (~500 SKUs with ATC hierarchy)
  Layer 2: External Signals (ILI%, holidays, VBP shocks)
  Layer 3: Weekly Sales Transactions (by pharmacy type)

Design constraints (per project plan):
  - Weekly granularity
  - No censored demand (observed = true demand)
  - National unified ILI% pattern
  - VBP shock = instantaneous step change
  - Random seed fixed for reproducibility
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

# ---------------------------------------------------------------------------
# Global random seed for full reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Matplotlib style
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    n_skus: int = 500
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    freq: str = "W-MON"  # Weekly, Monday start
    output_dir: Path = Path("data/raw")
    validation_dir: Path = Path("docs/validation")

    # Pharmacy types
    pharmacy_types: Tuple[str, ...] = ("hospital", "chain", "independent")
    pharmacy_type_weights: Dict[str, float] = None  # set in __post_init__

    # Demand-type mix (must sum to 1.0)
    demand_type_mix: Dict[str, float] = None  # set in __post_init__

    # VBP batch dates (instant step-change)
    vbp_batches: Dict[str, str] = None  # set in __post_init__

    # Therapeutic areas (ATC-1 level)
    therapeutic_areas: List[Dict] = None  # set in __post_init__

    def __post_init__(self):
        object.__setattr__(
            self,
            "pharmacy_type_weights",
            {"hospital": 1.5, "chain": 1.0, "independent": 0.6},
        )
        object.__setattr__(
            self,
            "demand_type_mix",
            {
                "smooth": 0.20,      # chronic meds, steady demand
                "seasonal": 0.15,    # flu-related, allergy
                "intermittent": 0.55, # long-tail, low frequency
                "shocked": 0.10,     # VBP-affected SKUs
            },
        )
        object.__setattr__(
            self,
            "vbp_batches",
            {
                "vbp_10": "2025-04-01",  # 10th round (largest cut, ~74.5%)
            },
        )
        object.__setattr__(
            self,
            "therapeutic_areas",
            [
                {"atc1": "C", "name": "Cardiovascular", "atc2": ["C08", "C09", "C10"]},
                {"atc1": "J", "name": "Antiinfectives", "atc2": ["J01", "J02"]},
                {"atc1": "A", "name": "Alimentary/Metabolism", "atc2": ["A10", "A02"]},
                {"atc1": "R", "name": "Respiratory", "atc2": ["R03", "R05", "R06"]},
                {"atc1": "N", "name": "Nervous System", "atc2": ["N02", "N05", "N06"]},
            ],
        )


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------
class PharmaDataGenerator:
    """Orchestrates generation of all three data layers."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.dates: pd.DatetimeIndex = pd.date_range(
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            freq=self.cfg.freq,
        )
        self.n_weeks: int = len(self.dates)

        # Ensure output dirs exist
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.validation_dir.mkdir(parents=True, exist_ok=True)

        # Containers for generated data
        self.master_data: pd.DataFrame | None = None
        self.external_signals: pd.DataFrame | None = None
        self.sales_data: pd.DataFrame | None = None

    # =====================================================================
    # Layer 1: Master Data
    # =====================================================================
    def generate_master_data(self) -> pd.DataFrame:
        """Generate ~500 SKUs with ATC hierarchy and clinical attributes."""
        cfg = self.cfg
        n = cfg.n_skus

        # Distribute SKUs across therapeutic areas
        areas = cfg.therapeutic_areas
        skus_per_area = [n // len(areas)] * len(areas)
        for i in range(n % len(areas)):
            skus_per_area[i] += 1

        records: List[Dict] = []
        sku_idx = 0

        for area, n_area in zip(areas, skus_per_area):
            for _ in range(n_area):
                sku_idx += 1
                sku_id = f"SKU_{sku_idx:04d}"

                # ATC hierarchy
                atc1 = area["atc1"]
                atc2 = np.random.choice(area["atc2"])
                atc3 = f"{atc2}{np.random.choice(['A', 'B', 'C'])}"
                atc4 = f"{atc3}{np.random.choice(['A', 'B', 'C', 'D'])}"

                # Demand type (stratified sampling)
                demand_type = np.random.choice(
                    list(cfg.demand_type_mix.keys()),
                    p=list(cfg.demand_type_mix.values()),
                )

                # Price (RMB per unit)
                price = np.round(np.random.lognormal(mean=3.0, sigma=1.2), 2)
                price = np.clip(price, 0.5, 5000.0)

                # Expiry window (months)
                expiry_months = int(np.random.choice([24, 36, 48, 60], p=[0.4, 0.35, 0.2, 0.05]))

                # VBP status (only for shocked + some smooth/seasonal)
                if demand_type == "shocked":
                    vbp_status = np.random.choice(["selected", "non_selected"], p=[0.6, 0.4])
                else:
                    vbp_status = np.random.choice(["na", "selected", "non_selected"], p=[0.7, 0.2, 0.1])

                # Generic vs branded
                if vbp_status == "selected":
                    generic_flag = 1  # selected generics
                elif demand_type == "shocked" and vbp_status == "non_selected":
                    generic_flag = np.random.choice([0, 1], p=[0.7, 0.3])  # mostly branded originators
                else:
                    generic_flag = np.random.choice([0, 1], p=[0.3, 0.7])

                # Cold chain
                cold_chain = 1 if atc1 == "J" and np.random.random() < 0.2 else 0

                # Base demand level (units per week) varies by therapeutic area
                area_base = {"C": 50, "J": 30, "A": 40, "R": 25, "N": 20}[atc1]
                base_demand = max(1, int(np.random.gamma(shape=area_base / 10, scale=10)))

                # Demand volatility (CV)
                if demand_type == "smooth":
                    cv = np.random.uniform(0.1, 0.4)
                elif demand_type == "seasonal":
                    cv = np.random.uniform(0.3, 0.8)
                elif demand_type == "intermittent":
                    cv = np.random.uniform(0.8, 2.5)
                else:  # shocked
                    cv = np.random.uniform(0.3, 1.0)

                records.append({
                    "sku_id": sku_id,
                    "atc1": atc1,
                    "atc2": atc2,
                    "atc3": atc3,
                    "atc4": atc4,
                    "therapeutic_area": area["name"],
                    "demand_type": demand_type,
                    "price_rmb": price,
                    "expiry_months": expiry_months,
                    "vbp_status": vbp_status,
                    "is_generic": generic_flag,
                    "cold_chain": cold_chain,
                    "base_demand": base_demand,
                    "demand_cv": round(cv, 2),
                })

        self.master_data = pd.DataFrame(records)
        print(f"[Master Data] Generated {len(self.master_data)} SKUs.")
        return self.master_data

    # =====================================================================
    # Layer 2: External Signals
    # =====================================================================
    def generate_external_signals(self) -> pd.DataFrame:
        """Generate weekly external time-series signals."""
        dates = self.dates
        n_weeks = self.n_weeks

        # --- ILI% (Influenza-Like Illness percentage) ---
        # National unified pattern: peaks in winter (weeks 49-10)
        week_of_year = dates.isocalendar().week.values

        # Seasonal component: two-peak structure (Dec-Jan major, Apr minor)
        phase = 2 * np.pi * week_of_year / 52
        seasonal_ili = (
            3.0 * np.sin(phase - np.pi / 3) +      # main winter peak
            1.0 * np.sin(2 * phase + np.pi / 4) +   # minor spring bump
            2.5                                     # baseline
        )
        # Add random year-to-year variation and noise
        yearly_trend = np.linspace(0, -0.5, n_weeks)  # post-COVID dampening
        noise = np.random.normal(0, 0.4, n_weeks)
        ili_pct = np.clip(seasonal_ili + yearly_trend + noise, 0.5, 12.0)

        # --- Holidays (binary flags) ---
        holidays = pd.DataFrame({"date": dates})
        holidays["year"] = holidays["date"].dt.year
        holidays["month"] = holidays["date"].dt.month
        holidays["day"] = holidays["date"].dt.day

        # Spring Festival (simplified: Jan or Feb, ~2 weeks)
        spring_festival = ((holidays["month"] == 1) & (holidays["day"] >= 20)) | \
                          ((holidays["month"] == 2) & (holidays["day"] <= 10))

        # National Day Golden Week (Oct 1-7)
        golden_week = (holidays["month"] == 10) & (holidays["day"] <= 7)

        # Labor Day (May 1-5)
        labor_day = (holidays["month"] == 5) & (holidays["day"] <= 5)

        # --- VBP batch flags ---
        vbp_10_flag = (dates >= pd.Timestamp("2025-04-01")).astype(int)

        self.external_signals = pd.DataFrame({
            "date": dates,
            "week_of_year": week_of_year,
            "ili_pct": np.round(ili_pct, 2),
            "is_spring_festival": spring_festival.astype(int),
            "is_golden_week": golden_week.astype(int),
            "is_labor_day": labor_day.astype(int),
            "is_holiday": (spring_festival | golden_week | labor_day).astype(int),
            "vbp_10_active": vbp_10_flag,
        })

        print(f"[External Signals] Generated {len(self.external_signals)} weeks.")
        return self.external_signals

    # =====================================================================
    # Layer 3: Sales Transaction Data
    # =====================================================================
    def generate_sales_data(self) -> pd.DataFrame:
        """Generate weekly sales for all SKU × pharmacy_type combinations."""
        if self.master_data is None:
            self.generate_master_data()
        if self.external_signals is None:
            self.generate_external_signals()

        master = self.master_data
        signals = self.external_signals
        dates = self.dates
        n_weeks = self.n_weeks
        ili = signals["ili_pct"].values
        vbp_flag = signals["vbp_10_active"].values
        holiday_flag = signals["is_holiday"].values

        all_sales: List[Dict] = []

        for _, sku in master.iterrows():
            sku_id = sku["sku_id"]
            d_type = sku["demand_type"]
            base = sku["base_demand"]
            cv = sku["demand_cv"]
            vbp_status = sku["vbp_status"]

            for p_type in self.cfg.pharmacy_types:
                weight = self.cfg.pharmacy_type_weights[p_type]
                adj_base = base * weight

                # Route to appropriate generator
                if d_type == "smooth":
                    sales = self._generate_smooth(
                        adj_base, cv, n_weeks, holiday_flag
                    )
                elif d_type == "seasonal":
                    sales = self._generate_seasonal(
                        adj_base, cv, ili, n_weeks, holiday_flag
                    )
                elif d_type == "intermittent":
                    sales = self._generate_intermittent(
                        adj_base, cv, n_weeks
                    )
                elif d_type == "shocked":
                    sales = self._generate_shocked(
                        adj_base, cv, vbp_status, vbp_flag, n_weeks, holiday_flag
                    )
                else:
                    sales = np.zeros(n_weeks)

                # Ensure non-negative integers
                sales = np.clip(np.round(sales), 0, None).astype(int)

                for t, (d, s) in enumerate(zip(dates, sales)):
                    all_sales.append({
                        "date": d,
                        "sku_id": sku_id,
                        "pharmacy_type": p_type,
                        "demand_type": d_type,
                        "units_sold": s,
                        "week_of_year": signals.iloc[t]["week_of_year"],
                        "ili_pct": signals.iloc[t]["ili_pct"],
                        "is_holiday": signals.iloc[t]["is_holiday"],
                        "vbp_10_active": signals.iloc[t]["vbp_10_active"],
                    })

        self.sales_data = pd.DataFrame(all_sales)
        print(f"[Sales Data] Generated {len(self.sales_data):,} rows.")
        return self.sales_data

    # -----------------------------------------------------------------
    # Demand generators by type
    # -----------------------------------------------------------------
    @staticmethod
    def _generate_smooth(
        base: float, cv: float, n_weeks: int, holiday_flag: np.ndarray
    ) -> np.ndarray:
        """Smooth demand: trend + weak seasonality + holiday dip + noise."""
        trend = np.linspace(0, base * 0.1, n_weeks)  # gentle growth
        # Weak annual seasonality
        t = np.arange(n_weeks)
        season = base * 0.05 * np.sin(2 * np.pi * t / 52)
        # Holiday dip
        holiday_effect = -base * 0.15 * holiday_flag
        # Noise
        noise = np.random.normal(0, base * cv, n_weeks)
        return base + trend + season + holiday_effect + noise

    @staticmethod
    def _generate_seasonal(
        base: float, cv: float, ili: np.ndarray, n_weeks: int, holiday_flag: np.ndarray
    ) -> np.ndarray:
        """Seasonal demand: strongly correlated with ILI% (flu season)."""
        # Elasticity: how much demand moves with ILI
        elasticity = np.random.uniform(2.0, 5.0)
        ili_effect = (ili - np.mean(ili)) * elasticity
        trend = np.linspace(0, base * 0.05, n_weeks)
        holiday_effect = -base * 0.1 * holiday_flag
        noise = np.random.normal(0, base * cv * 0.5, n_weeks)
        return base + trend + ili_effect + holiday_effect + noise

    @staticmethod
    def _generate_intermittent(
        base: float, cv: float, n_weeks: int
    ) -> np.ndarray:
        """Intermittent demand: Bernoulli occurrence + log-normal size."""
        # Probability of demand in any given week
        # Lower base demand → lower probability
        prob = np.clip(1.0 / (1.0 + base / 20.0), 0.05, 0.8)
        occurrence = np.random.binomial(1, prob, n_weeks)
        # Demand size (when it occurs)
        size = np.random.lognormal(mean=np.log(max(base, 1)), sigma=cv * 0.5)
        sizes = np.random.lognormal(
            mean=np.log(max(base, 1)), sigma=cv * 0.5, size=n_weeks
        )
        return occurrence * sizes

    @staticmethod
    def _generate_shocked(
        base: float,
        cv: float,
        vbp_status: str,
        vbp_flag: np.ndarray,
        n_weeks: int,
        holiday_flag: np.ndarray,
    ) -> np.ndarray:
        """Shocked demand: base pattern + instantaneous VBP step change."""
        t = np.arange(n_weeks)
        trend = np.linspace(0, base * 0.05, n_weeks)
        season = base * 0.03 * np.sin(2 * np.pi * t / 52)
        holiday_effect = -base * 0.1 * holiday_flag
        noise = np.random.normal(0, base * cv, n_weeks)

        # VBP shock multiplier
        if vbp_status == "selected":
            # Selected generic: retail may see some decline as patients go to hospital
            shock_mult = np.random.uniform(-0.25, -0.05)
        elif vbp_status == "non_selected":
            # Non-selected branded: retail demand surges (hospital spillover)
            shock_mult = np.random.uniform(0.4, 1.0)
        else:
            shock_mult = 0.0

        shock_effect = base * shock_mult * vbp_flag
        return base + trend + season + holiday_effect + noise + shock_effect

    # =====================================================================
    # Persistence
    # =====================================================================
    def save_all(self) -> None:
        """Save all three layers to CSV and Parquet."""
        if self.master_data is None:
            raise RuntimeError("Run generate_master_data() first.")
        if self.external_signals is None:
            raise RuntimeError("Run generate_external_signals() first.")
        if self.sales_data is None:
            raise RuntimeError("Run generate_sales_data() first.")

        out = self.cfg.output_dir

        # Master data
        self.master_data.to_csv(out / "master_data.csv", index=False, encoding="utf-8-sig")
        self.master_data.to_parquet(out / "master_data.parquet", index=False)

        # External signals
        self.external_signals.to_csv(out / "external_signals.csv", index=False, encoding="utf-8-sig")
        self.external_signals.to_parquet(out / "external_signals.parquet", index=False)

        # Sales data
        self.sales_data.to_csv(out / "sales_data.csv", index=False, encoding="utf-8-sig")
        self.sales_data.to_parquet(out / "sales_data.parquet", index=False)

        print(f"[Save] All data written to {out.absolute()}")

    # =====================================================================
    # Validation & Plots
    # =====================================================================
    def validate_and_plot(self) -> None:
        """Generate validation charts and save as PNG."""
        if self.sales_data is None:
            raise RuntimeError("Run generate_sales_data() first.")

        val_dir = self.cfg.validation_dir
        val_dir.mkdir(parents=True, exist_ok=True)

        # Prepare a pivoted view for plotting
        sales = self.sales_data.copy()
        signals = self.external_signals.copy()

        # --- Plot 1: Example time series by demand type ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Example SKU Demand by Type (Chain Pharmacy)", fontsize=14, fontweight="bold")

        demand_types = ["smooth", "seasonal", "intermittent", "shocked"]
        for ax, d_type in zip(axes.flat, demand_types):
            # Pick one example SKU of this type
            sku_ids = sales.loc[
                (sales["demand_type"] == d_type) & (sales["pharmacy_type"] == "chain"),
                "sku_id"
            ].unique()
            if len(sku_ids) == 0:
                ax.set_title(f"{d_type} (no data)")
                continue
            example_sku = sku_ids[0]
            subset = sales[
                (sales["sku_id"] == example_sku) & (sales["pharmacy_type"] == "chain")
            ].sort_values("date")

            ax.plot(subset["date"], subset["units_sold"], label=example_sku, linewidth=1.2)
            if d_type == "shocked":
                ax.axvline(
                    pd.Timestamp("2025-04-01"), color="red", linestyle="--",
                    label="VBP-10 Implementation"
                )
            ax.set_title(d_type.capitalize(), fontweight="bold")
            ax.set_ylabel("Units Sold")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.savefig(val_dir / "01_demand_types_examples.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved 01_demand_types_examples.png")

        # --- Plot 2: ILI% Seasonal Pattern ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(signals["date"], signals["ili_pct"], color="steelblue", linewidth=1.5)
        ax.fill_between(signals["date"], signals["ili_pct"], alpha=0.3, color="steelblue")
        ax.set_title("Simulated National ILI% Seasonal Pattern", fontsize=13, fontweight="bold")
        ax.set_ylabel("ILI%")
        ax.set_xlabel("Date")
        for year in signals["date"].dt.year.unique():
            ax.axvline(pd.Timestamp(f"{year}-04-01"), color="red", linestyle="--", alpha=0.4)
        ax.text(
            signals["date"].iloc[len(signals) // 2], ax.get_ylim()[1] * 0.9,
            "Red dashed = VBP-10 (Apr 2025)", color="red", fontsize=9, ha="center"
        )
        plt.tight_layout()
        plt.savefig(val_dir / "02_ili_seasonal_pattern.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved 02_ili_seasonal_pattern.png")

        # --- Plot 3: VBP Shock Before/After ---
        shocked_sales = sales[sales["demand_type"] == "shocked"]
        pre = shocked_sales[shocked_sales["date"] < "2025-04-01"].groupby("sku_id")["units_sold"].mean().reset_index()
        post = shocked_sales[shocked_sales["date"] >= "2025-04-01"].groupby("sku_id")["units_sold"].mean().reset_index()
        pre_post = pre.merge(post, on="sku_id", suffixes=("_pre", "_post"), how="inner")

        # Color by VBP status
        status_map = self.master_data.set_index("sku_id")["vbp_status"].to_dict()
        colors = ["green" if status_map.get(s, "na") == "selected" else "orange" for s in pre_post["sku_id"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(pre_post["units_sold_pre"], pre_post["units_sold_post"], c=colors, alpha=0.6, s=50)
        max_val = max(pre_post["units_sold_pre"].max(), pre_post["units_sold_post"].max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="No change line")
        ax.set_xlabel("Pre-VBP Average Weekly Sales")
        ax.set_ylabel("Post-VBP Average Weekly Sales")
        ax.set_title("VBP-10 Shock: Pre vs Post Demand (Shocked SKUs)", fontsize=13, fontweight="bold")
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="green", label="Selected Generic (↓ retail)"),
            Patch(facecolor="orange", label="Non-Selected Branded (↑ retail)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        plt.tight_layout()
        plt.savefig(val_dir / "03_vbp_shock_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved 03_vbp_shock_comparison.png")

        # --- Plot 4: ADI Distribution ---
        # Compute ADI per SKU from generated sales
        adi_df = (
            sales.groupby(["sku_id", "pharmacy_type"])
            .apply(lambda g: self._compute_adi(g["units_sold"].values))
            .reset_index(name="adi")
        )
        adi_df = adi_df.merge(self.master_data[["sku_id", "demand_type"]], on="sku_id", how="left")

        fig, ax = plt.subplots(figsize=(10, 5))
        for d_type, color in zip(
            ["smooth", "seasonal", "intermittent", "shocked"],
            ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
        ):
            subset = adi_df[adi_df["demand_type"] == d_type]["adi"]
            ax.hist(subset, bins=30, alpha=0.5, label=d_type.capitalize(), color=color, density=True)
        ax.axvline(1.32, color="black", linestyle="--", label="ADI = 1.32 threshold")
        ax.set_xlabel("Average Inter-Demand Interval (ADI)")
        ax.set_ylabel("Density")
        ax.set_title("ADI Distribution by Designated Demand Type", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(val_dir / "04_adi_distribution.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved 04_adi_distribution.png")

        # --- Plot 5: Pharmacy Type Comparison ---
        type_summary = (
            sales.groupby("pharmacy_type")["units_sold"]
            .agg(["mean", "std", "median"])
            .reset_index()
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Mean demand by type
        axes[0].bar(type_summary["pharmacy_type"], type_summary["mean"], color=["#e74c3c", "#3498db", "#2ecc71"])
        axes[0].set_title("Average Weekly Demand by Pharmacy Type", fontweight="bold")
        axes[0].set_ylabel("Mean Units Sold")

        # Coefficient of variation
        type_summary["cv"] = type_summary["std"] / type_summary["mean"]
        axes[1].bar(type_summary["pharmacy_type"], type_summary["cv"], color=["#e74c3c", "#3498db", "#2ecc71"])
        axes[1].set_title("Demand Volatility (CV) by Pharmacy Type", fontweight="bold")
        axes[1].set_ylabel("CV = Std / Mean")

        plt.tight_layout()
        plt.savefig(val_dir / "05_pharmacy_type_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved 05_pharmacy_type_comparison.png")

        print(f"[Validation] All 5 plots saved to {val_dir.absolute()}")

    @staticmethod
    def _compute_adi(series: np.ndarray) -> float:
        """Compute Average Inter-Demand Interval from a demand series."""
        positive_indices = np.where(series > 0)[0]
        if len(positive_indices) <= 1:
            return float(len(series))  # very sparse
        intervals = np.diff(positive_indices)
        return float(np.mean(intervals))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Pharmaceutical Synthetic Data Generator")
    print("=" * 60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Date range: 2023-01-01 to 2024-12-31 (weekly)")
    print()

    cfg = Config()
    gen = PharmaDataGenerator(cfg)

    # Layer 1
    gen.generate_master_data()
    print()

    # Layer 2
    gen.generate_external_signals()
    print()

    # Layer 3
    gen.generate_sales_data()
    print()

    # Save
    gen.save_all()
    print()

    # Validation plots
    gen.validate_and_plot()
    print()

    print("=" * 60)
    print("All done! Check data/raw/ and docs/validation/")
    print("=" * 60)


if __name__ == "__main__":
    main()
