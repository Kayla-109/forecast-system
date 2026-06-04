# Pharmaceutical Demand Forecasting & Intelligent Replenishment System

> A runnable prototype for China's largest out-of-hospital (OOH) pharmaceutical B2B platform.
> Built with Python, Streamlit, and XGBoost. Synthetic data powered.

## Project Overview

This prototype demonstrates two core capabilities for a pharmaceutical B2B e-commerce platform serving **28,600 franchise pharmacies** and **380,000 SKUs**:

1. **Multi-Factor Demand Forecasting** — 14-30 day ahead forecasts using segmented models (ETS/Prophet for smooth demand, Croston/SBA/TSB for intermittent demand, XGBoost for high-value SKUs with external regressors).
2. **Intelligent Replenishment Recommendations** — Periodic review (R,S) policy with FEFO expiry logic, dynamic safety stock, and network-wide reallocation suggestions.

Secondary features (kept functionally concise):
- Inventory Health Diagnosis (ABC/XYZ analysis, expiry risk flags)
- Policy Simulator (VBP batch impact simulation)
- Proactive Alerts (stockout < 5 days, expiry < 6 months)

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.10+, Pandas, NumPy, SciPy |
| Statistical Forecasting | Statsmodels (ETS), Prophet (seasonality+trend) |
| ML Forecasting | XGBoost, Scikit-learn |
| Intermittent Demand | Custom Croston / SBA / TSB implementations |
| Frontend | Streamlit (interactive, Python-native) |
| Visualization | Plotly (time-series), Matplotlib/Seaborn (validation) |
| Data Format | CSV (human-readable) + Parquet (compute-efficient) |

## Repository Structure

```
my-kimi-project/
├── data/
│   ├── raw/                    # Generated synthetic data (CSV + Parquet)
│   └── processed/              # Feature-engineered data
├── docs/
│   ├── validation/             # Auto-generated validation plots (PNG)
│   ├── research_day1_digest.md # Domain & literature research
│   └── research_day2_market_policy.md  # Market & policy research
├── notebooks/                  # Exploratory analysis (optional)
├── src/
│   ├── data_generator.py       # Synthetic data generator (Day 3-4)
│   ├── forecast_engine.py      # Demand forecasting engine (Day 5)
│   ├── replenishment_engine.py # Intelligent replenishment (Day 6)
│   └── utils.py                # Shared utilities
├── app.py                      # Streamlit application entry point (Day 8-10)
├── requirements.txt            # Python dependencies
├── PLAN.md                     # Master project plan
└── README.md                   # This file
```

## Quick Start

### 1. Clone & Enter Directory

```bash
cd my-kimi-project
```

### 2. Create Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `prophet` requires a C++ compiler. If installation fails:
> - **Windows:** Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> - **macOS:** Run `xcode-select --install`
> - **Linux:** Run `sudo apt-get install build-essential`

### 4. Generate Synthetic Data

Run the data generator to produce all three data layers:

```bash
python src/data_generator.py
```

This will:
1. Generate **~500 SKUs** with ATC hierarchy (Layer 1: Master Data)
2. Generate **external signals** — ILI%, holidays, VBP shock flags (Layer 2)
3. Generate **2 years of weekly sales** across 3 pharmacy types (Layer 3)
4. Save outputs to `data/raw/` (both CSV and Parquet)
5. Produce **5 validation plots** in `docs/validation/`:
   - `01_demand_types_examples.png` — Example time series by demand type
   - `02_ili_seasonal_pattern.png` — Simulated flu season curve
   - `03_vbp_shock_comparison.png` — Pre/post VBP demand shift
   - `04_adi_distribution.png` — Inter-demand interval distribution
   - `05_pharmacy_type_comparison.png` — Demand by pharmacy segment

Expected runtime: **5–15 seconds** on a modern laptop.

### 5. Inspect Output

```bash
# View generated data
ls data/raw/
# → master_data.csv / .parquet
# → external_signals.csv / .parquet
# → sales_data.csv / .parquet

# View validation charts
ls docs/validation/
```

## Data Architecture

### Layer 1: Master Data (`master_data.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| `sku_id` | Unique SKU identifier | `SKU_0001` |
| `name_cn` | Chinese drug name | `氨氯地平片` |
| `name_en` | English drug name | `Amlodipine` |
| `atc1` … `atc4` | WHO ATC hierarchy | `C` → `C08` → `C08C` → `C08CA01` |
| `therapeutic_area` | Human-readable category | `Cardiovascular` |
| `demand_type` | Generated pattern class | `smooth` / `seasonal` / `intermittent` / `shocked` |
| `price_rmb` | Unit price (CNY) | `8.5` |
| `expiry_months` | Shelf life from manufacture | `36` |
| `vbp_status` | VBP batch inclusion status | `selected` / `non_selected` / `na` |
| `is_generic` | Generic (1) vs branded (0) | `1` |
| `is_demo_sku` | Flag for demo SKUs | `1` (Metformin, Oseltamivir) |
| `ili_elasticity` | Sensitivity to ILI% | `0.0`–`8.0` |

### Layer 2: External Signals (`external_signals.csv`)

| Column | Description |
|--------|-------------|
| `date` | Week start (Monday) |
| `week_of_year` | ISO week number (1–52) |
| `ili_pct` | Simulated national ILI percentage |
| `is_spring_festival` | Binary flag (CNY period) |
| `is_golden_week` | Binary flag (Oct 1–7) |
| `is_labor_day` | Binary flag (May 1–5) |
| `is_holiday` | Union of all holiday flags |
| `vbp_10_active` | VBP 10th round active (from 2025-04-01, instantaneous step) |

### Layer 3: Sales Transactions (`sales_data.csv`)

| Column | Description |
|--------|-------------|
| `date` | Week start |
| `sku_id` | SKU identifier |
| `pharmacy_type` | `hospital` / `chain` / `independent` |
| `demand_type` | Designated pattern (for verification) |
| `units_sold` | Weekly sales volume (non-negative integer) |
| `week_of_year` | Calendar week |
| `ili_pct` | Joined ILI signal |
| `is_holiday` | Joined holiday flag |
| `vbp_10_active` | Joined VBP flag |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Time granularity** | Weekly | 2024-06-01 to 2026-05-31 = ~104 weeks |
| **SKU count** | 80 | Real Chinese drug names; stratified by demand type (16/12/44/8) |
| **Intermittent demand** | SBA (primary), TSB (advanced) | SBA is the academic benchmark; TSB handles obsolescence risk |
| **Censored demand** | **Not simulated** | Simplifies prototype; observed = true demand |
| **ILI% geography** | National unified | Sufficient for demo; regional split adds complexity without proportional insight |
| **VBP shock** | Instantaneous step change (2025-04-01) | Matches real-world policy implementation (mandated switch on effective date) |
| **Pharmacy heterogeneity** | Hospital/Chain/Independent with distinct CV and VBP response | Hospital: low CV, strong VBP shock; Independent: high CV, weaker shock |
| **Data** | Fully synthetic | No real dataset provided; synthetic allows perfect injection of business problems (long-tail, policy shocks) |

## Reproducibility

All random processes use a **fixed seed (`RANDOM_SEED = 42`)**. Running `data_generator.py` twice produces **identical output**.

To change the seed, edit the constant at the top of `src/data_generator.py`:

```python
RANDOM_SEED = 42  # Change to any integer
```

## Development Roadmap

| Phase | Days | Deliverable |
|-------|------|-------------|
| Research | 1–2 | Domain knowledge + market intelligence |
| Foundation | 3 | Tech stack + data architecture (✅ current) |
| Data & Algorithms | 4–7 | Synthetic data refinement + forecasting + replenishment engines |
| Application | 8–10 | Streamlit app with interactive dashboards |
| Business Case | 11–12 | Feasibility analysis + demo rehearsal |

## Research Bibliography

See `docs/research_day1_digest.md` and `docs/research_day2_market_policy.md` for full citations. Key sources include:

- Boylan, J. E., & Syntetos, A. A. (2021). *Forecasting: Theory and Practice* (intermittent demand review)
- IIETA (2026). Machine Learning Approaches for Pharmaceutical Demand Forecasting
- Nature Scientific Reports (2026). Hybrid GWO–XGBoost for hospital pharmaceutical demand
- China CDC / CNIC Weekly Influenza Surveillance Reports
- NHSA VBP batch announcements (1st–10th round)

## License

This is an academic / case-competition prototype. Not for commercial deployment without appropriate domain validation.

---

**Built for:** Pharmaceutical Demand Forecasting Case Competition  
**Team:** Data Science Background, Limited Pharma Domain Knowledge  
**Core Philosophy:** Polish 2 primary features (Forecasting + Replenishment) with excellent visualizations; keep secondary features functionally concise.
