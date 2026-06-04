# Research Digest: Day 1 — Pharma 101, Literature & Methods

**Project:** Pharmaceutical Demand Forecasting & Intelligent Replenishment System  
**Date:** 2026-06-04  
**Audience:** Data-science team with strong ML skills, zero prior pharma domain knowledge  
**Reading time:** ~20 minutes

---

## Executive Summary

1. **China's pharma demand is uniquely volatile.** The National Volume-Based Procurement (VBP/集采) program creates ~50–70% price cuts and abrupt hospital-to-retail demand spillovers that act as known structural breaks. Seasonal epidemics (flu, RSV) and public holidays add cyclical volatility on top.

2. **ATC hierarchy is the secret weapon for SKU portfolios.** The WHO Anatomical Therapeutic Chemical (ATC) classification provides a medically validated 5-level taxonomy. It lets us borrow forecasting strength across related drugs (e.g., all ACE inhibitors), handle generic substitution, and reconcile forecasts bottom-up or top-down across 380k SKUs.

3. **Segmented modeling is non-negotiable for pharma.** 78% of SKUs are long-tail / intermittent. Academic consensus (and SAP/IQVIA practice) assigns **ETS/Prophet to smooth demand** (ADI < 1.32), **Croston/SBA/TSB to intermittent demand**, and **gradient boosting (XGBoost/LightGBM)** to high-value demo SKUs with rich external regressors.

4. **The competitive whitespace is real.** SAP IBP and IQVIA serve large manufacturers and hospital chains. No major incumbent offers long-tail-aware, VBP-responsive forecasting for China's fragmented OOH B2B franchise network — precisely where the client operates.

---

## Domain Primer

### 1. Out-of-Hospital (OOH) / Retail Pharmacy vs. Hospital Distribution

**(a) Definition:**  
- **Hospital distribution:** Drugs flow from manufacturers → national/regional wholesalers → hospital pharmacies. Pricing and procurement are tightly regulated; VBP-selected drugs must be purchased by public hospitals at the winning bid price.
- **OOH / Retail:** Drugs reach patients through **chain pharmacies, independent/franchise pharmacies, clinics, and online channels**. This is the "retail" or "院外市场" market. The client operates a B2B e-commerce platform connecting upstream suppliers to these downstream OOH points.

**(b) Why it matters for demand forecasting:**  
- **Two distinct demand regimes:** Hospital demand is driven by physician prescribing behavior and insurance reimbursement. Retail demand is driven by patient self-selection, OTC habits, insurance portability, and price sensitivity.
- **VBP spillover effect:** When a drug is selected in VBP, hospital procurement is mandated, but **retail pharmacies may see demand surge** as patients who previously filled prescriptions in hospitals shift to retail to access non-selected brands or generics. This creates a **structural break** in retail time-series.
- **Inventory ownership differs:** Hospitals typically hold inventory on consignment or via centralized procurement. Franchise pharmacies own their own stock, creating fragmented, decentralized replenishment decisions.

**(c) Data-modeling implications:**  
- Need separate demand models (or at least channel-specific features) for hospital vs. retail.
- VBP batch dates are **known exogenous shock variables** — perfect for intervention analysis / dummy regressors.
- Pharmacy-type features (chain vs. independent) should be included as categorical predictors.

---

### 2. National Volume-Based Procurement (VBP) / 集采 in China

**(a) Definition:**  
China's National Healthcare Security Administration (NHSA) runs centralized, tender-based bulk purchasing for drugs used in public hospitals. Winners receive guaranteed volume in exchange for steep price reductions. As of 2025, **10 rounds** of national VBP have been completed.

**(b) Mechanics & batch history:**

| Batch | Date | Key Stats | Affected Categories |
|-------|------|-----------|---------------------|
| 1st–9th | 2018–2023 | Progressive expansion | Cardiovascular, diabetes, oncology, antibiotics, etc. |
| **10th** | **Dec 2024 bid; Apr 2025 rollout** | **62 varieties, ~74.5% price cut** (largest ever); 439 firms bid | CVD, diabetes, oncology, respiratory, psychiatry, emergency meds |
| 11th | Nov 2025 bid | ~53% price cut; "anti-internal competition" rules added | Expanding to biologics, complex generics |

**Key mechanics:**
- **Price cuts:** Typically 50–70%, with the 10th round seeing some injectables drop 90–96%.
- **Demand shift:** When a molecule is VBP-selected, hospital demand concentrates on the winning generic(s). **Retail demand for non-selected branded alternatives or other generics can spike** as patients and physicians switch channels.
- **Market concentration:** Winners are guaranteed volume, but must supply at extremely thin margins. Losers (including many originators) pivot to retail/OOH.

**(c) Data-modeling implications:**
- **VBP batch dates** are known in advance → use as **binary indicator features** in forecasting models.
- **Post-VBP demand trajectory** typically shows: (i) immediate drop as hospital volume shifts to winning bidder, (ii) potential retail spike for alternatives, (iii) new equilibrium after 2–4 quarters.
- **Cross-elasticity:** Model demand correlation between VBP-selected drugs and their therapeutic substitutes within the same ATC-4 group.

---

### 3. ATC Classification System

**(a) Definition:**  
The WHO **Anatomical Therapeutic Chemical (ATC)** classification is a 5-level hierarchy for drug grouping:

| Level | Code | Description | Example |
|-------|------|-------------|---------|
| 1 | 1 letter | Anatomical main group | **C** = Cardiovascular system |
| 2 | 3 chars | Therapeutic subgroup | **C10** = Lipid-modifying agents |
| 3 | 4 chars | Pharmacological subgroup | **C10A** = Lipid-modifying agents, plain |
| 4 | 5 chars | Chemical subgroup | **C10AA** = HMG CoA reductase inhibitors (statins) |
| 5 | 7 chars | Chemical substance | **C10AA05** = Atorvastatin |

**(b) Why it matters for forecasting:**
- **Borrowing strength:** Related drugs (same ATC-4) share demand drivers. A flu antiviral forecast can be improved by pooling information across all neuraminidase inhibitors.
- **Substitution:** When one statin is VBP-selected, demand shifts to other statins in the same ATC-4. The hierarchy encodes these relationships.
- **Portfolio management:** ATC-1/ATC-2 levels map to business franchises; ATC-5 maps to individual SKUs.

**(c) Data-modeling implications:**
- Build **hierarchical forecasting** with reconciliation (bottom-up, top-down, or optimal combination) across ATC levels.
- Use ATC-4 similarity for **demand segmentation** and **feature engineering** (e.g., "number of VBP-selected drugs in same ATC-4").
- ATC codes are stable, globally standardized, and publicly available via [atcddd.fhi.no](https://atcddd.fhi.no/atc_ddd_index/).

---

### 4. SKU Attributes That Matter for Inventory

| Attribute | Why It Matters | Modeling Implication |
|-----------|---------------|----------------------|
| **Shelf life / expiry** | Pharma SKUs have finite lifespans (often 24–36 months). Expired stock = 100% write-off. | FEFO picking rules; constrain order quantity by days-to-expiry; flag >6 months as "at-risk." |
| **Cold-chain** | Vaccines, biologics, insulin require 2–8°C. Breaks = spoilage. | Higher safety stock at nodes with reliable cold storage; shorter review periods. |
| **Narcotic / controlled** | Schedule I/II substances have strict procurement quotas and tracking. | Demand is capped by regulatory limits; model as bounded time-series. |
| **Generic vs. branded** | Post-VBP, generics dominate hospital channels; brands retreat to retail. | Generic flag interacts with VBP status; branded drugs may see residual premium demand. |
| **Formulation** | Same molecule, different forms (tablet, injectable, sustained-release) have different demand patterns. | Formulation type as a categorical feature; injectables often spikey (hospital-driven). |

---

### 5. Terminal Inventory

**(a) Definition:**  
"Terminal inventory" refers to stock held at the **final point of distribution** before consumption — in this case, the **franchise pharmacies and clinics** that buy through the B2B platform.

**(b) Who owns it:**  
- **Franchise pharmacies** typically own their inventory. They order from the platform (or via the platform from wholesalers) and bear holding costs and expiry risk.
- **The platform** itself may hold inventory in regional DCs, but the "last mile" stock is pharmacy-owned.
- This creates a **decentralized inventory problem:** 28,600 pharmacies each making independent replenishment decisions with limited visibility into network-wide demand.

**(c) Data-availability challenges:**  
- **POS data lag:** Pharmacies may not report sales in real-time. The platform sees **orders**, not **consumption** — and orders can be lumpy (batch ordering, forward-buying before price changes).
- **Expiry tracking:** Batch/lot-level expiry data may not be digitized at the franchise level.
- **Phantom inventory:** Stockouts at one pharmacy may not trigger network-wide reallocation if visibility is poor.

---

## Literature Review

### Paper 1: Machine Learning Approaches for Pharmaceutical Demand Forecasting

**Citation:**  
IIETA. (2026). Machine Learning Approaches for Pharmaceutical Demand Forecasting. *Intelligent Systems and Information*, 31(1), 17–30. https://doi.org/10.18280/isi.310117

**Summary:**  
Systematic review of ML methods applied to pharmaceutical demand forecasting. Surveys Random Forest, LSTM, ANN, XGBoost, and Linear Regression across multiple drug categories. Finds that **LSTM captures temporal dependencies better than ARIMA** for drug demand with complex seasonality, while **XGBoost and Random Forest excel at handling high-dimensional, nonlinear SKU-level data**.

**Key findings:**
- XGBoost and RF are the most frequently applied algorithms in recent pharma forecasting literature.
- Sustainability (reducing waste) and inventory optimization are central themes.
- Ensemble methods consistently outperform single statistical models when external regressors (seasonality, promotional activity) are available.

**Pros:** Comprehensive survey; directly relevant methods.  
**Cons:** Does not address intermittent demand or VBP-like policy shocks specifically.  
**Relevance to our case:** High — confirms XGBoost/LightGBM as the right choice for the advanced demo SKU.

---

### Paper 2: Hierarchical Forecasting for Large SKU Portfolios (ATC-Based)

**Citation:**  
DOAJ. (2024). Applying Machine Learning and Statistical Forecasting Methods for Enhancing Pharmaceutical Sales Predictions. *Directory of Open Access Journals*. https://doaj.org/article/cb9fa4b1c909406988ca7837e8fbd2e1

**Summary:**  
Applied **XGBoost, LSTM, and ARIMA** to ~600,000 pharmacy sales records, segmenting drugs into **8 clusters based on ATC codes**. XGBoost achieved the **lowest MAPE (16–18%)** for categories including anti-inflammatories (M01AB, M01AE), analgesics (N02BE), and sedatives (N05C). ATC-based segmentation revealed distinct seasonality patterns across therapeutic categories.

**Key findings:**
- ATC clustering improves forecast accuracy by grouping drugs with similar demand patterns.
- Seasonality varies significantly by therapeutic class (e.g., respiratory drugs spike in winter, CNS drugs are more stable).

**Pros:** Directly demonstrates ATC-based hierarchical forecasting at scale; uses real pharmacy data.  
**Cons:** Does not report on intermittent demand handling or policy shocks.  
**Relevance:** Very high — validates our core strategy of ATC hierarchy + ML segmentation.

---

### Paper 3: Hybrid GWO–XGBoost for Hospital Pharmaceutical Demand

**Citation:**  
Scientific Reports. (2026). A hybrid grey wolf optimized eXtreme gradient boosting-based machine learning model for hospital pharmaceutical demand forecasting. *Nature Scientific Reports*. https://doi.org/10.1038/s41598-026-48590-4

**Summary:**  
Developed a **Hybrid GWO–XGBoost** model to forecast weekly pharmaceutical demand at two provincial hospitals in Thailand (Oct 2023–May 2025). Used external regressors including meteorological data (temperature, precipitation, humidity). Achieved **R² = 0.984, MAE = 0.81, RMSE = 2.65**, outperforming GWO–LightGBM and GWO–RF baselines.

**Key findings:**
- External regressors (weather, temporal features) significantly boost accuracy.
- Authors explicitly note that **epidemiological data (flu incidence) should be incorporated** in future work as it is known to drive drug consumption.
- 1-week lagged features dominated feature importance.

**Pros:** State-of-the-art accuracy; rigorous validation; clear feature importance.  
**Cons:** Hospital-only setting; no VBP/shock variables; computationally expensive hyperparameter optimization.  
**Relevance:** High — provides the blueprint for our advanced XGBoost model. We should add ILI%, VBP flags, and holiday dummies to their regressor set.

---

### Paper 4: Intermittent Demand Theory and Practice (The Definitive Review)

**Citation:**  
Boylan, J. E., & Syntetos, A. A. (2021). Intermittent demand forecasting. In *Forecasting: Theory and Practice* (pp. chapter 14). https://doi.org/10.1016/j.ijforecast.2021.10.002

**Summary:**  
The canonical review of intermittent demand forecasting. Establishes **Croston's method** (1972) as the foundational approach, **SBA (Syntetos-Boylan Approximation, 2005)** as the corrected benchmark, and **TSB (Teunter-Syntetos-Babai)** as the modern alternative that updates demand probability every period. SBA is shown to **outperform Croston when ADI > 1.25 periods**.

**Key findings:**
- SBA "constitutes the benchmark against which other proposed methodologies in the area of intermittent demand forecasting are assessed."
- Parametric methods (SBA, TSB) outperform bootstrapping for safety-stock determination on real industry datasets.
- TSB is preferred when obsolescence risk exists (relevant for pharma expiry).

**Pros:** Authoritative; cited by SAP and major planning systems.  
**Cons:** Not pharma-specific; validated primarily on spare-parts and MRO data.  
**Relevance:** Very high — provides the theoretical basis for our long-tail SKU segmentation (ADI < 1.32 → ETS; ADI ≥ 1.32 → Croston/SBA/TSB).

---

### Paper 5: Multi-Echelon Inventory Optimization (MEIO) in Pharma

**Citation:**  
Valluri, K. (2024). Part-1: Multi-Echelon Inventory Optimization (MEIO) in the Pharmaceutical Supply Chain. *LinkedIn / Industry Analysis*. https://www.linkedin.com/pulse/multi-echelon-inventory-optimization-meio-supply-chain-valluri-ph0wc

**Summary:**  
Industry case study of MEIO applied to a global vaccine manufacturer. Modeled a 6-echelon network (API supplier → manufacturing → packaging → regional DC → wholesaler → clinic/pharmacy). Introduced **shelf-life-aware safety stock** and **FEFO-based replenishment**. Results: **26% expiry reduction, OTIF improved from 92% → 98.4%, $12M/year inventory savings**.

**Key findings:**
- Shifting buffers from clinics to regional 3PL hubs reduces expiry while maintaining service levels.
- Simulation-optimization models that account for perishability outperform static policies.
- Johnson & Johnson achieved 25% cost reduction + 30% fewer stockouts using MEIO for medical supplies.

**Pros:** Real-world pharma case study with quantified ROI.  
**Cons:** Grey literature (LinkedIn); specific model details not peer-reviewed.  
**Relevance:** High — validates the multi-echelon perspective and expiry-aware replenishment logic for our engine.

---

## Inventory & Replenishment Methods

### 1. Safety-Stock Policies by Demand Type

| Demand Type | Classification (ADI) | Recommended Forecast Method | Safety Stock Formula | Key Assumptions |
|-------------|----------------------|----------------------------|----------------------|-----------------|
| **Smooth** | ADI < 1.32 | ETS / Prophet / ARIMA | $SS = Z \times \sigma_d \times \sqrt{L}$ | Normal lead-time demand; continuous review or periodic review |
| **Intermittent** | 1.32 ≤ ADI < 5 | Croston / SBA / TSB | Negative binomial–Bernoulli (NBB) or Croston-based SS | Demand occurrence ≠ demand size; update on positive-demand periods only |
| **Lumpy** | ADI ≥ 5 | TSB (preferred) or bootstrapping | Simulation-based or parametric NBB | High obsolescence risk; expiry dominates cost |

**Practical notes:**
- **SBA vs. TSB:** Use SBA for stable intermittent demand; use TSB when demand may be declining (expiry risk, generic substitution, end-of-life products).
- **Service level Z:** Adjust by pharmacy type — hospitals may warrant Z = 2.33 (99% service), while independents may use Z = 1.65 (95%).
- **Lead time L:** In pharma B2B, L is typically 3–7 days for domestic generics, 14–30 days for imported brands or cold-chain products.

---

### 2. Expiry-Aware Replenishment

**Core principle:** FEFO (First-Expired-First-Out), not FIFO.  

**Operational practices:**
- **Shelf-life buffer:** Do not order if expected demand over (review period + lead time) exceeds the remaining shelf life minus a safety buffer (e.g., 6 months).
- **Expiry risk scoring:** Flag lots with < 6 months to expiry. Prioritize redistribution from low-demand to high-demand locations before ordering new stock.
- **Discard cost:** Modeled as the full unit cost of expired inventory plus disposal fees. For the demo, a simplified penalty function: `discard_cost = unit_cost × max(0, inventory_on_hand − expected_demand_until_expiry)`.
- **Lot-level visibility:** Required for FEFO enforcement. If batch data is unavailable, use "age-based inventory" approximations (FIFO as a fallback with expiry risk heuristics).

**Industry benchmarks:** AI-driven expiry management has shown **20–40% reduction in expired product write-offs** (Sakara Digital, 2024; BlueSight, 2024).

---

### 3. Multi-Echelon Inventory (Platform Context)

The client platform sits between **manufacturers/wholesalers** and **franchise pharmacies**. This is a **2-echelon distribution network** (platform DC → pharmacy) with these characteristics:

- **Echelon 1 (Platform/Wholesaler):** Holds pooled inventory. Benefits from risk pooling across pharmacies.
- **Echelon 2 (Pharmacies):** 28,600 independent decision-makers with limited coordination.

**Applicable models:**
- **Base-stock policies:** Each echelon maintains an order-up-to level S. For pharmacies, periodic review (R, S) is more realistic than continuous review (Q, R) because pharmacies do not monitor inventory continuously.
- **DDMRP (Demand Driven MRO):** Less applicable here. DDMRP excels in manufacturing BOM contexts, not retail pharmacy distribution.
- **Virtual pooling:** The platform can recommend **network-wide reallocation** (move stock from overstocked pharmacies to stockout-risk pharmacies) before triggering new orders.

**Recommendation for the prototype:** Implement a **periodic review (R, S) policy at the pharmacy level**, with R = 7 days (weekly) or 14 days (bi-weekly), and platform-level visibility for reallocation recommendations.

---

### 4. Replenishment Frequency in Pharmacy Retail

**Real-world practice:**
- **Chain pharmacies:** Often review inventory weekly (R = 7) via centralized procurement systems.
- **Independent pharmacies:** May review bi-weekly or even monthly (R = 14 or 30), especially for slow-movers.
- **Cold-chain / high-value items:** Daily or twice-weekly review (R = 1–3).
- **Hospital-affiliated pharmacies:** Tied to hospital procurement cycles; typically weekly.

**Modeling choice:** Use **R = 7 days** as the default for the demo, with pharmacy-type adjustment (chain = 7, independent = 14, hospital = 7).

---

## External Data Sources

| Source | URL / Access Method | Update Frequency | Integration Method | Notes |
|--------|--------------------|------------------|--------------------|-------|
| **China CDC Weekly ILI Surveillance** | https://ivdc.chinacdc.cn/cnic/en/Surveillance/WeeklyReport/ | Weekly (Thurs/Fri) | Scrape PDF → extract Southern/Northern ILI% | No public CSV/API; researchers manually extract or contact fluchina@ivdc.chinacdc.cn |
| **WHO FluNet** | https://www.who.int/tools/flunet | Weekly | API or CSV download | China reports to WHO; aggregated international view |
| **China Public Holidays** | https://www.china-briefing.com/news/china-public-holiday-2025-schedule/ | Annual | Binary dummy features (Spring Festival, Golden Week, etc.) | 2025 added 2 days; Spring Festival = major demand disruption |
| **VBP Batch Dates** | NHSA announcements + industry reports (e.g., 36Kr, Pharmaboard) | Per batch (~annually) | Binary dummy features by batch date + ATC class | 10th round: Apr 2025 rollout; 11th round: Nov 2025 bid |
| **Baidu Index** | https://index.baidu.com/ | Daily | API or scrape for flu-related search terms | Proxy for consumer health concern; correlates with OTC demand |
| **Open-Meteo Weather** | https://open-meteo.com/en/docs | Daily | API for temperature, humidity by city | Weather drives respiratory drug demand; used in Nature GWO-XGBoost study |
| **ATC/DDD Index** | https://atcddd.fhi.no/atc_ddd_index/ | Annual updates | Lookup table for ATC-1 to ATC-5 mapping | WHO Collaborating Centre; free and authoritative |
| **Pharbers / IQVIA Market Data** | Commercial (syndicated) | Monthly/Quarterly | Purchase or academic license | Real-world sales data; expensive but ground-truth for validation |

**Recommended synthetic data approach:** Since most sources above require scraping, manual extraction, or commercial licenses, the demo should **simulate realistic ILI curves, VBP shock dates, and holiday effects** based on the historical patterns documented above. The ILI% time-series can be modeled as a smoothed epidemic curve with seasonal peaks (weeks 49–10) and variable amplitude year-to-year.

---

## Competitive Landscape

### 1. IQVIA — Forecast Horizon & Commercial AI Suite

**What it does:**  
IQVIA is the dominant healthcare data and analytics company. Its **Forecast Horizon** platform provides algorithmic forecasting for later-stage and inline pharmaceutical products. Integrates Python/R, real-world data (prescriptions, claims), and HCP segmentation. Also offers **OCE** (CRM/commercial engagement), **Market Prognosis** (10-year disease-level forecasts), and **GenAI Launch Cockpit**.

**Forecasting method:** Algorithmic (ensemble) forecasting with machine learning; integrates syndicated market data, competitive dynamics, and prescriber-level segmentation.

**Pricing / deployment:** Enterprise SaaS; high six-to-seven-figure annual contracts. Targets large pharma manufacturers.

**Gap it leaves:**  
- Does **not** serve China's fragmented OOH retail pharmacy network.
- Focuses on **brand-level commercial forecasting** ("how much will this drug sell?"), not **inventory replenishment** ("how much should this pharmacy order?").
- No VBP-specific demand-shift modeling for retail channels.

---

### 2. SAP IBP (Integrated Business Planning) + SAP Business Data Cloud

**What it does:**  
SAP IBP is the enterprise-standard platform for demand planning, S&OP, supply planning, and inventory optimization. The newer **SAP Business Data Cloud** connects internal SAP data (IBP, S/4HANA) with external market data (including IQVIA feeds) for unified forecasting.

**Forecasting method:** Statistical baseline + ML-enhanced demand sensing. Supports hierarchical forecasting, safety-stock optimization, and multi-echelon inventory.

**Pricing / deployment:** Enterprise license + implementation consulting; typical projects cost $500K–$5M+.

**Gap it leaves:**  
- Targets **large enterprises with existing SAP estates** — not accessible to small/medium franchise pharmacies.
- Requires heavy IT integration and master data governance.
- No native understanding of China's VBP policy mechanics or OOH channel dynamics.

---

### 3. 药师帮 (Yaoshibang) / Chinese B2B Platforms

**What it does:**  
药师帮 is China's largest third-party pharmaceutical B2B platform (2021 GMV: RMB 275 billion, 18.5% market share, 434,000+ pharmacy clients). It connects upstream manufacturers/distributors to downstream small pharmacies and clinics. Competitors include **药京采 (JD Health)**, **1药网**, **药帮忙**, and **合纵药易购**.

**Forecasting method:** These platforms primarily operate as **transaction marketplaces** with limited predictive analytics. Some offer basic sales dashboards and reorder suggestions, but **no advanced demand forecasting or VBP-aware replenishment** has been publicly disclosed.

**Pricing / deployment:** Free to join; revenue from transaction commissions (typically 1–3%) and value-added services.

**Gap it leaves (our opportunity):**  
- **No long-tail SKU forecasting:** 78% of SKUs are low-frequency; current platforms likely focus on fast-movers.
- **No VBP shock modeling:** Pharmacists manually adjust to policy changes.
- **No expiry-aware replenishment:** FEFO logic and shelf-life constraints are not automated.
- **No external signal integration:** Flu season, holiday effects, and epidemiological data are not used to predict demand.
- **Fragmented decision-making:** 28,600 pharmacies order independently with no network-wide optimization.

---

## Bibliography

1. Boylan, J. E., & Syntetos, A. A. (2021). Intermittent demand forecasting. In *Forecasting: Theory and Practice*. International Journal of Forecasting. https://doi.org/10.1016/j.ijforecast.2021.10.002

2. IIETA. (2026). Machine Learning Approaches for Pharmaceutical Demand Forecasting. *Intelligent Systems and Information*, 31(1), 17–30. https://doi.org/10.18280/isi.310117

3. DOAJ. (2024). Applying Machine Learning and Statistical Forecasting Methods for Enhancing Pharmaceutical Sales Predictions. *Directory of Open Access Journals*. https://doaj.org/article/cb9fa4b1c909406988ca7837e8fbd2e1

4. Nature Scientific Reports. (2026). A hybrid grey wolf optimized eXtreme gradient boosting-based machine learning model for hospital pharmaceutical demand forecasting. *Scientific Reports*. https://doi.org/10.1038/s41598-026-48590-4

5. Valluri, K. (2024). Part-1: Multi-Echelon Inventory Optimization (MEIO) in the Pharmaceutical Supply Chain. *LinkedIn*. https://www.linkedin.com/pulse/multi-echelon-inventory-optimization-meio-supply-chain-valluri-ph0wc

6. Sakara Digital. (2024). AI-Powered Pharma Supply Chains. https://sakaradigital.com/blog/ai-powered-pharma-supply-chains-demand-sensing-inventory/

7. BlueSight / KitCheck. (2024). Pharmaceutical Inventory Management Best Practices for Growing Health Systems. https://bluesight.com/news/pharmaceutical-inventory-management-best-practices-for-growing-health-systems/

8. China CDC / CNIC. (2025). Chinese Weekly Influenza Surveillance Report. https://ivdc.chinacdc.cn/cnic/en/Surveillance/WeeklyReport/

9. ChemLinked / Baipharm. (2024). China Announces Results of the 10th Round of Volume-Based Drug Procurement (VBP). https://baipharm.chemlinked.com/news/china-announces-results-of-the-10th-round-of-volume-based-drug-procurement-vbp

10. 36Kr Research. (2023). 2023年中国医药电商B2B行业洞察报告. https://www.36kr.com/p/2153883523240200

11. WHO Collaborating Centre for Drug Statistics Methodology. (2026). ATC/DDD Index 2026. https://atcddd.fhi.no/atc_ddd_index/

12. IQVIA. (2021). How Can Open Source and Self-Service Analytics Help Pharma Forecasting? https://www.iqvia.com/blogs/2021/02/how-can-open-source-and-self-service-analytics-help-pharma-forecasting

13. IQVIA. (2024). Algorithmic Forecasting for the Life Sciences Industry (White Paper). https://www.iqvia.com/-/media/iqvia/pdfs/library/white-papers/algorithmic-forecasting.pdf

14. SAP. (2025). Building AI Ready Life Sciences Enterprise with SAP Business Data Cloud. https://community.sap.com/t5/technology-blog-posts-by-members/building-ai-ready-life-sciences-enterprise-with-sap-business-data-cloud/ba-p/14242504

15. MDPI Applied Sciences. (2025). A New Approach to Forecast Intermittent Demand and Stock. https://www.mdpi.com/2076-3417/15/22/12030

16. Babai, M. Z., et al. (2019). A new method to forecast intermittent demand in the presence of obsolescence. *International Journal of Production Economics*, 209, 30–41. https://doi.org/10.1016/j.ijpe.2018.06.017

17. ECDC. (2024). Seasonal influenza, 2023–2024, Annual Epidemiological Report. https://www.ecdc.europa.eu/sites/default/files/documents/seasonal-influenza-annual-epidemiological-report-2023-2024.pdf

18. arXiv. (2023). Big Data–Supply Chain Management Framework for Forecasting. https://arxiv.org/pdf/2307.12971v5.pdf

19. University of Vaasa. (2025). Demand forecasting in the retail environment: A comparative study of LightGBM, XGBoost, and MLP models. https://osuva.uwasa.fi/items/36a81ce5-6b36-4858-89a7-18d5eb05f695

20. 智研咨询. (2025). 2025年中国医药B2B电商行业发展历程、产业链、销售额、竞争格局及未来趋势研判. https://m.chyxx.com/industry/1220525.html

---

*Document generated by deep-research workflow. All citations verified against available sources. Where direct pharma-specific studies were unavailable (e.g., SBA in drug inventory), methods are inferred from validated spare-parts/MRO literature and flagged accordingly.*
