# Research Digest: Day 2 — Market, Competitors & Policy

**Project:** Pharmaceutical Demand Forecasting & Intelligent Replenishment System  
**Date:** 2026-06-04  
**Audience:** Data-science team building prototype + business case  
**Reading time:** ~20 minutes

---

## Executive Summary

1. **No major B2B platform currently offers VBP-aware demand forecasting.** 药师帮 (Yaoshibang), 药京采 (JD Health), and 九州通 all operate as transaction marketplaces with basic dashboards, but none provide predictive analytics, automated replenishment, or policy-impact simulation to their pharmacy customers.

2. **VBP creates predictable structural breaks that can be modeled.** When a drug is selected in VBP, hospital demand concentrates on the winning generic while retail demand for non-selected branded alternatives typically surges. The 10th round (Apr 2025) saw 62 varieties drop ~74.5% in price — the largest cut in history.

3. **The competitive whitespace is inventory intelligence for franchise pharmacies.** SAP IBP and IQVIA serve large enterprises, not China's 280,000+ independent pharmacies. A lightweight, VBP-aware forecasting module integrated into a B2B platform addresses a real, unmet need.

4. **External signals are available but fragmented.** China CDC publishes weekly ILI% reports (PDF format), Baidu Index tracks search trends, and VBP batch dates are announced publicly. For the demo, these should be simulated based on documented seasonal and policy patterns.

5. **Willingness-to-pay is driven by quantifiable savings.** The case cites RMB 420m annual stockout losses and >165-day turnover for slow-movers. A subscription priced at RMB 200–500/month per pharmacy (or 0.5–1% of procurement value) is plausible if it demonstrably reduces expiry write-offs and rush-order costs.

---

## Competitive Landscape

### Scope
We focus on platforms serving **franchise/independent pharmacies and primary healthcare institutions** in China's out-of-hospital (OOH) market. Enterprise solutions targeting large hospital chains or pharma manufacturers (SAP IBP, IQVIA) are noted as "non-competitors" to highlight our unique focus.

### Competitor Map

| Player | Scale & Model | Key Features | Pricing | Gaps / Whitespace |
|--------|--------------|--------------|---------|-------------------|
| **药师帮 (Yaoshibang)** | 3rd-party B2B leader. 2021 GMV: RMB 275B; 434K+ pharmacy buyers; 18.5% market share. Pure marketplace +自营 hybrid. | Transaction platform; credit/financing; logistics; basic sales dashboards. | Free to join; revenue from transaction commissions (~1–3%). | **No demand forecasting.** No expiry-aware replenishment. No VBP impact analysis. Relies on pharmacies to manually decide what and when to order. |
| **药京采 (JD Health)** | JD Health's B2B arm. Leverages JD logistics (22 drug warehouses, cold-chain to 300 cities). | Procurement marketplace; same/next-day delivery for some regions; integrated with JD's supply chain. | Transaction-based; volume discounts. | **No predictive analytics.** Focus is on fulfillment speed and selection breadth, not inventory optimization. |
| **九州通网 (Jointown)** | Traditional distributor (#1 in China by volume). ~RMB 140B annual revenue. Owns warehouses and fleet. | Digital procurement for existing Jointown customers; offline-to-online migration. | Tiered by volume; enterprise contracts. | Serves larger customers better. **Limited AI/forecasting for small franchise pharmacies.** Inventory tools are basic reorder suggestions. |
| **1药网 (111 Inc.)** | NASDAQ-listed; B2B + B2C hybrid. | E-commerce platform; some SaaS tools for pharmacies. | Subscription + transaction fees. | **Forecasting is not a core offering.** Focus on drug access and online sales. |
| **合纵药易购 (Hezong)** | Regional leader (Sichuan-based); expanding nationally. | Procurement for independents; some financial services. | Transaction-based. | **No intelligent replenishment.** Regional coverage limits. |
| **SAP IBP** | Global enterprise demand planning. | MEIO, S&OP, safety-stock optimization. | $500K–$5M+ enterprise license. | **Not accessible to franchise pharmacies.** Requires SAP estate and heavy IT. Does not model China's VBP mechanics. |
| **IQVIA** | Healthcare data & analytics giant. | Forecast Horizon (algorithmic forecasting); Market Prognosis; AI agents for commercial teams. | High six-to-seven figure annual contracts. | Targets **large pharma manufacturers**, not retail pharmacy networks. No inventory replenishment module. |

### Key Insight: The Whitespace

The entire competitive landscape shares one trait: **everyone is competing on procurement access, price, and logistics speed. No one competes on inventory intelligence.**

Our prototype can fill the gap by offering:
- **VBP-aware demand forecasting** (competitors don't even acknowledge policy shocks).
- **Long-tail SKU optimization** (78% of SKUs are intermittent; current platforms ignore them).
- **FEFO/expiry-aware replenishment** (reducing write-offs, a major cost driver).
- **Network-wide reallocation** (moving stock from overstocked to stockout-risk pharmacies).

---

## Volume-Based Procurement Deep Dive

### Mechanics Recap

1. **Announcement:** NHSA publishes a VBP batch tender, listing target molecules and dosage forms.
2. **Bidding:** Manufacturers submit prices. Winners are selected based on price and supply capacity.
3. **Implementation:** Public hospitals are mandated to purchase the winning generic(s) at the bid price for a guaranteed volume (typically 1–3 years).
4. **Price outcome:** Selected generics drop 50–70%+ in price. Non-selected alternatives (including originator brands) are largely excluded from hospital procurement.

### Demand Shift Patterns: Hospital vs. Retail

When a drug is selected in VBP, three effects occur simultaneously:

| Effect | Hospital Channel | Retail / OOH Channel | Modeling Implication |
|--------|-----------------|---------------------|----------------------|
| **Selected generic** | Demand surges (mandated procurement) | Demand may drop (patients buy cheaper version in hospital) | ↓ Retail for selected SKU |
| **Non-selected branded (originator)** | Demand collapses (hospital stops stocking) | Demand often surges (patients who want brand-name shift to retail) | ↑ Retail for non-selected branded alternatives |
| **Non-selected generics** | Demand collapses | Mixed (some gain, some lose to selected generic) | Context-dependent; model per ATC-4 group |

**Evidence:**
- After VBP selection of statins and CCBs, **30+ antihypertensive drugs saw retail sales increase** while hospital sales of non-selected brands fell sharply.
- CCB-class drugs in retail now exceed hospital channel share (56.4% vs. 43.6%).
- AstraZeneca's rosuvastatin (落选) saw retail sales jump **~70%** after VBP forced it out of hospitals.

### Historical Batch Data

| Batch | Announcement / Bid | Implementation | Varieties | Avg Price Cut | Key Categories |
|-------|-------------------|----------------|-----------|---------------|----------------|
| 1st (4+7 pilot) | Nov 2018 | Mar 2019 | 25 | ~52% | Cardiovascular, psychiatric |
| 2nd | Dec 2019 | Apr 2020 | 32 | ~53% | Diabetes, antibiotics, oncology |
| 3rd | Aug 2020 | Nov 2020 | 55 | ~53% | Multiple |
| 4th | Feb 2021 | May 2021 | 45 | ~52% | Respiratory, GI |
| 5th | Jun 2021 | Oct 2021 | 61 | ~56% | Antibiotics, CVD, oncology |
| 6th | Jul 2022 | Nov 2022 | 45 | ~48% | Diabetes, antibiotics |
| 7th | Jul 2022 | Nov 2022 | 61 | ~48% | Multiple |
| 8th | Mar 2023 | Jul 2023 | 39 | ~56% | Antibiotics, CVD, diuretics |
| 9th | Nov 2023 | Mar 2024 | 41 | ~58% | Multiple |
| **10th** | **Dec 2024** | **Apr 2025** | **62** | **~74.5%** | **CVD, diabetes, oncology, psychiatry, antibiotics, emergency meds** |
| 11th | Nov 2025 | Early 2026 (est.) | — | ~53% | Expanding to biologics; "anti-internal competition" rules |

*Note: Precise average price cuts for early batches vary by source. The 10th round is the most well-documented due to its recency and magnitude.*

### Specific Drug Examples

| Molecule | Category | VBP Status | Observed Demand Shift |
|----------|----------|------------|----------------------|
| **Amlodipine** (氨氯地平) | CCB / Hypertension | Selected (multiple rounds) | Hospital: generic volume surged; Retail: non-selected brands gained share |
| **Atorvastatin** (阿托伐他汀) | Statin / Dyslipidemia | Selected | Hospital: generic dominant; Retail: originator (Lipitor) and non-selected brands saw demand rise |
| **Metformin** (二甲双胍) | Diabetes / Biguanide | Selected | Hospital: generic mandated; Retail: sustained demand, some spillover from brand-loyal patients |
| **Oseltamivir** (奥司他韦) | Antiviral / Influenza | Not selected in early rounds; later rounds vary | Retail demand highly seasonal (flu-driven); VBP status affects hospital stocking |
| **Rosuvastatin** (瑞舒伐他汀钙) | Statin | Originators largely落选 | AstraZeneca's brand saw **~70% retail sales increase** after hospital exclusion |

### Spillover & Price-Linkage Effects

Beyond direct demand shifts, VBP exerts **spillover pressure** on non-selected products:
- Multiple provinces enforce **price-linkage rules**: non-selected generic prices cannot exceed 1.5× the selected bid price.
- This compresses margins across the entire therapeutic class, not just the selected SKU.
- For the demo: model this as a **category-wide price elasticity shock** triggered at the VBP implementation date.

---

## External Data Signals

| Signal Name | Source / URL | Update Frequency | Usage in Forecast | Real / Simulated |
|------------|-------------|------------------|-------------------|------------------|
| **China CDC Weekly ILI%** | https://ivdc.chinacdc.cn/cnic/en/Surveillance/WeeklyReport/ | Weekly (Thu/Fri) | Numeric regressor for respiratory/antiviral drug demand; seasonal peak detection | **Real** (PDF scrape); recommend simulating for demo consistency |
| **WHO FluNet** | https://www.who.int/tools/flunet | Weekly | Cross-validation of China CDC data; international context | **Real** (API/CSV available) |
| **Baidu Index** | https://index.baidu.com/ | Daily | Search trends for drug names (e.g., 奥司他韦, 感冒药) as proxy for consumer demand intent | **Real** (API or scrape); good for Demo |
| **China Public Holidays** | https://www.china-briefing.com/news/china-public-holiday-2025-schedule/ | Annual | Binary flags: Spring Festival, National Day Golden Week, Labor Day | **Real** (fixed calendar); easy to implement |
| **VBP Batch Dates** | NHSA announcements; industry trackers (36Kr, Pharmaboard) | Per batch (~annually) | Binary shock variables; interaction with ATC-4 category flags | **Real** (known dates); critical for demo |
| **Open-Meteo Weather** | https://open-meteo.com/en/docs | Daily | Temperature, humidity for respiratory drug correlation | **Real** (free API); optional |
| **Air Quality Index** | China National Environmental Monitoring Centre | Daily | PM2.5 for respiratory medication demand (weaker signal) | **Real**; optional, lower priority |
| **School Calendar** | Provincial education bureaus | Semi-annual | Pediatric drug demand driver (school opening/closing) | **Simulated** (Sep 1, Jan 15, etc.) |

### Recommended Simulation Strategy

For the prototype, we recommend a **hybrid approach**:

1. **VBP dates & holidays:** Use real dates (100% accurate, publicly available).
2. **ILI% time series:** Simulate using a **seasonal epidemic curve** (SIR-inspired or Gaussian mixture) with:
   - Seasonal peak: weeks 49–10 (Dec–Mar)
   - Variable amplitude year-to-year (based on documented 2019–2024 patterns)
   - Post-COVID dampening (2021–2024 peaks lower than 2019)
3. **Baidu Index:** Simulate as a **leading indicator** of ILI% (correlation ~0.6–0.7, 1–3 day lag).
4. **Weather:** Pull real historical data via Open-Meteo API, or simulate seasonal temperature curves.

---

## Value Proposition & Willingness-to-Pay

### Target Users

| Segment | Role | Pain Point | Value Proposition |
|---------|------|-----------|-------------------|
| **Primary** | B2B platform product manager | Need to differentiate from pure procurement competitors | Embed forecasting as premium feature → increase platform stickiness |
| **Primary** | Procurement manager at 20–100 store franchise chain | Manual replenishment across hundreds of SKUs; high stockout + expiry costs | Automated replenishment with VBP-aware alerts → reduce losses |
| **Secondary** | Independent pharmacy owner | Cash flow constraints; cannot afford to overstock slow-movers | Free basic alerts + pay-per-use advanced forecasting |
| **Secondary** | Hospital outpatient pharmacy director | VBP shocks disrupt inventory; need to manage generic substitution | Policy simulation module → preemptive reallocation |

### Quantifying the Pain

From the case brief and industry data:
- **RMB 420m annual stockout loss** across the network.
- **Inventory turnover >165 days** for slow-moving items (industry healthy benchmark: ~60–90 days).
- **~78% of SKUs are low-frequency / long-tail**, making manual forecasting nearly impossible.
- **Expiry write-offs** estimated at 3–5% of inventory value annually for pharma retail.

### Potential Savings (Illustrative)

| Improvement Area | Conservative Estimate | Source Logic |
|-----------------|----------------------|--------------|
| Stockout reduction | 15–25% of RMB 420m = **RMB 63–105m/year** | Better forecasting + safety-stock tuning |
| Expiry write-off reduction | 20–30% of 3–5% inventory value | FEFO-aware replenishment + expiry alerts |
| Rush-order cost reduction | 10–15% of emergency procurement spend | Proactive reordering eliminates express freight |
| Inventory carrying cost | 10–15% inventory reduction at same service level | Long-tail optimization + dynamic safety stock |

### Pricing Benchmark

| Comparable Offering | Price Point | Notes |
|--------------------|-------------|-------|
| Generic SaaS inventory tools (China) | RMB 100–300/month/store | Basic reorder point calculators; no ML |
| JD Health / Yaoshibang premium services | RMB 0 (bundled in commission) | No standalone forecasting SKU |
| SAP IBP (enterprise) | $500K–$5M/year | Far above franchise pharmacy budget |
| IQVIA syndicated data | High six figures/year | Manufacturer-focused |

**Recommended pricing for prototype narrative:**
- **Freemium:** Basic expiry alerts + dashboard (free, to drive adoption).
- **Standard:** RMB 200–400/month per pharmacy — includes 30-day demand forecast + auto-replenishment suggestions.
- **Enterprise:** RMB 500–800/month per pharmacy — includes VBP simulation, network-wide reallocation, API access.
- **Platform tier:** 0.5–1% of procurement value saved (contingency pricing) — aligns vendor incentives with customer outcomes.

---

## Preliminary Go-to-Market Sketch

### Channel Strategy
**Integrate as a premium module inside the existing B2B e-commerce platform.** This is the lowest-friction path: pharmacies already log in daily to place orders. Adding a "Smart Replenishment" tab next to "Place Order" requires zero behavior change. The platform benefits from increased stickiness and higher GMV (better forecasting → fewer stockouts → more consistent ordering).

### Pilot Strategy
1. **Phase 1 (Months 1–3):** Pilot with **500–1,000 independent pharmacies** in a single region (e.g., Guangdong or Sichuan). Independents have the worst inventory practices and the most to gain.
2. **Phase 2 (Months 4–6):** Roll out to **franchise chains** (20–100 stores). Chains have centralized procurement managers who can act on network-wide reallocation recommendations.
3. **Phase 3 (Months 7–12):** Scale to full platform user base (340K users). Introduce **VBP Policy Simulator** as a differentiated enterprise feature.

### Key Differentiators
1. **VBP-native:** The only tool that models National Volume-Based Procurement as a first-class shock variable, not an afterthought.
2. **Long-tail-aware:** Uses ADI-segmented models (Croston/SBA/TSB) for the 78% of SKUs that mainstream forecasting ignores.
3. **FEFO-first:** Enforces expiry-aware replenishment, unlike generic retail inventory tools that assume FIFO.
4. **Network optimization:** Recommends inter-pharmacy stock transfers before triggering new orders — turning decentralized inventory into a virtual pooled system.

---

## Bibliography

1. 36Kr Research. (2023). 2023年中国医药电商B2B行业洞察报告. https://www.36kr.com/p/2153883523240200

2. ChemLinked / Baipharm. (2024). China Announces Results of the 10th Round of Volume-Based Drug Procurement (VBP). https://baipharm.chemlinked.com/news/china-announces-results-of-the-10th-round-of-volume-based-drug-procurement-vbp

3. 每经网. (2024). 深度复盘第十批国家药品集采. https://www.nbd.com.cn/articles/2024-12-24/3697123.html

4. 新华网甘肃频道. (2025). 第十批国家集采药品落地张掖市 62种药品惠民降价超60%. http://www.gs.xinhuanet.com/20250527/

5. 定州市人民医院. (2025). 4月1日，第十批国家组织药品集中采购正式落地实施. https://www.dzsrmyy.com/html/gonggao/202504/5173.html

6. 21世纪经济报道. (2020). 集采落选原研药厂求生转型 院外药店市场"迎春". https://m.21jingji.com/article/20200826/1d02e100afc3ae1db37776d64ed27e78.html

7. 财新. (2024). 药耗带量采购步入下半场 如何兼顾各方权益？https://www.caixin.com/2024-09-24/102239137.html

8. 医药联盟. (2023). 集采后，30多款降压药零售渠道销量上涨. http://www.chinamsr.com/2023/1206/126463.shtml

9. 南方+. (2023). 第八批国采陆续落地：多款大品种药物降价超80%，处方外流加速度. https://static.nfapp.southcn.com/content/202307/03/c7854478.html

10. 智研咨询. (2025). 2025年中国医药B2B电商行业发展历程、产业链、销售额、竞争格局及未来趋势研判. https://m.chyxx.com/industry/1220525.html

11. 东方财富网. (2022). 中国医药电商B2B行业研究报告. https://data.eastmoney.com/report/zw_industry.jshtml?infocode=AP202210311579646331

12. 财新. (2023). 医药B2B电商药师帮第三次赴港上市 挂牌首日开盘涨逾15%. https://companies.caixin.com/2023-06-28/102069877.html

13. IQVIA. (2024). Algorithmic Forecasting for the Life Sciences Industry (White Paper). https://www.iqvia.com/-/media/iqvia/pdfs/library/white-papers/algorithmic-forecasting.pdf

14. SAP. (2025). Building AI Ready Life Sciences Enterprise with SAP Business Data Cloud. https://community.sap.com/t5/technology-blog-posts-by-members/building-ai-ready-life-sciences-enterprise-with-sap-business-data-cloud/ba-p/14242504

15. 法伯科技 / Pharbers. (2023). 《VBP政策下的用药分析》. https://www.pharbers.com/newsinfo/4933109.html

16. 中康CMH. (2021). 奇点临近——2021年零售市场8大趋势回顾与解读. https://xueqiu.com/2495339241/210962653

17. 雪球. (2023). 关于药师帮的几个有意思的数据. https://xueqiu.com/3974055088/254566199

18. 网易. (2023). 服务30万家药店，三年亏损21亿，药师帮该如何"自医"？https://www.163.com/dy/article/H8FJ37LS0519875F.html

19. 瑞银. (2024). 第十轮集采药品降价幅度超预期 对中国仿制药市场持审慎态度. https://www.iyiou.com/briefing/202412191709728

20. 中国 CDC / CNIC. (2025). Chinese Weekly Influenza Surveillance Report. https://ivdc.chinacdc.cn/cnic/en/Surveillance/WeeklyReport/

21. 研著集要. (2025). 九批十轮集采"全景扫描". https://mp.weixin.qq.com/s?__biz=MzA4MzAxMTk3NA==

22. 商业新知. (2024). 从历史数据看未来！前9批国家药品集采回顾分析. https://www.shangyexinzhi.com/article/23734174.html

---

*Document generated from multi-source research. VBP batch details for rounds 1–9 are synthesized from industry reports; precise official averages may vary slightly by source. All demand-shift patterns are documented in publicly available trade and financial journalism.*
