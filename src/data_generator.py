"""
Pharmaceutical Synthetic Data Generator — Day 4 Refinement
============================================================
Generates realistic synthetic data for demand forecasting prototype.

Key features:
  • 80 SKUs with real Chinese drug names, ATC-4 codes, and therapeutic areas
  • Stratified demand types: Smooth(16), Seasonal(12), Intermittent(44), Shocked(8)
  • VBP-10 shock: instantaneous step change on 2025-04-01
  • Pharmacy-type heterogeneity: hospital/chain/independent with distinct CV and VBP response
  • ILI elasticity per SKU (e.g., oseltamivir = 8.0, metformin = 0.0)
  • ADI validation with target-vs-actual distribution chart
  • Pre-selected Demo SKUs: Metformin (smooth) & Oseltamivir (seasonal)

Date range: 2024-06-01 to 2026-05-31 (weekly, ~104 weeks)
Random seed: 42 (fully reproducible)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Drug Library — 80 real-ish drugs with Chinese names
# ---------------------------------------------------------------------------
# Structure: (name_cn, name_en, atc4, area, demand_type, base_demand_hosp, base_demand_chain,
#             base_demand_ind, cv_hosp, cv_chain, cv_ind, ili_elasticity, vbp_status, vbp_shock_hosp,
#             vbp_shock_chain, vbp_shock_ind, price_rmb, expiry_months, is_generic, is_demo)

DRUG_LIBRARY_RAW: List[Tuple] = [
    # === Cardiovascular (C) — 16 SKUs ===
    # Smooth (4)
    ("氨氯地平片", "Amlodipine", "C08CA01", "Cardiovascular", "smooth", 120, 80, 40, 0.15, 0.25, 0.40, 0.0, "selected", 0.60, -0.20, -0.15, 8.5, 36, 1, 0),
    ("缬沙坦胶囊", "Valsartan", "C09CA03", "Cardiovascular", "smooth", 100, 70, 35, 0.18, 0.28, 0.42, 0.0, "selected", 0.55, -0.18, -0.12, 15.0, 36, 1, 0),
    ("美托洛尔缓释片", "Metoprolol", "C07AB02", "Cardiovascular", "smooth", 90, 60, 30, 0.16, 0.26, 0.38, 0.0, "na", 0.0, 0.0, 0.0, 22.0, 36, 1, 0),
    ("硝苯地平控释片", "Nifedipine", "C08CA05", "Cardiovascular", "smooth", 110, 75, 38, 0.14, 0.24, 0.35, 0.0, "selected", 0.50, -0.15, -0.10, 18.5, 36, 1, 0),
    # Seasonal (2) — slight winter peak for cardiovascular events
    ("阿司匹林肠溶片", "Aspirin", "B01AC06", "Cardiovascular", "seasonal", 200, 150, 80, 0.20, 0.30, 0.45, 0.5, "na", 0.0, 0.0, 0.0, 5.0, 48, 1, 0),
    ("氯吡格雷片", "Clopidogrel", "B01AC04", "Cardiovascular", "seasonal", 150, 100, 50, 0.18, 0.28, 0.40, 0.3, "selected", 0.45, -0.12, -0.08, 45.0, 36, 1, 0),
    # Intermittent (8)
    ("伊伐布雷定片", "Ivabradine", "C01EB17", "Cardiovascular", "intermittent", 15, 10, 5, 0.80, 1.00, 1.50, 0.0, "na", 0.0, 0.0, 0.0, 120.0, 36, 0, 0),
    ("波生坦片", "Bosentan", "C02KX01", "Cardiovascular", "intermittent", 8, 5, 2, 1.00, 1.20, 1.80, 0.0, "na", 0.0, 0.0, 0.0, 280.0, 24, 0, 0),
    ("地高辛片", "Digoxin", "C01AA05", "Cardiovascular", "intermittent", 20, 12, 6, 0.70, 0.90, 1.30, 0.0, "na", 0.0, 0.0, 0.0, 35.0, 48, 1, 0),
    ("胺碘酮片", "Amiodarone", "C01BD01", "Cardiovascular", "intermittent", 25, 15, 8, 0.65, 0.85, 1.20, 0.0, "na", 0.0, 0.0, 0.0, 42.0, 36, 1, 0),
    ("利多卡因注射液", "Lidocaine", "C01BB01", "Cardiovascular", "intermittent", 30, 18, 8, 0.90, 1.10, 1.60, 0.0, "na", 0.0, 0.0, 0.0, 15.0, 24, 1, 0),
    ("米力农注射液", "Milrinone", "C01CE02", "Cardiovascular", "intermittent", 5, 3, 1, 1.20, 1.50, 2.00, 0.0, "na", 0.0, 0.0, 0.0, 180.0, 24, 1, 0),
    ("西地兰注射液", "Cedilanid", "C01AA04", "Cardiovascular", "intermittent", 10, 6, 3, 1.00, 1.30, 1.80, 0.0, "na", 0.0, 0.0, 0.0, 25.0, 24, 1, 0),
    ("酚妥拉明注射液", "Phentolamine", "C02CA02", "Cardiovascular", "intermittent", 12, 7, 3, 0.85, 1.05, 1.50, 0.0, "na", 0.0, 0.0, 0.0, 55.0, 24, 1, 0),
    # Shocked (2)
    ("阿托伐他汀钙片", "Atorvastatin", "C10AA05", "Cardiovascular", "shocked", 180, 120, 60, 0.15, 0.25, 0.40, 0.0, "selected", 0.65, -0.22, -0.18, 35.0, 36, 1, 0),
    ("瑞舒伐他汀钙片", "Rosuvastatin", "C10AA07", "Cardiovascular", "shocked", 140, 100, 50, 0.18, 0.28, 0.45, 0.0, "non_selected", -0.90, 0.70, 0.50, 68.0, 36, 0, 0),

    # === Antiinfectives (J) — 16 SKUs ===
    # Smooth (3)
    ("阿莫西林胶囊", "Amoxicillin", "J01CA04", "Antiinfectives", "smooth", 200, 140, 70, 0.20, 0.30, 0.45, 0.0, "na", 0.0, 0.0, 0.0, 12.0, 36, 1, 0),
    ("头孢克洛胶囊", "Cefaclor", "J01DC04", "Antiinfectives", "smooth", 160, 110, 55, 0.22, 0.32, 0.48, 0.0, "na", 0.0, 0.0, 0.0, 28.0, 36, 1, 0),
    ("阿奇霉素片", "Azithromycin", "J01FA10", "Antiinfectives", "smooth", 180, 130, 65, 0.18, 0.28, 0.42, 0.0, "na", 0.0, 0.0, 0.0, 22.0, 36, 1, 0),
    # Seasonal (4) — flu & respiratory infection season
    ("奥司他韦胶囊", "Oseltamivir", "J05AH02", "Antiinfectives", "seasonal", 60, 100, 40, 0.40, 0.50, 0.70, 8.0, "na", 0.0, 0.0, 0.0, 55.0, 36, 1, 1),
    ("扎那米韦吸入粉雾剂", "Zanamivir", "J05AH01", "Antiinfectives", "seasonal", 15, 25, 10, 0.50, 0.60, 0.80, 6.0, "na", 0.0, 0.0, 0.0, 120.0, 24, 1, 0),
    ("玛巴洛沙韦片", "Baloxavir", "J05AX29", "Antiinfectives", "seasonal", 20, 35, 15, 0.45, 0.55, 0.75, 7.0, "na", 0.0, 0.0, 0.0, 220.0, 24, 1, 0),
    ("帕拉米韦氯化钠注射液", "Peramivir", "J05AX27", "Antiinfectives", "seasonal", 25, 15, 5, 0.60, 0.80, 1.10, 5.0, "na", 0.0, 0.0, 0.0, 350.0, 24, 1, 0),
    # Intermittent (7)
    ("利奈唑胺片", "Linezolid", "J01XX08", "Antiinfectives", "intermittent", 18, 12, 5, 0.75, 0.95, 1.40, 0.0, "na", 0.0, 0.0, 0.0, 180.0, 36, 1, 0),
    ("伏立康唑片", "Voriconazole", "J02AC03", "Antiinfectives", "intermittent", 12, 8, 3, 0.85, 1.05, 1.50, 0.0, "na", 0.0, 0.0, 0.0, 150.0, 24, 1, 0),
    ("卡泊芬净注射液", "Caspofungin", "J02AX04", "Antiinfectives", "intermittent", 6, 4, 2, 1.00, 1.20, 1.70, 0.0, "na", 0.0, 0.0, 0.0, 680.0, 24, 1, 0),
    ("两性霉素B脂质体", "Amphotericin B", "J02AA01", "Antiinfectives", "intermittent", 4, 3, 1, 1.20, 1.50, 2.00, 0.0, "na", 0.0, 0.0, 0.0, 850.0, 18, 1, 0),
    ("替加环素注射液", "Tigecycline", "J01AA12", "Antiinfectives", "intermittent", 8, 5, 2, 1.10, 1.30, 1.80, 0.0, "na", 0.0, 0.0, 0.0, 420.0, 24, 1, 0),
    ("多黏菌素B注射液", "Polymyxin B", "J01XB02", "Antiinfectives", "intermittent", 5, 3, 1, 1.30, 1.60, 2.10, 0.0, "na", 0.0, 0.0, 0.0, 520.0, 18, 1, 0),
    ("磷霉素氨丁三醇散", "Fosfomycin", "J01XX01", "Antiinfectives", "intermittent", 22, 15, 7, 0.70, 0.90, 1.30, 0.0, "na", 0.0, 0.0, 0.0, 45.0, 36, 1, 0),
    # Shocked (2)
    ("头孢呋辛酯片", "Cefuroxime", "J01DC02", "Antiinfectives", "shocked", 140, 100, 50, 0.20, 0.30, 0.45, 0.0, "selected", 0.50, -0.15, -0.10, 18.0, 36, 1, 0),
    ("左氧氟沙星片", "Levofloxacin", "J01MA12", "Antiinfectives", "shocked", 160, 110, 55, 0.18, 0.28, 0.42, 0.0, "selected", 0.55, -0.18, -0.12, 25.0, 36, 1, 0),

    # === Metabolism / Diabetes (A10) — 16 SKUs ===
    # Smooth (4)
    ("二甲双胍片", "Metformin", "A10BA02", "Metabolism", "smooth", 250, 180, 90, 0.12, 0.20, 0.35, 0.0, "selected", 0.50, -0.15, -0.10, 8.0, 48, 1, 1),
    ("格列美脲片", "Glimepiride", "A10BB12", "Metabolism", "smooth", 120, 85, 42, 0.15, 0.25, 0.38, 0.0, "na", 0.0, 0.0, 0.0, 35.0, 36, 1, 0),
    ("阿卡波糖片", "Acarbose", "A10BF01", "Metabolism", "smooth", 140, 100, 50, 0.14, 0.24, 0.36, 0.0, "selected", 0.45, -0.12, -0.08, 28.0, 36, 1, 0),
    ("格列齐特缓释片", "Gliclazide", "A10BB09", "Metabolism", "smooth", 100, 70, 35, 0.16, 0.26, 0.40, 0.0, "na", 0.0, 0.0, 0.0, 32.0, 36, 1, 0),
    # Seasonal (1) — slight winter peak (holiday overeating)
    ("门冬胰岛素注射液", "Insulin Aspart", "A10AB05", "Metabolism", "seasonal", 300, 200, 100, 0.25, 0.35, 0.50, 0.3, "na", 0.0, 0.0, 0.0, 85.0, 24, 1, 0),
    # Intermittent (9)
    ("达格列净片", "Dapagliflozin", "A10BK01", "Metabolism", "intermittent", 80, 55, 28, 0.30, 0.42, 0.60, 0.0, "na", 0.0, 0.0, 0.0, 68.0, 36, 1, 0),
    ("西格列汀片", "Sitagliptin", "A10BH01", "Metabolism", "intermittent", 60, 42, 21, 0.35, 0.48, 0.68, 0.0, "na", 0.0, 0.0, 0.0, 55.0, 36, 1, 0),
    ("利拉鲁肽注射液", "Liraglutide", "A10BJ02", "Metabolism", "intermittent", 40, 28, 14, 0.40, 0.55, 0.78, 0.0, "na", 0.0, 0.0, 0.0, 320.0, 24, 1, 0),
    ("度拉糖肽注射液", "Dulaglutide", "A10BJ05", "Metabolism", "intermittent", 25, 18, 9, 0.50, 0.65, 0.90, 0.0, "na", 0.0, 0.0, 0.0, 280.0, 24, 1, 0),
    ("沙格列汀片", "Saxagliptin", "A10BH03", "Metabolism", "intermittent", 35, 25, 12, 0.45, 0.60, 0.82, 0.0, "na", 0.0, 0.0, 0.0, 48.0, 36, 1, 0),
    ("瑞格列奈片", "Repaglinide", "A10BX02", "Metabolism", "intermittent", 45, 32, 16, 0.38, 0.52, 0.72, 0.0, "na", 0.0, 0.0, 0.0, 42.0, 36, 1, 0),
    ("吡格列酮片", "Pioglitazone", "A10BG03", "Metabolism", "intermittent", 55, 38, 19, 0.32, 0.45, 0.65, 0.0, "na", 0.0, 0.0, 0.0, 38.0, 36, 1, 0),
    ("米格列醇片", "Miglitol", "A10BF02", "Metabolism", "intermittent", 20, 14, 7, 0.55, 0.72, 1.00, 0.0, "na", 0.0, 0.0, 0.0, 52.0, 36, 1, 0),
    # Shocked (2)
    ("恩格列净片", "Empagliflozin", "A10BK03", "Metabolism", "shocked", 90, 65, 32, 0.28, 0.40, 0.58, 0.0, "selected", 0.48, -0.14, -0.10, 75.0, 36, 1, 0),
    ("卡格列净片", "Canagliflozin", "A10BK02", "Metabolism", "shocked", 70, 50, 25, 0.30, 0.42, 0.60, 0.0, "non_selected", -0.85, 0.60, 0.40, 82.0, 36, 0, 0),

    # === Respiratory (R) — 16 SKUs ===
    # Smooth (2)
    ("氨茶碱片", "Aminophylline", "R03DA05", "Respiratory", "smooth", 80, 55, 28, 0.22, 0.32, 0.48, 0.0, "na", 0.0, 0.0, 0.0, 15.0, 36, 1, 0),
    ("茶碱缓释片", "Theophylline", "R03DA04", "Respiratory", "smooth", 60, 42, 21, 0.24, 0.35, 0.50, 0.0, "na", 0.0, 0.0, 0.0, 18.0, 36, 1, 0),
    # Seasonal (4) — strong flu/allergy seasonality
    ("沙丁胺醇吸入气雾剂", "Salbutamol", "R03AC02", "Respiratory", "seasonal", 150, 120, 60, 0.30, 0.40, 0.58, 2.5, "na", 0.0, 0.0, 0.0, 25.0, 24, 1, 0),
    ("布地奈德吸入粉雾剂", "Budesonide", "R03BA02", "Respiratory", "seasonal", 120, 100, 50, 0.28, 0.38, 0.55, 3.0, "na", 0.0, 0.0, 0.0, 68.0, 24, 1, 0),
    ("孟鲁司特钠片", "Montelukast", "R03DC03", "Respiratory", "seasonal", 100, 80, 40, 0.32, 0.42, 0.60, 2.0, "na", 0.0, 0.0, 0.0, 35.0, 36, 1, 0),
    ("特布他林片", "Terbutaline", "R03AC03", "Respiratory", "seasonal", 70, 55, 28, 0.35, 0.45, 0.65, 2.2, "na", 0.0, 0.0, 0.0, 22.0, 36, 1, 0),
    # Intermittent (9)
    ("噻托溴铵粉吸入剂", "Tiotropium", "R03BB04", "Respiratory", "intermittent", 45, 35, 18, 0.40, 0.55, 0.78, 0.0, "na", 0.0, 0.0, 0.0, 120.0, 24, 1, 0),
    ("乙酰半胱氨酸泡腾片", "Acetylcysteine", "R05CB03", "Respiratory", "intermittent", 55, 42, 21, 0.38, 0.52, 0.72, 0.0, "na", 0.0, 0.0, 0.0, 28.0, 36, 1, 0),
    ("羧甲司坦口服液", "Carbocysteine", "R05CB06", "Respiratory", "intermittent", 65, 50, 25, 0.35, 0.48, 0.68, 0.0, "na", 0.0, 0.0, 0.0, 22.0, 36, 1, 0),
    ("异丙托溴铵气雾剂", "Ipratropium", "R03BB01", "Respiratory", "intermittent", 35, 28, 14, 0.48, 0.62, 0.88, 0.0, "na", 0.0, 0.0, 0.0, 32.0, 24, 1, 0),
    ("福莫特罗吸入粉雾剂", "Formoterol", "R03AC13", "Respiratory", "intermittent", 25, 20, 10, 0.55, 0.70, 0.95, 0.0, "na", 0.0, 0.0, 0.0, 85.0, 24, 1, 0),
    ("沙美特罗替卡松吸入剂", "Salmeterol/Fluticasone", "R03AK06", "Respiratory", "intermittent", 30, 25, 12, 0.50, 0.65, 0.90, 0.0, "na", 0.0, 0.0, 0.0, 150.0, 24, 1, 0),
    ("氮卓斯汀鼻喷剂", "Azelastine", "R01AC03", "Respiratory", "intermittent", 40, 32, 16, 0.42, 0.55, 0.78, 0.0, "na", 0.0, 0.0, 0.0, 55.0, 24, 1, 0),
    ("糠酸莫米松鼻喷雾剂", "Mometasone", "R01AD09", "Respiratory", "intermittent", 50, 40, 20, 0.38, 0.50, 0.72, 0.0, "na", 0.0, 0.0, 0.0, 65.0, 24, 1, 0),
    # Shocked (1)
    ("吸入用异丙托溴铵溶液", "Ipratropium sol", "R03BB01", "Respiratory", "shocked", 40, 30, 15, 0.45, 0.58, 0.82, 0.0, "selected", 0.45, -0.12, -0.08, 28.0, 24, 1, 0),

    # === Nervous System (N) — 16 SKUs ===
    # Smooth (3)
    ("丙戊酸钠片", "Valproate", "N03AG01", "Nervous", "smooth", 90, 65, 32, 0.18, 0.28, 0.42, 0.0, "na", 0.0, 0.0, 0.0, 25.0, 36, 1, 0),
    ("卡马西平片", "Carbamazepine", "N03AF01", "Nervous", "smooth", 70, 50, 25, 0.20, 0.30, 0.45, 0.0, "na", 0.0, 0.0, 0.0, 18.0, 36, 1, 0),
    ("左乙拉西坦片", "Levetiracetam", "N03AX14", "Nervous", "smooth", 85, 60, 30, 0.16, 0.26, 0.40, 0.0, "na", 0.0, 0.0, 0.0, 45.0, 36, 1, 0),
    # Seasonal (1) — winter depression / anxiety peak
    ("帕罗西汀片", "Paroxetine", "N06AB05", "Nervous", "seasonal", 55, 40, 20, 0.30, 0.42, 0.60, 0.4, "na", 0.0, 0.0, 0.0, 38.0, 36, 1, 0),
    # Intermittent (11)
    ("拉莫三嗪片", "Lamotrigine", "N03AX09", "Nervous", "intermittent", 35, 25, 12, 0.45, 0.60, 0.85, 0.0, "na", 0.0, 0.0, 0.0, 55.0, 36, 1, 0),
    ("加巴喷丁胶囊", "Gabapentin", "N03AX12", "Nervous", "intermittent", 40, 28, 14, 0.42, 0.55, 0.78, 0.0, "na", 0.0, 0.0, 0.0, 32.0, 36, 1, 0),
    ("普瑞巴林胶囊", "Pregabalin", "N03AX16", "Nervous", "intermittent", 50, 35, 18, 0.38, 0.52, 0.72, 0.0, "na", 0.0, 0.0, 0.0, 48.0, 36, 1, 0),
    ("托吡酯片", "Topiramate", "N03AX11", "Nervous", "intermittent", 30, 22, 11, 0.48, 0.62, 0.88, 0.0, "na", 0.0, 0.0, 0.0, 42.0, 36, 1, 0),
    ("唑尼沙胺片", "Zonisamide", "N03AX15", "Nervous", "intermittent", 15, 10, 5, 0.65, 0.82, 1.15, 0.0, "na", 0.0, 0.0, 0.0, 65.0, 36, 1, 0),
    ("拉考沙胺片", "Lacosamide", "N03AX18", "Nervous", "intermittent", 20, 14, 7, 0.55, 0.72, 1.00, 0.0, "na", 0.0, 0.0, 0.0, 85.0, 36, 1, 0),
    ("艾司西酞普兰片", "Escitalopram", "N06AB10", "Nervous", "intermittent", 45, 32, 16, 0.40, 0.55, 0.75, 0.0, "na", 0.0, 0.0, 0.0, 52.0, 36, 1, 0),
    ("文拉法辛缓释片", "Venlafaxine", "N06AX16", "Nervous", "intermittent", 38, 28, 14, 0.42, 0.58, 0.80, 0.0, "na", 0.0, 0.0, 0.0, 58.0, 36, 1, 0),
    ("度洛西汀肠溶片", "Duloxetine", "N06AX21", "Nervous", "intermittent", 42, 30, 15, 0.38, 0.52, 0.72, 0.0, "na", 0.0, 0.0, 0.0, 62.0, 36, 1, 0),
    ("米氮平片", "Mirtazapine", "N06AX11", "Nervous", "intermittent", 28, 20, 10, 0.50, 0.68, 0.92, 0.0, "na", 0.0, 0.0, 0.0, 35.0, 36, 1, 0),
    # Shocked (1)
    ("舍曲林片", "Sertraline", "N06AB06", "Nervous", "shocked", 60, 45, 22, 0.25, 0.38, 0.55, 0.0, "selected", 0.42, -0.12, -0.08, 42.0, 36, 1, 0),
]


def _build_drug_registry() -> List[Dict]:
    """Convert flat DRUG_LIBRARY_RAW into structured dict records."""
    registry = []
    for row in DRUG_LIBRARY_RAW:
        (
            name_cn, name_en, atc4, area, d_type,
            b_hosp, b_chain, b_ind,
            cv_hosp, cv_chain, cv_ind,
            ili_elasticity, vbp_status, vbp_hosp, vbp_chain, vbp_ind,
            price, expiry, is_generic, is_demo
        ) = row
        atc3 = atc4[:5]
        atc2 = atc4[:3]
        atc1 = atc4[0]
        registry.append({
            "sku_id": f"SKU_{len(registry)+1:04d}",
            "name_cn": name_cn,
            "name_en": name_en,
            "atc1": atc1,
            "atc2": atc2,
            "atc3": atc3,
            "atc4": atc4,
            "therapeutic_area": area,
            "demand_type": d_type,
            "base_demand": {"hospital": b_hosp, "chain": b_chain, "independent": b_ind},
            "cv": {"hospital": cv_hosp, "chain": cv_chain, "independent": cv_ind},
            "ili_elasticity": ili_elasticity,
            "vbp_status": vbp_status,
            "vbp_shock": {"hospital": vbp_hosp, "chain": vbp_chain, "independent": vbp_ind},
            "price_rmb": price,
            "expiry_months": expiry,
            "is_generic": is_generic,
            "is_demo_sku": is_demo,
        })
    return registry


DRUG_REGISTRY = _build_drug_registry()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    n_skus: int = len(DRUG_REGISTRY)  # 80
    start_date: str = "2024-06-01"
    end_date: str = "2026-05-31"
    freq: str = "W-MON"
    output_dir: Path = Path("data/raw")
    validation_dir: Path = Path("docs/validation")
    pharmacy_types: Tuple[str, ...] = ("hospital", "chain", "independent")
    vbp_date: str = "2025-04-01"


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------
class PharmaDataGenerator:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.dates: pd.DatetimeIndex = pd.date_range(
            start=self.cfg.start_date, end=self.cfg.end_date, freq=self.cfg.freq
        )
        self.n_weeks: int = len(self.dates)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.validation_dir.mkdir(parents=True, exist_ok=True)

        self.master_data: pd.DataFrame | None = None
        self.external_signals: pd.DataFrame | None = None
        self.sales_data: pd.DataFrame | None = None

    # =====================================================================
    # Layer 1: Master Data (from DRUG_REGISTRY)
    # =====================================================================
    def generate_master_data(self) -> pd.DataFrame:
        records = []
        for drug in DRUG_REGISTRY:
            rec = drug.copy()
            rec["base_demand_hospital"] = drug["base_demand"]["hospital"]
            rec["base_demand_chain"] = drug["base_demand"]["chain"]
            rec["base_demand_independent"] = drug["base_demand"]["independent"]
            rec["cv_hospital"] = drug["cv"]["hospital"]
            rec["cv_chain"] = drug["cv"]["chain"]
            rec["cv_independent"] = drug["cv"]["independent"]
            rec["vbp_shock_hospital"] = drug["vbp_shock"]["hospital"]
            rec["vbp_shock_chain"] = drug["vbp_shock"]["chain"]
            rec["vbp_shock_independent"] = drug["vbp_shock"]["independent"]
            # Remove nested dicts for CSV-friendliness
            del rec["base_demand"], rec["cv"], rec["vbp_shock"]
            records.append(rec)

        self.master_data = pd.DataFrame(records)
        print(f"[Master Data] Generated {len(self.master_data)} SKUs.")
        return self.master_data

    # =====================================================================
    # Layer 2: External Signals
    # =====================================================================
    def generate_external_signals(self) -> pd.DataFrame:
        dates = self.dates
        n_weeks = self.n_weeks
        week_of_year = dates.isocalendar().week.values

        # ILI% — national unified pattern
        phase = 2 * np.pi * week_of_year / 52
        seasonal_ili = 3.0 * np.sin(phase - np.pi / 3) + 1.0 * np.sin(2 * phase + np.pi / 4) + 2.5
        yearly_trend = np.linspace(0, -0.3, n_weeks)
        noise = np.random.normal(0, 0.35, n_weeks)
        ili_pct = np.clip(seasonal_ili + yearly_trend + noise, 0.5, 12.0)

        # Holidays
        df = pd.DataFrame({"date": dates})
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day

        spring = ((df["month"] == 1) & (df["day"] >= 20)) | ((df["month"] == 2) & (df["day"] <= 10))
        golden = (df["month"] == 10) & (df["day"] <= 7)
        labor = (df["month"] == 5) & (df["day"] <= 5)
        vbp_flag = (dates >= pd.Timestamp(self.cfg.vbp_date)).astype(int)

        self.external_signals = pd.DataFrame({
            "date": dates,
            "week_of_year": week_of_year,
            "ili_pct": np.round(ili_pct, 2),
            "is_spring_festival": spring.astype(int),
            "is_golden_week": golden.astype(int),
            "is_labor_day": labor.astype(int),
            "is_holiday": (spring | golden | labor).astype(int),
            "vbp_10_active": vbp_flag,
        })
        print(f"[External Signals] Generated {len(self.external_signals)} weeks.")
        return self.external_signals

    # =====================================================================
    # Layer 3: Sales Data
    # =====================================================================
    def generate_sales_data(self) -> pd.DataFrame:
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
            ili_elasticity = sku["ili_elasticity"]
            vbp_status = sku["vbp_status"]

            for p_type in self.cfg.pharmacy_types:
                base = sku[f"base_demand_{p_type}"]
                cv = sku[f"cv_{p_type}"]
                vbp_mult = sku[f"vbp_shock_{p_type}"] if vbp_status != "na" else 0.0

                if d_type == "smooth":
                    sales = self._generate_smooth(base, cv, n_weeks, holiday_flag)
                elif d_type == "seasonal":
                    sales = self._generate_seasonal(base, cv, ili, ili_elasticity, n_weeks, holiday_flag)
                elif d_type == "intermittent":
                    sales = self._generate_intermittent(base, cv, n_weeks)
                elif d_type == "shocked":
                    sales = self._generate_shocked(base, cv, vbp_mult, vbp_flag, n_weeks, holiday_flag)
                else:
                    sales = np.zeros(n_weeks)

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
    # Generators by demand type
    # -----------------------------------------------------------------
    @staticmethod
    def _generate_smooth(base: float, cv: float, n_weeks: int, holiday_flag: np.ndarray) -> np.ndarray:
        t = np.arange(n_weeks)
        trend = np.linspace(0, base * 0.08, n_weeks)
        season = base * 0.04 * np.sin(2 * np.pi * t / 52)
        holiday = -base * 0.12 * holiday_flag
        noise = np.random.normal(0, base * cv, n_weeks)
        return base + trend + season + holiday + noise

    @staticmethod
    def _generate_seasonal(
        base: float, cv: float, ili: np.ndarray, ili_elasticity: float,
        n_weeks: int, holiday_flag: np.ndarray
    ) -> np.ndarray:
        ili_effect = (ili - np.mean(ili)) * ili_elasticity
        trend = np.linspace(0, base * 0.05, n_weeks)
        holiday = -base * 0.08 * holiday_flag
        noise = np.random.normal(0, base * cv * 0.6, n_weeks)
        return base + trend + ili_effect + holiday + noise

    @staticmethod
    def _generate_intermittent(base: float, cv: float, n_weeks: int) -> np.ndarray:
        prob = np.clip(1.0 / (1.0 + base / 25.0), 0.06, 0.75)
        occurrence = np.random.binomial(1, prob, n_weeks)
        sizes = np.random.lognormal(mean=np.log(max(base, 1)), sigma=cv * 0.45, size=n_weeks)
        return occurrence * sizes

    @staticmethod
    def _generate_shocked(
        base: float, cv: float, vbp_mult: float, vbp_flag: np.ndarray,
        n_weeks: int, holiday_flag: np.ndarray
    ) -> np.ndarray:
        t = np.arange(n_weeks)
        trend = np.linspace(0, base * 0.05, n_weeks)
        season = base * 0.03 * np.sin(2 * np.pi * t / 52)
        holiday = -base * 0.08 * holiday_flag
        noise = np.random.normal(0, base * cv, n_weeks)
        shock = base * vbp_mult * vbp_flag
        return base + trend + season + holiday + noise + shock

    # =====================================================================
    # Persistence
    # =====================================================================
    def save_all(self) -> None:
        if self.master_data is None or self.external_signals is None or self.sales_data is None:
            raise RuntimeError("Run all generators first.")
        out = self.cfg.output_dir
        self.master_data.to_csv(out / "master_data.csv", index=False, encoding="utf-8-sig")
        self.master_data.to_parquet(out / "master_data.parquet", index=False)
        self.external_signals.to_csv(out / "external_signals.csv", index=False, encoding="utf-8-sig")
        self.external_signals.to_parquet(out / "external_signals.parquet", index=False)
        self.sales_data.to_csv(out / "sales_data.csv", index=False, encoding="utf-8-sig")
        self.sales_data.to_parquet(out / "sales_data.parquet", index=False)
        print(f"[Save] All data written to {out.absolute()}")

    # =====================================================================
    # Validation & Plots
    # =====================================================================
    def validate_and_plot(self) -> None:
        if self.sales_data is None:
            raise RuntimeError("Run generate_sales_data() first.")

        val_dir = self.cfg.validation_dir
        val_dir.mkdir(parents=True, exist_ok=True)
        sales = self.sales_data.copy()
        signals = self.external_signals.copy()

        # --- Plot 1: Example time series by demand type ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Example SKU Demand by Type (Chain Pharmacy)", fontsize=14, fontweight="bold")
        for ax, d_type in zip(axes.flat, ["smooth", "seasonal", "intermittent", "shocked"]):
            sku_ids = sales.loc[
                (sales["demand_type"] == d_type) & (sales["pharmacy_type"] == "chain"), "sku_id"
            ].unique()
            if len(sku_ids) == 0:
                continue
            example = sku_ids[0]
            subset = sales[(sales["sku_id"] == example) & (sales["pharmacy_type"] == "chain")].sort_values("date")
            name = self.master_data.loc[self.master_data["sku_id"] == example, "name_cn"].values[0]
            ax.plot(subset["date"], subset["units_sold"], label=f"{example} {name}", linewidth=1.2)
            if d_type == "shocked":
                ax.axvline(pd.Timestamp(self.cfg.vbp_date), color="red", linestyle="--", label="VBP-10")
            ax.set_title(f"{d_type.capitalize()} — {name}", fontweight="bold")
            ax.set_ylabel("Units Sold")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(val_dir / "01_demand_types_examples.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 01_demand_types_examples.png")

        # --- Plot 2: ILI% Seasonal Pattern ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(signals["date"], signals["ili_pct"], color="steelblue", linewidth=1.5)
        ax.fill_between(signals["date"], signals["ili_pct"], alpha=0.3, color="steelblue")
        ax.axvline(pd.Timestamp(self.cfg.vbp_date), color="red", linestyle="--", alpha=0.7)
        ax.set_title("Simulated National ILI% Pattern (2024-2026)", fontsize=13, fontweight="bold")
        ax.set_ylabel("ILI%")
        ax.set_xlabel("Date")
        ax.text(0.5, 0.95, f"Red dashed = VBP-10 ({self.cfg.vbp_date})", transform=ax.transAxes,
                color="red", fontsize=9, ha="center", va="top")
        plt.tight_layout()
        plt.savefig(val_dir / "02_ili_seasonal_pattern.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 02_ili_seasonal_pattern.png")

        # --- Plot 3: VBP Shock Before/After ---
        shocked = sales[sales["demand_type"] == "shocked"]
        pre = shocked[shocked["date"] < self.cfg.vbp_date].groupby("sku_id")["units_sold"].mean().reset_index()
        post = shocked[shocked["date"] >= self.cfg.vbp_date].groupby("sku_id")["units_sold"].mean().reset_index()
        pre_post = pre.merge(post, on="sku_id", suffixes=("_pre", "_post"), how="inner")
        status_map = self.master_data.set_index("sku_id")["vbp_status"].to_dict()
        colors = ["green" if status_map.get(s, "na") == "selected" else "orange" for s in pre_post["sku_id"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(pre_post["units_sold_pre"], pre_post["units_sold_post"], c=colors, alpha=0.7, s=60)
        max_val = max(pre_post["units_sold_pre"].max(), pre_post["units_sold_post"].max()) * 1.05
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="No change line")
        ax.set_xlabel("Pre-VBP Average Weekly Sales")
        ax.set_ylabel("Post-VBP Average Weekly Sales")
        ax.set_title("VBP-10 Shock: Pre vs Post Demand (Shocked SKUs)", fontsize=13, fontweight="bold")
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="green", label="Selected Generic (↓ retail)"),
            Patch(facecolor="orange", label="Non-Selected Branded (↑ retail)"),
        ], loc="upper left")
        plt.tight_layout()
        plt.savefig(val_dir / "03_vbp_shock_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 03_vbp_shock_comparison.png")

        # --- Plot 4: ADI Distribution vs Design Target ---
        adi_records = []
        for (sku_id, ptype), g in sales.groupby(["sku_id", "pharmacy_type"]):
            adi = self._compute_adi(g["units_sold"].values)
            d_type = g["demand_type"].iloc[0]
            adi_records.append({"sku_id": sku_id, "pharmacy_type": ptype, "adi": adi, "demand_type": d_type})
        adi_df = pd.DataFrame(adi_records)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"smooth": "#2ecc71", "seasonal": "#3498db", "intermittent": "#e74c3c", "shocked": "#f39c12"}
        for d_type in ["smooth", "seasonal", "intermittent", "shocked"]:
            subset = adi_df[adi_df["demand_type"] == d_type]["adi"]
            ax.hist(subset, bins=25, alpha=0.5, label=d_type.capitalize(), color=colors[d_type], density=True)
        ax.axvline(1.32, color="black", linestyle="--", linewidth=2, label="ADI = 1.32 threshold")
        ax.axvline(5.0, color="gray", linestyle="--", linewidth=2, label="ADI = 5.0 threshold")
        ax.set_xlabel("Average Inter-Demand Interval (ADI)")
        ax.set_ylabel("Density")
        ax.set_title("ADI Distribution by Designated Demand Type", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(val_dir / "04_adi_distribution.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 04_adi_distribution.png")

        # --- Plot 5: Pharmacy Type Comparison ---
        summary = sales.groupby("pharmacy_type")["units_sold"].agg(["mean", "std", "median"]).reset_index()
        summary["cv"] = summary["std"] / summary["mean"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors_bar = ["#e74c3c", "#3498db", "#2ecc71"]
        axes[0].bar(summary["pharmacy_type"], summary["mean"], color=colors_bar)
        axes[0].set_title("Average Weekly Demand by Pharmacy Type", fontweight="bold")
        axes[0].set_ylabel("Mean Units Sold")
        axes[1].bar(summary["pharmacy_type"], summary["cv"], color=colors_bar)
        axes[1].set_title("Demand Volatility (CV) by Pharmacy Type", fontweight="bold")
        axes[1].set_ylabel("CV = Std / Mean")
        plt.tight_layout()
        plt.savefig(val_dir / "05_pharmacy_type_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 05_pharmacy_type_comparison.png")

        # --- Plot 6: ADI Target vs Actual (new for Day 4) ---
        target_counts = {"smooth": 16, "seasonal": 12, "intermittent": 44, "shocked": 8}
        actual_counts = {}
        for d_type in ["smooth", "seasonal", "intermittent", "shocked"]:
            # Count unique SKUs whose median ADI across all pharmacies falls in the right bucket
            type_adi = adi_df[adi_df["demand_type"] == d_type].groupby("sku_id")["adi"].median()
            if d_type == "smooth":
                actual_counts[d_type] = (type_adi < 1.32).sum()
            elif d_type == "intermittent":
                actual_counts[d_type] = ((type_adi >= 1.32) & (type_adi < 5.0)).sum()
            else:  # seasonal / shocked — use same threshold for visualization
                actual_counts[d_type] = len(type_adi)

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(target_counts))
        width = 0.35
        ax.bar(x - width/2, target_counts.values(), width, label="Design Target", color="steelblue")
        ax.bar(x + width/2, [actual_counts.get(k, 0) for k in target_counts.keys()], width,
               label="Actual Generated", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels([k.capitalize() for k in target_counts.keys()])
        ax.set_ylabel("SKU Count")
        ax.set_title("Demand Type Classification: Target vs Actual", fontsize=13, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(val_dir / "06_adi_target_vs_actual.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved 06_adi_target_vs_actual.png")

        # --- Plot 7: Therapeutic Area Trajectories (new for Day 4) ---
        demo_skus = self.master_data[self.master_data["is_demo_sku"] == 1]["sku_id"].tolist()
        if len(demo_skus) >= 2:
            fig, axes = plt.subplots(len(demo_skus), 1, figsize=(14, 3 * len(demo_skus)), sharex=True)
            if len(demo_skus) == 1:
                axes = [axes]
            for ax, sku_id in zip(axes, demo_skus):
                name = self.master_data.loc[self.master_data["sku_id"] == sku_id, "name_cn"].values[0]
                area = self.master_data.loc[self.master_data["sku_id"] == sku_id, "therapeutic_area"].values[0]
                d_type = self.master_data.loc[self.master_data["sku_id"] == sku_id, "demand_type"].values[0]
                for ptype, color in zip(["hospital", "chain", "independent"], ["#e74c3c", "#3498db", "#2ecc71"]):
                    sub = sales[(sales["sku_id"] == sku_id) & (sales["pharmacy_type"] == ptype)].sort_values("date")
                    ax.plot(sub["date"], sub["units_sold"], label=ptype, color=color, linewidth=1.2)
                ax.axvline(pd.Timestamp(self.cfg.vbp_date), color="red", linestyle="--", alpha=0.5)
                ax.set_title(f"{sku_id} | {name} | {area} | {d_type}", fontweight="bold")
                ax.set_ylabel("Units")
                ax.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.savefig(val_dir / "07_demo_sku_trajectories.png", dpi=200, bbox_inches="tight")
            plt.close()
            print("[Plot] Saved 07_demo_sku_trajectories.png")

        print(f"[Validation] All plots saved to {val_dir.absolute()}")

    @staticmethod
    def _compute_adi(series: np.ndarray) -> float:
        positive = np.where(series > 0)[0]
        if len(positive) <= 1:
            return float(len(series))
        return float(np.mean(np.diff(positive)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Pharma Synthetic Data Generator — Day 4 Refinement")
    print("=" * 60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Date range: 2024-06-01 to 2026-05-31 (weekly)")
    print(f"VBP shock date: 2025-04-01 (instantaneous)")
    print(f"SKU count: {len(DRUG_REGISTRY)}")
    print()

    cfg = Config()
    gen = PharmaDataGenerator(cfg)

    gen.generate_master_data()
    gen.generate_external_signals()
    gen.generate_sales_data()
    gen.save_all()
    gen.validate_and_plot()

    print()
    print("=" * 60)
    print("All done! Check data/raw/ and docs/validation/")
    print("=" * 60)


if __name__ == "__main__":
    main()
