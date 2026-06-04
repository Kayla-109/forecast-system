# Day 7 Summary Report — SKU-Level Demo Preparation

**Generated:** 2026-06-04 22:54

---

## 1. XGBoost Hyperparameter Tuning

### SKU_0020 — 奥司他韦胶囊
- **Best Params:** n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8
- **Best Validation MAPE:** 17.77%
- **Pharmacy Type Tuned:** hospital

### SKU_0033 — 二甲双胍片
- **Best Params:** n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9
- **Best Validation MAPE:** 12.00%
- **Pharmacy Type Tuned:** hospital


## 2. 12-Week Walk-Forward Backtest

### SKU_0020 — 奥司他韦胶囊
- **MAPE:** 31.31%
- **SMAPE:** 33.92%
- **MASE:** 0.93
- **Records:** 12

### SKU_0033 — 二甲双胍片
- **MAPE:** 27.00%
- **SMAPE:** 28.00%
- **MASE:** 0.88
- **Records:** 12


## 3. Inventory Health Diagnosis

- **Red expiry risk:** 9 SKUs
- **Yellow expiry risk:** 29 SKUs
- **Green expiry risk:** 193 SKUs
- **Average expiry score:** 0.154


## 4. Conclusion

- Demo SKU XGBoost models tuned with small grid search.
- Walk-forward backtest covers last 12 weeks with 90% confidence bands.
- Inventory health dashboard highlights expiry risks and ABC/XYZ segmentation.
