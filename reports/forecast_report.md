# Demand Forecasting Report

**Generated:** 2026-06-04 21:41  
**Horizon:** 12 weeks  
**Test Period:** Last 12 weeks of historical data  

---

## Executive Summary

### Average Performance by Model

| model   |   mape |   rmse |   smape |   mase |   count |
|:--------|-------:|-------:|--------:|-------:|--------:|
| ETS     |  49.27 |  22.12 |   46.96 |   0.94 |     116 |
| SBA     |  47.99 |  12.69 |  122.52 |   0.91 |     109 |
| XGBoost |  27.79 |  35.6  |   25.76 |   0.96 |       6 |


## Best Performing SKUs (Lowest MAPE)

| sku_id   | name_cn   | pharmacy_type   | model   |   mape |   mase |
|:---------|:----------|:----------------|:--------|-------:|-------:|
| SKU_0032 | 左氧氟沙星片    | hospital        | ETS     |   7.57 |   0.54 |
| SKU_0033 | 二甲双胍片     | hospital        | XGBoost |   8.29 |   0.64 |
| SKU_0034 | 格列美脲片     | hospital        | ETS     |   8.92 |   0.56 |
| SKU_0065 | 左乙拉西坦片    | hospital        | ETS     |   9.27 |   0.46 |
| SKU_0004 | 硝苯地平控释片   | hospital        | ETS     |   9.4  |   0.57 |


## Worst Performing SKUs (Highest MAPE)

| sku_id   | name_cn    | pharmacy_type   | model   |   mape |   mase |
|:---------|:-----------|:----------------|:--------|-------:|-------:|
| SKU_0047 | 卡格列净片      | hospital        | ETS     | 313.19 |   1.47 |
| SKU_0046 | 恩格列净片      | independent     | ETS     | 224.38 |   0.97 |
| SKU_0062 | 吸入用异丙托溴铵溶液 | independent     | ETS     | 194.38 |   1.59 |
| SKU_0009 | 地高辛片       | independent     | ETS     | 131.22 |   0.97 |
| SKU_0047 | 卡格列净片      | independent     | ETS     | 127.72 |   1.56 |


## Demo SKU Performance

| sku_id   | name_cn   | pharmacy_type   | model   |   mape |   rmse |   smape |   mase |
|:---------|:----------|:----------------|:--------|-------:|-------:|--------:|-------:|
| SKU_0020 | 奥司他韦胶囊    | chain           | XGBoost |  31.39 |  49.46 |   33.39 |   1.16 |
| SKU_0020 | 奥司他韦胶囊    | hospital        | XGBoost |  20.91 |  17.96 |   19.26 |   0.79 |
| SKU_0020 | 奥司他韦胶囊    | independent     | XGBoost |  43.34 |  19.77 |   35.84 |   0.91 |
| SKU_0033 | 二甲双胍片     | chain           | XGBoost |  29.43 |  66.88 |   29.86 |   1.52 |
| SKU_0033 | 二甲双胍片     | hospital        | XGBoost |   8.29 |  30.8  |    8.82 |   0.64 |
| SKU_0033 | 二甲双胍片     | independent     | XGBoost |  33.38 |  28.73 |   27.4  |   0.76 |


## XGBoost Feature Importance (Demo SKUs)

## ADI Classification Check

| demand_type_actual   |   sku_id |   mape |   mase |
|:---------------------|---------:|-------:|-------:|
| intermittent         |      109 |  47.99 |   0.91 |
| smooth               |      122 |  48.21 |   0.94 |

