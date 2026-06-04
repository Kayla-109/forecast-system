# Intelligent Replenishment Report

**Generated:** 2026-06-04 22:30  
**Lead Time:** 1 week(s)  

---

## Executive Summary

- **Red (urgent):** 15 SKUs
- **Yellow (caution):** 32 SKUs
- **Green (healthy):** 184 SKUs

## ABC / XYZ Distribution

| abc_class   |   X |   Y |   Z |
|:------------|----:|----:|----:|
| A           |  63 |  21 |   0 |
| B           |   0 |  75 |   3 |
| C           |   0 |  21 |  48 |


## Top 10 Urgent Replenishments (Red Risk)

| sku_id   | name_cn   | pharmacy_type   |   current_stock |   days_supply |   recommended_qty | reason   |
|:---------|:----------|:----------------|----------------:|--------------:|------------------:|:---------|
| SKU_0055 | 乙酰半胱氨酸泡腾片 | hospital        |               0 |             0 |                89 | 库存告急     |
| SKU_0044 | 吡格列酮片     | hospital        |               0 |             0 |                75 | 库存告急     |
| SKU_0073 | 艾司西酞普兰片   | hospital        |               0 |             0 |                74 | 库存告急     |
| SKU_0039 | 西格列汀片     | chain           |               0 |             0 |                62 | 库存告急     |
| SKU_0061 | 糠酸莫米松鼻喷雾剂 | chain           |               0 |             0 |                62 | 库存告急     |
| SKU_0057 | 异丙托溴铵气雾剂  | chain           |               0 |             0 |                59 | 库存告急     |
| SKU_0070 | 托吡酯片      | hospital        |               0 |             0 |                58 | 库存告急     |
| SKU_0076 | 米氮平片      | hospital        |               0 |             0 |                56 | 库存告急     |
| SKU_0076 | 米氮平片      | chain           |               0 |             0 |                42 | 库存告急     |
| SKU_0009 | 地高辛片      | independent     |               0 |             0 |                39 | 库存告急     |


## Inter-Pharmacy Transfer Suggestions

| sku_id   | name_cn   | from_pharmacy   | to_pharmacy   |   transfer_qty |   from_stock |   to_stock | reason                           |
|:---------|:----------|:----------------|:--------------|---------------:|-------------:|-----------:|:---------------------------------|
| SKU_0009 | 地高辛片      | hospital        | independent   |             28 |          104 |          0 | hospital过剩(28)→independent缺货(39) |
| SKU_0039 | 西格列汀片     | hospital        | chain         |             30 |          132 |          0 | hospital过剩(30)→chain缺货(62)       |
| SKU_0061 | 糠酸莫米松鼻喷雾剂 | independent     | chain         |             52 |          117 |          0 | independent过剩(52)→chain缺货(62)    |


## Policy Simulation (Engine vs Fixed Threshold, 90 days)

- Fixed threshold avg stockouts: **2.62**
- Engine (R,S) avg stockouts: **0.32**
- Fixed threshold avg inventory: **48.7**
- Engine (R,S) avg inventory: **87.6**
