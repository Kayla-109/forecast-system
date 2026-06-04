"""
Internationalization (i18n) for Streamlit App
Default language: English
"""

TRANSLATIONS = {
    "en": {
        # App
        "app_title": "Pharma Demand Forecasting & Intelligent Replenishment",
        "app_subtitle": "AI-powered inventory optimization for China's pharmaceutical B2B network",
        "sidebar_nav": "📍 Navigation",
        "language_label": "Language / 语言",

        # Pages
        "page_home": "🏠 Home",
        "page_forecast": "📈 Demand Forecast",
        "page_replenishment": "📦 Replenishment",
        "page_inventory": "🏥 Inventory Health",
        "page_simulator": "⚡ Policy Simulator",
        "page_alerts": "🔔 Alerts",

        # Home
        "home_welcome": "Welcome",
        "home_intro": """
This prototype demonstrates intelligent demand forecasting and replenishment
for China's largest out-of-hospital (OOH) pharmaceutical B2B platform.

**Core Capabilities:**
- **Multi-factor demand forecasting** (14–30 days ahead) using ETS, Croston/SBA, and XGBoost
- **Intelligent replenishment** with FEFO expiry-aware (R,S) policies
- **Inventory health diagnosis** via ABC/XYZ matrix and expiry risk scoring
- **VBP policy simulation** for hospital-to-retail demand spillover
        """,
        "home_how_to_use": "How to Use",
        "home_guide": """
1. **Demand Forecast** — Select a drug and pharmacy type to view historical sales,
   AI-generated forecasts, and confidence intervals.
2. **Replenishment** — Check automated restock recommendations with risk flags
   (Red/Yellow/Green) and suggested order quantities.
3. **Inventory Health** — Explore ABC/XYZ segmentation and expiry risk heatmaps.
4. **Policy Simulator** — Simulate the impact of National Volume-Based Procurement
   (VBP) on demand and revenue.
5. **Alerts** — Monitor impending stockouts and near-expiry lots.
        """,
        "home_data_scope": "Data Scope",
        "home_data_detail": """
- **80 SKUs** across 5 therapeutic areas (Cardiovascular, Antiinfectives, Metabolism, Respiratory, CNS)
- **3 pharmacy types:** Hospital, Chain, Independent
- **2-year weekly data:** June 2024 – May 2026
- **External signals:** ILI%, holidays, VBP batch shocks
        """,

        # Forecast
        "forecast_title": "Demand Forecasting Dashboard",
        "forecast_select_sku": "Select Drug (SKU)",
        "forecast_select_pharmacy": "Pharmacy Type",
        "forecast_chart_title": "Historical Sales & 12-Week Forecast",
        "forecast_actual": "Actual Sales",
        "forecast_predicted": "Forecast",
        "forecast_ci": "90% Confidence Interval",
        "forecast_metrics": "Forecast Accuracy Metrics",
        "forecast_mape": "MAPE",
        "forecast_smape": "SMAPE",
        "forecast_mase": "MASE",
        "forecast_trend": "Trend Direction",
        "forecast_trend_up": "📈 Rising",
        "forecast_trend_down": "📉 Declining",
        "forecast_trend_flat": "➡️ Stable",
        "forecast_feature_importance": "XGBoost Feature Importance",
        "forecast_feature_importance_hint": "Top 10 features driving the forecast for this Demo SKU.",
        "forecast_feature_importance_missing": "Feature importance chart is only available for Demo SKUs (Metformin, Oseltamivir).",
        "forecast_no_data": "⚠️ Forecast data not found. Please run `python src/forecast_engine.py` first.",

        # Replenishment
        "replenishment_title": "Intelligent Replenishment",
        "replenishment_filter_risk": "Filter by Risk Level",
        "replenishment_filter_pharmacy": "Filter by Pharmacy Type",
        "replenishment_table_title": "Restock Recommendations",
        "replenishment_transfer_title": "Inter-Pharmacy Transfer Suggestions",
        "replenishment_no_data": "No replenishment data found. Please run `python src/replenishment_engine.py` first.",
        "replenishment_sku": "Drug",
        "replenishment_pharmacy": "Pharmacy",
        "replenishment_current_stock": "Current Stock",
        "replenishment_days_supply": "Days of Supply",
        "replenishment_recommended_qty": "Recommended Qty",
        "replenishment_order_date": "Order Date",
        "replenishment_risk": "Risk",
        "replenishment_reason": "Rationale",
        "replenishment_abc_xyz": "ABC/XYZ",
        "replenishment_expiry": "Expiry (weeks)",
        "replenishment_all_risks": "All",
        "replenishment_red": "🔴 Red (Urgent)",
        "replenishment_yellow": "🟡 Yellow (Caution)",
        "replenishment_green": "🟢 Green (Healthy)",
        "replenishment_transfer_from": "From",
        "replenishment_transfer_to": "To",
        "replenishment_transfer_qty": "Qty",

        # Inventory Health
        "inventory_title": "Inventory Health Dashboard",
        "inventory_kpi_title": "Key Metrics",
        "inventory_abc_xyz_title": "ABC / XYZ Segmentation",
        "inventory_expiry_title": "Expiry Risk Heatmap",
        "inventory_no_data": "No inventory data found. Please run `python src/replenishment_engine.py` first.",
        "inventory_total_skus": "Total SKU-Pharmacy Pairs",
        "inventory_red_count": "Urgent (Red)",
        "inventory_yellow_count": "Caution (Yellow)",
        "inventory_green_count": "Healthy (Green)",
        "inventory_expiry_risk_count": "Near Expiry (< 6 mo)",
        "inventory_abc_tooltip": "ABC = Revenue concentration (A=top 80%). XYZ = Demand volatility (X=stable, Z=erratic).",
        "inventory_expiry_tooltip": "Red zones indicate lots with less than 6 months remaining shelf life.",

        # Policy Simulator
        "simulator_title": "VBP Policy Simulator",
        "simulator_select_batch": "Select VBP Batch",
        "simulator_batch_10": "Batch 10 (Apr 2025) — Cardiovascular & Diabetes",
        "simulator_impact_chart_title": "Demand Shift by Therapeutic Area",
        "simulator_kpi_title": "Impact Summary",
        "simulator_no_data": "No master data found. Please run data generation first.",
        "simulator_hospital_demand": "Hospital Demand Change",
        "simulator_retail_demand": "Retail Demand Change",
        "simulator_revenue_impact": "Revenue Impact",
        "simulator_affected_skus": "Affected SKUs",
        "simulator_before": "Before VBP",
        "simulator_after": "After VBP",
        "simulator_selected": "Selected (price-cut winners)",
        "simulator_nonselected": "Non-Selected (spillover beneficiaries)",
        "simulator_channel_hospital": "Hospital",
        "simulator_channel_chain": "Chain Pharmacy",
        "simulator_channel_independent": "Independent Pharmacy",
        "simulator_table_title": "Affected SKU Details",

        # Alerts
        "alerts_title": "Alert Center",
        "alerts_stockout_title": "⏰ Stockout Risk (< 5 days supply)",
        "alerts_expiry_title": "📅 Expiry Risk (< 6 months)",
        "alerts_no_alerts": "No alerts at this time. All inventory is healthy.",
        "alerts_sku": "Drug",
        "alerts_pharmacy": "Pharmacy",
        "alerts_days_supply": "Days Supply",
        "alerts_expiry_weeks": "Weeks to Expiry",
        "alerts_risk_reason": "Risk Reason",
        "alerts_action_needed": "Recommended Action",
        "alerts_total_stockout": "Stockout Alerts",
        "alerts_total_expiry": "Expiry Alerts",
        "alerts_filter_pharmacy": "Filter by Pharmacy",

        # Common
        "loading": "Loading data...",
        "error": "Error",
        "warning": "Warning",
        "success": "Success",
        "info": "Info",
        "submit": "Submit",
        "cancel": "Cancel",
        "refresh": "Refresh",
    },

    "zh": {
        # App
        "app_title": "药品需求预测与智能补货系统",
        "app_subtitle": "面向中国药品B2B平台的AI驱动库存优化方案",
        "sidebar_nav": "📍 导航",
        "language_label": "Language / 语言",

        # Pages
        "page_home": "🏠 首页",
        "page_forecast": "📈 需求预测",
        "page_replenishment": "📦 智能补货",
        "page_inventory": "🏥 库存健康",
        "page_simulator": "⚡ 政策模拟",
        "page_alerts": "🔔 预警中心",

        # Home
        "home_welcome": "欢迎使用",
        "home_intro": """
本原型展示了中国最大院外（OOH）药品B2B平台的智能需求预测与补货系统。

**核心能力：**
- **多因素需求预测**（提前14–30天），采用ETS、Croston/SBA和XGBoost
- **智能补货建议**，支持FEFO效期感知的(R,S)周期盘点策略
- **库存健康诊断**，通过ABC/XYZ矩阵和效期风险评分
- **VBP政策模拟**，模拟集采后医院到零售的需求转移
        """,
        "home_how_to_use": "使用指南",
        "home_guide": """
1. **需求预测** — 选择药品和药店类型，查看历史销量、AI预测和置信区间。
2. **智能补货** — 查看自动生成的补货建议及风险标签（红/黄/绿）。
3. **库存健康** — 探索ABC/XYZ分层和效期风险热力图。
4. **政策模拟** — 模拟国家药品集采（VBP）对需求和收入的影响。
5. **预警中心** — 监控即将断货和临近效期的品种。
        """,
        "home_data_scope": "数据范围",
        "home_data_detail": """
- **80个SKU**，覆盖5大治疗领域（心血管、抗感染、代谢、呼吸、中枢神经）
- **3种药店类型：** 医院药房、连锁药店、单体药店
- **2年周度数据：** 2024年6月 – 2026年5月
- **外部信号：** 流感ILI%、节假日、VBP批次冲击
        """,

        # Forecast
        "forecast_title": "需求预测仪表盘",
        "forecast_select_sku": "选择药品（SKU）",
        "forecast_select_pharmacy": "药店类型",
        "forecast_chart_title": "历史销量与12周预测",
        "forecast_actual": "实际销量",
        "forecast_predicted": "预测值",
        "forecast_ci": "90%置信区间",
        "forecast_metrics": "预测精度指标",
        "forecast_mape": "MAPE",
        "forecast_smape": "SMAPE",
        "forecast_mase": "MASE",
        "forecast_trend": "趋势方向",
        "forecast_trend_up": "📈 上升",
        "forecast_trend_down": "📉 下降",
        "forecast_trend_flat": "➡️ 平稳",
        "forecast_feature_importance": "XGBoost特征重要性",
        "forecast_feature_importance_hint": "驱动该Demo SKU预测结果的Top 10特征。",
        "forecast_feature_importance_missing": "特征重要性图仅对Demo SKU（二甲双胍、奥司他韦）可用。",
        "forecast_no_data": "⚠️ 未找到预测数据。请先运行 `python src/forecast_engine.py` 生成数据。",

        # Replenishment
        "replenishment_title": "智能补货",
        "replenishment_filter_risk": "按风险等级筛选",
        "replenishment_filter_pharmacy": "按药店类型筛选",
        "replenishment_table_title": "补货建议列表",
        "replenishment_transfer_title": "店间调拨建议",
        "replenishment_no_data": "未找到补货数据。请先运行 `python src/replenishment_engine.py` 生成数据。",
        "replenishment_sku": "药品",
        "replenishment_pharmacy": "药店",
        "replenishment_current_stock": "当前库存",
        "replenishment_days_supply": "可售天数",
        "replenishment_recommended_qty": "建议补货量",
        "replenishment_order_date": "建议下单日",
        "replenishment_risk": "风险",
        "replenishment_reason": "建议理由",
        "replenishment_abc_xyz": "ABC/XYZ",
        "replenishment_expiry": "效期（周）",
        "replenishment_all_risks": "全部",
        "replenishment_red": "🔴 红色（紧急）",
        "replenishment_yellow": "🟡 黄色（注意）",
        "replenishment_green": "🟢 绿色（健康）",
        "replenishment_transfer_from": "调出",
        "replenishment_transfer_to": "调入",
        "replenishment_transfer_qty": "数量",

        # Inventory Health
        "inventory_title": "库存健康仪表盘",
        "inventory_kpi_title": "关键指标",
        "inventory_abc_xyz_title": "ABC / XYZ 分层矩阵",
        "inventory_expiry_title": "效期风险热力图",
        "inventory_no_data": "未找到库存数据。请先运行 `python src/replenishment_engine.py` 生成数据。",
        "inventory_total_skus": "SKU-药店组合总数",
        "inventory_red_count": "紧急（红色）",
        "inventory_yellow_count": "注意（黄色）",
        "inventory_green_count": "健康（绿色）",
        "inventory_expiry_risk_count": "临近效期（<6月）",
        "inventory_abc_tooltip": "ABC = 销售额集中度（A=前80%）。XYZ = 需求波动度（X=稳定，Z=波动大）。",
        "inventory_expiry_tooltip": "红色区域表示效期不足6个月的批次。",

        # Policy Simulator
        "simulator_title": "VBP 集采政策模拟器",
        "simulator_select_batch": "选择集采批次",
        "simulator_batch_10": "第10批（2025年4月）— 心血管 & 糖尿病",
        "simulator_impact_chart_title": "各治疗领域需求变化",
        "simulator_kpi_title": "影响汇总",
        "simulator_no_data": "未找到主数据。请先运行数据生成脚本。",
        "simulator_hospital_demand": "医院渠道需求变化",
        "simulator_retail_demand": "零售渠道需求变化",
        "simulator_revenue_impact": "收入影响",
        "simulator_affected_skus": "受影响 SKU 数",
        "simulator_before": "集采前",
        "simulator_after": "集采后",
        "simulator_selected": "中选品种（降价入围）",
        "simulator_nonselected": "非中选品种（溢出受益）",
        "simulator_channel_hospital": "医院",
        "simulator_channel_chain": "连锁药店",
        "simulator_channel_independent": "单体药店",
        "simulator_table_title": "受影响 SKU 明细",

        # Alerts
        "alerts_title": "预警中心",
        "alerts_stockout_title": "⏰ 断货风险（可售天数 < 5 天）",
        "alerts_expiry_title": "📅 效期风险（效期 < 6 个月）",
        "alerts_no_alerts": "当前无预警。所有库存状态健康。",
        "alerts_sku": "药品",
        "alerts_pharmacy": "药店",
        "alerts_days_supply": "可售天数",
        "alerts_expiry_weeks": "剩余效期（周）",
        "alerts_risk_reason": "风险原因",
        "alerts_action_needed": "建议措施",
        "alerts_total_stockout": "断货预警数",
        "alerts_total_expiry": "效期预警数",
        "alerts_filter_pharmacy": "按药店筛选",

        # Common
        "loading": "加载数据中...",
        "error": "错误",
        "warning": "警告",
        "success": "成功",
        "info": "提示",
        "submit": "提交",
        "cancel": "取消",
        "refresh": "刷新",
    }
}


def get_text(key: str, lang: str = "en") -> str:
    """Get translated text by key."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
