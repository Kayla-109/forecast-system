"""Replenishment recommendations page for Streamlit app."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.i18n import get_text


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_replenishment_data():
    path = Path("reports/replenishment_recommendations.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_transfer_data():
    path = Path("reports/transfer_recommendations.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) > 0 else None


# ---------------------------------------------------------------------------
# Risk color helpers
# ---------------------------------------------------------------------------
def risk_badge(risk: str) -> str:
    """Return HTML badge for risk level."""
    colors = {
        "Red": "#e74c3c",
        "Yellow": "#f39c12",
        "Green": "#2ecc71",
    }
    color = colors.get(risk, "#95a5a6")
    return f"<span style='background-color:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.85rem;font-weight:600;'>{risk}</span>"


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------
def show_replenishment(lang: str = "en"):
    """Render the replenishment dashboard."""
    st.title(get_text("replenishment_title", lang))
    st.divider()

    # Load data
    df = load_replenishment_data()
    transfers = load_transfer_data()

    if df is None:
        st.warning(get_text("replenishment_no_data", lang))
        return

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    filt_col1, filt_col2, filt_col3 = st.columns(3)

    with filt_col1:
        risk_options = {
            get_text("replenishment_all_risks", lang): "All",
            get_text("replenishment_red", lang): "Red",
            get_text("replenishment_yellow", lang): "Yellow",
            get_text("replenishment_green", lang): "Green",
        }
        selected_risk_label = st.selectbox(
            get_text("replenishment_filter_risk", lang),
            options=list(risk_options.keys()),
            index=0,
        )
        selected_risk = risk_options[selected_risk_label]

    with filt_col2:
        pharmacy_types = ["All", "hospital", "chain", "independent"]
        pharmacy_labels = {
            "All": "All",
            "hospital": "Hospital",
            "chain": "Chain",
            "independent": "Independent",
        }
        selected_pharmacy = st.selectbox(
            get_text("replenishment_filter_pharmacy", lang),
            options=pharmacy_types,
            index=0,
            format_func=lambda x: pharmacy_labels.get(x, x),
        )

    with filt_col3:
        # Search by drug name
        search_term = st.text_input("🔍 Search drug", "")

    # Apply filters
    filtered = df.copy()
    if selected_risk != "All":
        filtered = filtered[filtered["risk_level"] == selected_risk]
    if selected_pharmacy != "All":
        filtered = filtered[filtered["pharmacy_type"] == selected_pharmacy]
    if search_term.strip():
        mask = (
            filtered["name_cn"].str.contains(search_term, case=False, na=False)
            | filtered["sku_id"].str.contains(search_term, case=False, na=False)
        )
        filtered = filtered[mask]

    # ------------------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------------------
    st.subheader(get_text("inventory_kpi_title", lang))
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Items", len(filtered))
    kpi2.metric(
        get_text("inventory_red_count", lang),
        len(filtered[filtered["risk_level"] == "Red"]),
    )
    kpi3.metric(
        get_text("inventory_yellow_count", lang),
        len(filtered[filtered["risk_level"] == "Yellow"]),
    )
    kpi4.metric(
        get_text("inventory_green_count", lang),
        len(filtered[filtered["risk_level"] == "Green"]),
    )

    st.divider()

    # ------------------------------------------------------------------
    # Recommendations Table
    # ------------------------------------------------------------------
    st.subheader(get_text("replenishment_table_title", lang))

    if len(filtered) == 0:
        st.info("No items match the current filters.")
    else:
        # Prepare display dataframe
        display = filtered[[
            "sku_id", "name_cn", "pharmacy_type", "current_stock",
            "days_supply", "recommended_qty", "order_date",
            "risk_level", "risk_reason", "reason",
            "abc_class", "xyz_class", "expiry_weeks",
        ]].copy()

        display["abc_xyz"] = display["abc_class"] + " / " + display["xyz_class"]

        # Rename columns for display
        col_names = {
            "name_cn": get_text("replenishment_sku", lang),
            "pharmacy_type": get_text("replenishment_pharmacy", lang),
            "current_stock": get_text("replenishment_current_stock", lang),
            "days_supply": get_text("replenishment_days_supply", lang),
            "recommended_qty": get_text("replenishment_recommended_qty", lang),
            "order_date": get_text("replenishment_order_date", lang),
            "risk_level": get_text("replenishment_risk", lang),
            "risk_reason": get_text("alerts_risk_reason", lang),
            "reason": get_text("replenishment_reason", lang),
            "abc_xyz": get_text("replenishment_abc_xyz", lang),
            "expiry_weeks": get_text("replenishment_expiry", lang),
        }
        display = display.rename(columns=col_names)

        # Format risk level with colored text using Pandas Styler
        def color_risk(val):
            colors = {"Red": "color: #e74c3c; font-weight: 600;",
                      "Yellow": "color: #f39c12; font-weight: 600;",
                      "Green": "color: #2ecc71; font-weight: 600;"}
            return colors.get(val, "")

        styled = display.style.map(color_risk, subset=[get_text("replenishment_risk", lang)])

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                get_text("replenishment_sku", lang): st.column_config.TextColumn(width="medium"),
                get_text("replenishment_recommended_qty", lang): st.column_config.NumberColumn(
                    help="Recommended order quantity based on (R,S) policy"
                ),
            },
        )

    # ------------------------------------------------------------------
    # Transfer Recommendations
    # ------------------------------------------------------------------
    if transfers is not None and len(transfers) > 0:
        st.divider()
        st.subheader(get_text("replenishment_transfer_title", lang))

        transfer_display = transfers[[
            "name_cn", "from_pharmacy", "to_pharmacy", "transfer_qty",
            "from_stock", "to_stock", "reason",
        ]].copy()

        transfer_display = transfer_display.rename(columns={
            "name_cn": get_text("replenishment_sku", lang),
            "from_pharmacy": get_text("replenishment_transfer_from", lang),
            "to_pharmacy": get_text("replenishment_transfer_to", lang),
            "transfer_qty": get_text("replenishment_transfer_qty", lang),
            "from_stock": "From Stock",
            "to_stock": "To Stock",
            "reason": get_text("replenishment_reason", lang),
        })

        st.dataframe(transfer_display, use_container_width=True, hide_index=True)
    elif transfers is not None and len(transfers) == 0:
        st.divider()
        st.info("No inter-pharmacy transfer recommendations at this time.")
