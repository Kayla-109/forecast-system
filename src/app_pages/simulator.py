"""VBP Policy Simulator page — based on Early-bird design."""

import pandas as pd
import streamlit as st

from src.data_loader import load_vbp_impact, load_master_data
from src.utils.i18n import get_text


def show_simulator(lang: str = "en"):
    """Render the VBP policy simulator dashboard."""
    st.title(get_text("simulator_title", lang))
    st.divider()

    vbp = load_vbp_impact()
    master = load_master_data()

    if vbp is None or master is None:
        st.warning(get_text("simulator_no_data", lang))
        return

    # ------------------------------------------------------------------
    # Scenario controls
    # ------------------------------------------------------------------
    st.subheader(get_text("simulator_kpi_title", lang) if lang == "en" else "政策影响模拟")

    c1, c2, c3 = st.columns(3)
    with c1:
        batches = sorted(vbp["vbp_batch"].unique().tolist())
        batch = st.selectbox(
            get_text("simulator_select_batch", lang) if lang == "en" else "选择集采批次",
            options=batches,
            format_func=lambda x: f"Batch {x}" if lang == "en" else f"第{x}批",
        )
    with c2:
        pr_drop = st.slider(
            "Price Reduction (%)" if lang == "en" else "降价幅度 (%)",
            0, 80, 51,
        ) / 100
    with c3:
        vol_up = st.slider(
            "Volume Uplift (%)" if lang == "en" else "销量 uplift (%)",
            0, 100, 45,
        ) / 100

    batch_df = vbp[vbp["vbp_batch"] == batch]
    if batch_df.empty:
        st.warning("No data for selected batch.")
        return

    avg_pre = batch_df["pre_vbp_price"].mean()
    avg_post = batch_df["post_vbp_price"].mean()

    # ------------------------------------------------------------------
    # Metric cards: Before / After / Reduction
    # ------------------------------------------------------------------
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(
            "Before VBP" if lang == "en" else "集采前均价",
            f"¥{avg_pre:.2f}",
        )
    with k2:
        st.metric(
            "After VBP" if lang == "en" else "集采后均价",
            f"¥{avg_post:.2f}",
        )
    with k3:
        pct = (1 - avg_post / avg_pre) * 100 if avg_pre > 0 else 0
        st.metric(
            "Price Drop" if lang == "en" else "降幅",
            f"-{pct:.1f}%",
            delta_color="inverse",
        )

    st.divider()

    # ------------------------------------------------------------------
    # Revenue impact calculator
    # ------------------------------------------------------------------
    st.subheader("Revenue Impact Calculator" if lang == "en" else "收入影响计算器")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        mq = st.number_input(
            "Monthly Qty (boxes)" if lang == "en" else "月销量 (盒)",
            0, 100000, 1000, 100,
        )
    with cc2:
        cp = st.number_input(
            "Current Price (¥)" if lang == "en" else "当前单价 (¥)",
            0.0, 1000.0, float(avg_pre), 1.0,
        )
    with cc3:
        np_val = round(cp * (1 - pr_drop), 2)
        np_in = st.number_input(
            "VBP Price (¥)" if lang == "en" else "集采后单价 (¥)",
            0.0, 1000.0, np_val, 1.0,
        )

    cr = mq * cp
    nr = int(mq * (1 + vol_up)) * np_in
    chg = nr - cr

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric(
            "Current Revenue" if lang == "en" else "当前月收入",
            f"¥{cr:,.0f}",
        )
    with r2:
        st.metric(
            "VBP Revenue" if lang == "en" else "集采后月收入",
            f"¥{nr:,.0f}",
        )
    with r3:
        cc = '#059669' if chg > 0 else '#c0392b'
        st.markdown(
            f"<div style='text-align:center; padding:16px; background:#f8f9fa; border-radius:8px;'>"
            f"<div style='font-size:14px; color:#6b7280;'>"
            f"{'Revenue Change' if lang == 'en' else '收入变化'}"
            f"</div>"
            f"<div style='font-size:28px; font-weight:700; color:{cc};'>"
            f"¥{chg:,.0f} ({chg/cr*100:.1f}%)"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # Affected SKU table
    # ------------------------------------------------------------------
    st.subheader(get_text("simulator_table_title", lang))

    display = batch_df.merge(
        master[["sku_id", "generic_name_cn", "therapy_area"]], on="sku_id", how="left"
    )
    display = display[[
        "sku_id", "generic_name_cn", "therapy_area",
        "pre_vbp_price", "post_vbp_price", "price_drop_pct", "volume_uplift_pct",
    ]]

    if lang == "zh":
        display.columns = [
            "SKU", "药品名称", "治疗领域", "集采前价格", "集采后价格", "降幅(%)", "量增(%)",
        ]
    else:
        display.columns = [
            "SKU", "Drug Name", "Therapy Area", "Pre-VBP Price", "Post-VBP Price",
            "Price Drop (%)", "Vol. Uplift (%)",
        ]

    st.dataframe(display, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Price drop bar chart
    # ------------------------------------------------------------------
    st.subheader("Price Drop by SKU" if lang == "en" else "各SKU降价幅度")

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display["药品名称" if lang == "zh" else "Drug Name"],
        y=display["降幅(%)" if lang == "zh" else "Price Drop (%)"],
        marker_color="#e74c3c",
        name="Price Drop %" if lang == "en" else "降幅(%)",
    ))
    fig.update_layout(
        xaxis_title="Drug" if lang == "en" else "药品",
        yaxis_title="Price Drop (%)" if lang == "en" else "降价幅度(%)",
        margin=dict(l=40, r=40, t=40, b=80),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
