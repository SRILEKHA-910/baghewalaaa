"""
Baghewala Digital Twin -- Streamlit prototype.

Run with:  streamlit run app.py

Tabs:
  1. Live Digital Twin  - the unified real-time loop (item #7): CSS + SRP
     reacting to each other automatically, with an event log and alerts.
  2. CSS Optimizer       - items #1, #2, #6: pick steam parameters that hit a
     target viscosity at minimum SOR, replacing gut-feel injection.
  3. SRP Auto-Tuner      - items #3, #4, #5: recommend SPM/stroke for the
     current viscosity, flag rod-float/overload risk before it happens.
  4. Field Data          - the seed data this whole twin was calibrated from.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import time
import pandas as pd
import numpy as np
import streamlit as st

from srp_physics import simulate_srp, recommend_spm, baseline_spm, baseline_fillage
from css_physics import simulate_css_cycle, optimize_css_cycle, viscosity_from_temp, temp_from_viscosity
from twin_engine import WellTwin

st.set_page_config(page_title="Baghewala Digital Twin", layout="wide", page_icon="🛢️")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
if "twins" not in st.session_state:
    st.session_state.twins = {
        "BGW_01": WellTwin(well_id="BGW_01", reservoir_temp_c=62, target_viscosity_cp=4500, spm=3.5),
        "BGW_02": WellTwin(well_id="BGW_02", reservoir_temp_c=58, target_viscosity_cp=4500, spm=3.2),
    }
if "running" not in st.session_state:
    st.session_state.running = False

st.title("🛢️ Baghewala Digital Twin")
st.caption(
    "CSS steam injection + SRP pumping, closed into one loop. "
    "No new sensors required -- built to poll the existing field portal (see `src/portal_connector.py`)."
)

tab_live, tab_css, tab_srp, tab_data = st.tabs(
    ["🔴 Live Digital Twin", "🔥 CSS Optimizer", "⚙️ SRP Auto-Tuner", "📊 Field Data"]
)

# ---------------------------------------------------------------------------
# TAB 1: Live Digital Twin
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Unified real-time loop")
    st.write(
        "Each well cycles between **PRODUCING** (reservoir cools, SRP auto-tunes SPM as viscosity "
        "creeps up) and **INJECTING** (an optimized CSS cycle runs and hands back a lower-viscosity "
        "reservoir). When SRP tuning alone can no longer keep the pump safe, the well automatically "
        "triggers its own CSS cycle -- the loop that doesn't exist in the field today."
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        if st.button("▶ Step x1", use_container_width=True):
            for w in st.session_state.twins.values():
                w.step()
    with col_ctrl2:
        if st.button("⏩ Step x10", use_container_width=True):
            for _ in range(10):
                for w in st.session_state.twins.values():
                    w.step()
    with col_ctrl3:
        if st.button("🔄 Reset simulation", use_container_width=True):
            st.session_state.twins = {
                "BGW_01": WellTwin(well_id="BGW_01", reservoir_temp_c=62, target_viscosity_cp=4500, spm=3.5),
                "BGW_02": WellTwin(well_id="BGW_02", reservoir_temp_c=58, target_viscosity_cp=4500, spm=3.2),
            }
            st.rerun()

    st.divider()

    well_cols = st.columns(len(st.session_state.twins))
    for col, (well_id, twin) in zip(well_cols, st.session_state.twins.items()):
        snap = twin.snapshot()
        with col:
            st.markdown(f"### {well_id}")
            phase_badge = "🔥 INJECTING" if snap["phase"] == "INJECTING" else "⚙️ PRODUCING"
            st.markdown(f"**Phase:** {phase_badge}   |   **Tick:** {snap['tick']}")

            status_color = {
                "Normal": "🟢", "Reduced Efficiency": "🟡", "High Load": "🟠", "Failure Risk": "🔴",
            }.get(snap["status"], "⚪")
            st.metric("Status", f"{status_color} {snap['status']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Viscosity (cP)", f"{snap['viscosity_cp']:,.0f}")
            m2.metric("SPM", f"{snap['spm']:.2f}")
            m3.metric("Fillage %", f"{snap['fillage_pct']:.0f}")

            m4, m5, m6 = st.columns(3)
            m4.metric("Oil rate (BPD)", f"{snap['oil_rate_bpd']:.0f}")
            m5.metric("Max rod load (kN)", f"{snap['max_rod_load_kn']:.0f}")
            m6.metric("SOR so far", f"{snap['sor_so_far']:.3f}")

            if snap["rod_float_risk"]:
                st.warning("⚠️ Rod-float / fluid-pound risk")
            if snap["overload_risk"]:
                st.error("🚨 Rod overload risk")

            with st.expander("Event log", expanded=False):
                for e in reversed(twin.log[-15:]):
                    icon = {"info": "•", "action": "🔧", "warning": "⚠️", "danger": "🚨"}.get(e["level"], "•")
                    st.write(f"`t={e['tick']:>3}` {icon} {e['message']}")

# ---------------------------------------------------------------------------
# TAB 2: CSS Optimizer
# ---------------------------------------------------------------------------
with tab_css:
    st.subheader("Optimize the next steam cycle")
    st.write(
        "Instead of injecting steam by gut feel, pick a target viscosity and the optimizer finds the "
        "steam temperature / rate that reaches it at the **lowest steam-oil ratio (SOR)** -- stopping "
        "injection as soon as the target is hit rather than over-injecting."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        reservoir_temp_before = st.slider("Current reservoir temperature (°C)", 20, 100, 55)
        st.caption(f"Implied current viscosity: **{viscosity_from_temp(reservoir_temp_before):,.0f} cP**")
    with c2:
        target_viscosity = st.slider("Target viscosity after cycle (cP)", 1500, 9000, 4500, step=100)
        st.caption(f"Requires reservoir to reach ≈ **{temp_from_viscosity(target_viscosity):.1f} °C**")
    with c3:
        max_days = st.slider("Max allowed injection days", 3, 20, 10)

    if st.button("🔍 Find optimal steam schedule", type="primary"):
        with st.spinner("Searching steam temperature / rate combinations..."):
            opt = optimize_css_cycle(reservoir_temp_before, target_viscosity, max_days=max_days)
        r = opt["result"]

        if opt.get("css_target_reachable", True):
            st.success(opt["note"])
        else:
            st.warning(opt["note"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Steam temp", f"{opt['steam_temp_c']:.0f} °C")
        m2.metric("Steam rate", f"{opt['steam_rate_kg_hr']:.0f} kg/hr")
        m3.metric("Injection days", f"{opt['injection_days']}")
        m4.metric("Resulting SOR", f"{r.sor:.3f}")

        day_df = pd.DataFrame([{
            "Day": d.day, "Reservoir Temp (°C)": d.reservoir_temp_c,
            "Viscosity (cP)": d.viscosity_cp, "Steam that day (t)": d.steam_this_day_tonnes,
        } for d in r.days])
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("Reservoir heating")
            st.line_chart(day_df.set_index("Day")[["Reservoir Temp (°C)"]])
        with cc2:
            st.caption("Viscosity response")
            st.line_chart(day_df.set_index("Day")[["Viscosity (cP)"]])
        st.dataframe(day_df, use_container_width=True, hide_index=True)
        st.info(
            f"Final oil production potential: **{r.predicted_oil_rate_bpd:.0f} BPD** "
            f"using **{r.total_steam_tonnes:.0f} tonnes** of steam total."
        )

    with st.expander("Compare a manual steam setting"):
        st.write("See what a specific (non-optimized) steam setting would achieve, for comparison.")
        mc1, mc2, mc3 = st.columns(3)
        m_temp = mc1.slider("Steam temp (°C)", 260, 330, 290, key="manual_temp")
        m_rate = mc2.slider("Steam rate (kg/hr)", 2800, 3500, 3100, key="manual_rate")
        m_days = mc3.slider("Injection days", 1, 15, 5, key="manual_days")
        manual_result = simulate_css_cycle(m_temp, m_rate, reservoir_temp_before, injection_days=m_days)
        st.write(
            f"Final viscosity: **{manual_result.final_viscosity_cp:.0f} cP**, "
            f"oil rate: **{manual_result.predicted_oil_rate_bpd:.0f} BPD**, "
            f"steam used: **{manual_result.total_steam_tonnes:.0f} t**, "
            f"SOR: **{manual_result.sor:.3f}**"
        )

# ---------------------------------------------------------------------------
# TAB 3: SRP Auto-Tuner
# ---------------------------------------------------------------------------
with tab_srp:
    st.subheader("Continuously optimize SPM / stroke for current viscosity")
    st.write(
        "The field currently keeps SPM fixed and only reacts once cavitation/rod damage has already "
        "started. Here the recommended SPM adapts continuously to viscosity, staying inside the safe "
        "fillage and rod-load envelope -- minimizing impact loading (rod float) *before* it happens."
    )

    c1, c2 = st.columns(2)
    with c1:
        visc_input = st.slider("Current oil viscosity (cP)", 3000, 13000, 8000, step=100)
    with c2:
        st.caption(f"Field's current naive SPM policy at this viscosity: **{baseline_spm(visc_input):.2f} SPM**")
        st.caption(f"Fillage that naive policy would give: **{baseline_fillage(visc_input):.0f}%**")

    rec = recommend_spm(visc_input)
    naive_state = simulate_srp(visc_input, baseline_spm(visc_input))

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### 🔴 Field's current approach (fixed/naive SPM)")
        st.metric("SPM", f"{baseline_spm(visc_input):.2f}")
        st.metric("Fillage", f"{naive_state.fillage_pct:.0f}%")
        st.metric("Max rod load (kN)", f"{naive_state.max_rod_load_kn:.0f}")
        st.metric("Status", naive_state.status)
    with cB:
        st.markdown("#### 🟢 Digital twin recommendation")
        st.metric("SPM", f"{rec['spm']:.2f}", delta=f"{rec['spm'] - baseline_spm(visc_input):+.2f} vs naive")
        st.metric("Stroke length (m)", f"{rec['stroke']:.2f}")
        st.metric("Fillage", f"{rec['state'].fillage_pct:.0f}%")
        st.metric("Max rod load (kN)", f"{rec['state'].max_rod_load_kn:.0f}")
        st.metric("Status", rec["state"].status)

    if rec.get("css_intervention_recommended"):
        st.error(
            "🚨 No SPM/stroke setting is fully safe at this viscosity. " + rec["note"]
        )
    else:
        st.success(rec["note"])

    st.divider()
    st.caption("Safe operating envelope across a range of viscosities")
    rows = []
    for v in range(3000, 13000, 250):
        r = recommend_spm(v)
        rows.append({
            "Viscosity (cP)": v,
            "Naive SPM (field today)": baseline_spm(v),
            "Recommended SPM (twin)": r["spm"],
            "Naive fillage %": simulate_srp(v, baseline_spm(v)).fillage_pct,
            "Recommended fillage %": r["state"].fillage_pct,
        })
    env_df = pd.DataFrame(rows)
    ec1, ec2 = st.columns(2)
    with ec1:
        st.caption("SPM: naive vs recommended")
        st.line_chart(env_df.set_index("Viscosity (cP)")[["Naive SPM (field today)", "Recommended SPM (twin)"]])
    with ec2:
        st.caption("Fillage %: naive vs recommended")
        st.line_chart(env_df.set_index("Viscosity (cP)")[["Naive fillage %", "Recommended fillage %"]])

# ---------------------------------------------------------------------------
# TAB 4: Field Data
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Seed data this digital twin was calibrated against")
    st.write(
        "All models in this app (Arrhenius viscosity-temperature response, SRP fillage/rod-load "
        "curves, oil-rate power law) were fitted against this data, extracted from the SIH-provided "
        "SRP and CSS datasets. As real portal data accumulates, these fits should be refreshed."
    )
    st.markdown("**SRP data**")
    st.dataframe(pd.read_csv(os.path.join(DATA_DIR, "srp_seed.csv")), use_container_width=True, hide_index=True)
    st.markdown("**CSS data**")
    st.dataframe(pd.read_csv(os.path.join(DATA_DIR, "css_seed.csv")), use_container_width=True, hide_index=True)
