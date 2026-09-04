"""
Unified real-time digital twin engine.

This is item #7 from the problem statement: "tie it all together as one
real-time system". Each well is modelled as a small state machine that
alternates between two phases, mirroring the actual field process:

  SOAK/PRODUCE phase:
    - Reservoir cools and oil re-thickens over time (viscosity creeps back up).
    - The SRP auto-tuner continuously reacts to that rising viscosity,
      adjusting SPM/stroke, and flags rod-float / overload risk.
    - If the SRP auto-tuner can find no safe operating point, the engine
      raises a CSS_TRIGGER alert -- the loop that currently doesn't exist
      in the field (CSS and SRP teams working blind from each other).

  CSS INJECTION phase:
    - Triggered manually or automatically by a CSS_TRIGGER alert.
    - Runs the optimized steam schedule (from css_physics.optimize_css_cycle)
      instead of a gut-feel one.
    - Viscosity drops day by day; once the target is hit, the well returns
      to SOAK/PRODUCE with a freshly-lowered viscosity.

This engine holds per-well state and exposes `step()` to advance simulated
time by one tick, so the Streamlit app can either single-step it or run it
on an auto-refresh timer for a live demo.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from srp_physics import simulate_srp, recommend_spm, MIN_SAFE_FILLAGE_PCT
from css_physics import (
    viscosity_from_temp, temp_from_viscosity, simulate_css_cycle, optimize_css_cycle,
)

Phase = Literal["PRODUCING", "INJECTING"]

# How fast oil re-thickens during production as the reservoir cools (deg C per tick).
COOLING_RATE_C_PER_TICK = 0.9
# Once viscosity creeps back up past this, proactively schedule a new CSS cycle
# (instead of waiting until the pump is already in Failure Risk).
PROACTIVE_CSS_VISCOSITY_TRIGGER_CP = 9500


@dataclass
class WellTwin:
    well_id: str
    reservoir_temp_c: float
    target_viscosity_cp: float = 4500.0
    phase: Phase = "PRODUCING"
    spm: float = 3.0
    stroke_length_m: float = 2.0
    tick: int = 0
    css_schedule: list = field(default_factory=list)   # list[CSSDayResult] queued for injection
    css_day_index: int = 0
    log: list = field(default_factory=list)             # human-readable event log, most recent last
    cumulative_steam_tonnes: float = 0.0
    cumulative_oil_bbl: float = 0.0

    def viscosity(self) -> float:
        return viscosity_from_temp(self.reservoir_temp_c)

    def _log_event(self, message: str, level: str = "info"):
        self.log.append({"tick": self.tick, "level": level, "message": message})
        self.log = self.log[-200:]  # cap log growth

    def start_css_cycle(self):
        opt = optimize_css_cycle(self.reservoir_temp_c, self.target_viscosity_cp)
        self.css_schedule = opt["result"].days
        self.css_day_index = 0
        self.phase = "INJECTING"
        self._log_event(
            f"CSS cycle started: steam_temp={opt['steam_temp_c']:.0f}C, "
            f"rate={opt['steam_rate_kg_hr']:.0f}kg/hr, planned {len(self.css_schedule)} day(s). {opt['note']}",
            level="action",
        )

    def step(self):
        self.tick += 1
        if self.phase == "INJECTING":
            self._step_injecting()
        else:
            self._step_producing()

    def _step_injecting(self):
        if self.css_day_index >= len(self.css_schedule):
            self.phase = "PRODUCING"
            self._log_event("CSS cycle complete. Reservoir handed back to SRP for production.", level="action")
            return
        day = self.css_schedule[self.css_day_index]
        self.reservoir_temp_c = day.reservoir_temp_c
        self.cumulative_steam_tonnes += day.steam_this_day_tonnes
        self.css_day_index += 1
        self._log_event(
            f"CSS day {day.day}: reservoir now {day.reservoir_temp_c:.1f}C, "
            f"viscosity {day.viscosity_cp:.0f} cP.", level="info",
        )

    def _step_producing(self):
        # Reservoir cools and oil re-thickens during production.
        self.reservoir_temp_c = max(20.0, self.reservoir_temp_c - COOLING_RATE_C_PER_TICK)
        visc = self.viscosity()

        rec = recommend_spm(visc, stroke_options=(self.stroke_length_m,))
        state = rec["state"]
        old_spm = self.spm
        self.spm = rec["spm"]
        self.cumulative_oil_bbl += state.oil_rate_bpd  # per tick, illustrative units

        if abs(self.spm - old_spm) >= 0.15:
            self._log_event(
                f"Auto-tuner adjusted SPM {old_spm:.2f} -> {self.spm:.2f} "
                f"(viscosity now {visc:.0f} cP, fillage {state.fillage_pct:.0f}%).",
                level="action",
            )

        if state.rod_float_risk:
            self._log_event(
                f"Rod-float / fluid-pound risk detected (fillage {state.fillage_pct:.0f}%). "
                f"Impact loading is being minimized by the SPM reduction above.",
                level="warning",
            )
        if state.overload_risk:
            self._log_event(
                f"Rod overload risk (max load {state.max_rod_load_kn:.0f} kN). Flagging for inspection.",
                level="danger",
            )

        if rec.get("css_intervention_recommended") or visc >= PROACTIVE_CSS_VISCOSITY_TRIGGER_CP:
            self._log_event(
                f"Viscosity {visc:.0f} cP has outrun what SRP tuning alone can safely handle. "
                f"Recommending a new CSS steam cycle.",
                level="danger",
            )
            self.start_css_cycle()

        self._last_srp_state = state

    def snapshot(self) -> dict:
        visc = self.viscosity()
        srp_state = getattr(self, "_last_srp_state", None) or simulate_srp(visc, self.spm, self.stroke_length_m)
        return {
            "well_id": self.well_id,
            "tick": self.tick,
            "phase": self.phase,
            "reservoir_temp_c": round(self.reservoir_temp_c, 1),
            "viscosity_cp": round(visc, 0),
            "spm": round(self.spm, 2),
            "stroke_length_m": self.stroke_length_m,
            "fillage_pct": srp_state.fillage_pct,
            "oil_rate_bpd": srp_state.oil_rate_bpd,
            "pump_efficiency_pct": srp_state.pump_efficiency_pct,
            "max_rod_load_kn": srp_state.max_rod_load_kn,
            "min_rod_load_kn": srp_state.min_rod_load_kn,
            "status": srp_state.status,
            "rod_float_risk": srp_state.rod_float_risk,
            "overload_risk": srp_state.overload_risk,
            "cumulative_steam_tonnes": round(self.cumulative_steam_tonnes, 1),
            "cumulative_oil_bbl": round(self.cumulative_oil_bbl, 1),
            "sor_so_far": round(self.cumulative_steam_tonnes / max(self.cumulative_oil_bbl, 1e-6), 3),
        }
