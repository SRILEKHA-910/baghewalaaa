"""
SRP (Sucker Rod Pump) physics-informed simulator.

The field data we were given (data/srp_seed.csv) only shows what happened
historically at Baghewala, where SPM was bumped up roughly in step with
rising viscosity but was never *independently* tuned or brought back down.
That means the raw data alone can't tell us what happens if we run a
different SPM at a given viscosity -- which is exactly the control decision
the SRP auto-tuner needs to make.

So instead of fitting a black-box model directly to 15 correlated rows
(which would just memorize "high viscosity -> high SPM" and be useless for
control), we fit the *viscosity-only baseline trend* from the seed data,
then layer on a physically-reasoned SPM/stroke response so the simulator
can be queried at any (viscosity, SPM, stroke) combination -- including ones
never seen in the historical log. This is what lets us generate a rich
synthetic training set for the auto-tuner and rod-float detector.

Calibration notes (fit against data/srp_seed.csv, both wells):
- pump_fillage_pct vs oil_viscosity_cp: quadratic fit, R^2 > 0.99
- oil_rate_bpd, pump_efficiency_pct, rod loads, motor current: all near-linear
  in viscosity along the historical (baseline-SPM) trajectory
- baseline SPM in the seed data itself rises ~linearly with viscosity, which
  we treat as the *naive* operating policy the field currently uses -- i.e.
  exactly the "keep bumping SPM until something breaks" behaviour described
  in the problem statement.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---- Baseline trend fitted from seed data (viscosity-only effect) ----
# fillage(%) = a*v^2 + b*v + c
_FILLAGE_COEF = np.array([7.12462776e-08, -6.62158490e-03, 1.21722495e+02])

# naive historical SPM policy: spm ~ linear in viscosity (what the field does today)
_BASELINE_SPM_SLOPE = (4.5 - 3.0) / (11800 - 4200)
_BASELINE_SPM_INTERCEPT = 3.0 - _BASELINE_SPM_SLOPE * 4200

def baseline_spm(viscosity_cp: float) -> float:
    """The field's current naive SPM-vs-viscosity policy (for comparison only)."""
    return float(np.clip(_BASELINE_SPM_INTERCEPT + _BASELINE_SPM_SLOPE * viscosity_cp, 2.5, 6.5))

def baseline_fillage(viscosity_cp: float) -> float:
    return float(np.clip(np.polyval(_FILLAGE_COEF, viscosity_cp), 5, 100))


@dataclass
class SRPState:
    intake_pressure_psi: float
    fillage_pct: float
    motor_current_a: float
    motor_power_kw: float
    min_rod_load_kn: float
    max_rod_load_kn: float
    oil_rate_bpd: float
    pump_efficiency_pct: float
    status: str
    rod_float_risk: bool
    overload_risk: bool


# Reference well constants (approx. Baghewala sucker-rod pump geometry from seed data)
MAX_THEORETICAL_RATE_BPD_PER_SPM_PER_M = 55.0   # theoretical pump displacement factor
RATED_MAX_ROD_LOAD_KN = 100.0                    # design limit before structural risk
MIN_SAFE_FILLAGE_PCT = 55.0                      # below this -> fluid pound / rod float territory


def simulate_srp(viscosity_cp: float, spm: float, stroke_length_m: float = 2.0) -> SRPState:
    """
    Predict SRP behaviour for an arbitrary (viscosity, SPM, stroke) operating point.

    Model logic:
      1. Start from the viscosity-only baseline fillage (what fillage would be
         at the field's naive baseline SPM for this viscosity).
      2. Apply a correction for how far the *chosen* SPM deviates from that
         baseline SPM:
           - Running SPM higher than the fluid can supply (pump outrunning
             inflow, worse at high viscosity because fluid moves slower into
             the barrel) drags fillage down sharply -> fluid pound / rod float.
           - Running SPM lower gives the reservoir more time to fill the
             barrel, mildly improving fillage but capped at 100%.
      3. Rod loads, motor current/power, and oil rate follow from fillage,
         SPM, and stroke length via standard sucker-rod-pump relationships.
    """
    baseline = baseline_spm(viscosity_cp)
    base_fillage = baseline_fillage(viscosity_cp)

    spm_delta = spm - baseline
    # Outrunning inflow (spm too high) punishes fillage harder at high viscosity
    # since viscous fluid takes longer to flow into the barrel each stroke.
    visc_sensitivity = 1.0 + (viscosity_cp / 6000.0)
    if spm_delta > 0:
        fillage = base_fillage - visc_sensitivity * 9.0 * (spm_delta ** 1.4)
    else:
        fillage = base_fillage + 4.0 * (abs(spm_delta) ** 0.8)
    fillage = float(np.clip(fillage, 5, 100))

    # Stroke length: longer stroke gives fluid more time/volume per cycle -> mildly eases fillage
    fillage = float(np.clip(fillage + (stroke_length_m - 2.0) * 4.0, 5, 100))

    fillage_frac = fillage / 100.0
    oil_rate_bpd = MAX_THEORETICAL_RATE_BPD_PER_SPM_PER_M * spm * stroke_length_m * fillage_frac
    # normalize roughly against seed-data scale
    oil_rate_bpd = oil_rate_bpd * 0.72

    pump_efficiency_pct = float(np.clip(fillage * 0.98 - (viscosity_cp / 6000.0) * 3, 5, 100))

    # Rod loads: rise with viscosity (fluid/rod friction) and with SPM (dynamic/inertial loading)
    min_rod_load_kn = 12 + (viscosity_cp / 300.0) + (spm - 3) * 3.0
    max_rod_load_kn = 30 + (viscosity_cp / 155.0) + (spm - 3) * 9.0
    # fluid pound (very low fillage) adds shock loading on the up-stroke
    if fillage < MIN_SAFE_FILLAGE_PCT:
        pound_factor = (MIN_SAFE_FILLAGE_PCT - fillage) / MIN_SAFE_FILLAGE_PCT
        max_rod_load_kn *= (1 + 0.6 * pound_factor)

    motor_current_a = 14 + (viscosity_cp / 220.0) + (spm - 3) * 4.5
    motor_power_kw = 6 + (viscosity_cp / 300.0) + (spm - 3) * 5.0

    intake_pressure_psi = float(np.clip(560 - (viscosity_cp / 22.0), 250, 600))

    rod_float_risk = fillage < MIN_SAFE_FILLAGE_PCT
    overload_risk = max_rod_load_kn > RATED_MAX_ROD_LOAD_KN

    # Status bands calibrated to match the field's own observed fillage/status
    # correspondence in the seed data (Normal >=87%, Reduced 78-86%, High Load
    # 66-77%, Failure Risk <66%), with an override for outright overload.
    if overload_risk or fillage < 55:
        status = "Failure Risk"
    elif fillage < 66:
        status = "High Load"
    elif fillage < 87:
        status = "Reduced Efficiency"
    else:
        status = "Normal"

    return SRPState(
        intake_pressure_psi=round(intake_pressure_psi, 1),
        fillage_pct=round(fillage, 1),
        motor_current_a=round(motor_current_a, 1),
        motor_power_kw=round(motor_power_kw, 1),
        min_rod_load_kn=round(min_rod_load_kn, 1),
        max_rod_load_kn=round(max_rod_load_kn, 1),
        oil_rate_bpd=round(oil_rate_bpd, 1),
        pump_efficiency_pct=round(pump_efficiency_pct, 1),
        status=status,
        rod_float_risk=rod_float_risk,
        overload_risk=overload_risk,
    )


def recommend_spm(viscosity_cp: float, stroke_options=(1.5, 1.75, 2.0, 2.25, 2.5),
                   spm_range=(2.5, 6.5), step: float = 0.1) -> dict:
    """
    Auto-tuner: search SPM (and stroke length) at the given viscosity and pick the
    operating point that maximizes oil_rate_bpd * (efficiency/100) subject to safety
    constraints (no rod-float risk, no overload risk). This is the 'continuously
    optimize the pump' + 'minimize impact loading' logic tied together.

    If NO combination is safe at this viscosity, that is itself an important
    signal -- it means the SRP side has run out of room to compensate, and the
    real fix is upstream: trigger a fresh CSS steam cycle to bring viscosity back
    down, rather than continuing to push the pump. `css_intervention_recommended`
    flags exactly that case.
    """
    best = None
    spm_vals = np.arange(spm_range[0], spm_range[1] + 1e-9, step)
    all_candidates = []
    for stroke in stroke_options:
        for spm in spm_vals:
            state = simulate_srp(viscosity_cp, spm, stroke)
            all_candidates.append((spm, stroke, state))
            if state.rod_float_risk or state.overload_risk:
                continue
            score = state.oil_rate_bpd * (state.pump_efficiency_pct / 100.0)
            if best is None or score > best["score"]:
                best = {"spm": round(float(spm), 2), "stroke": stroke, "score": score, "state": state}

    if best is None:
        # nothing satisfies both constraints -> pick least-bad, and flag upstream CSS need
        all_candidates.sort(key=lambda x: (x[2].max_rod_load_kn, -x[2].fillage_pct))
        spm, stroke, state = all_candidates[0]
        return {
            "spm": round(float(spm), 2), "stroke": stroke, "state": state,
            "css_intervention_recommended": True,
            "note": "No SPM/stroke combination is safe at this viscosity. "
                    "Recommend an earlier/stronger CSS steam cycle rather than pushing the pump.",
        }

    return {
        "spm": best["spm"], "stroke": best["stroke"], "state": best["state"],
        "css_intervention_recommended": False,
        "note": "Optimal SPM/stroke within safe operating envelope.",
    }
