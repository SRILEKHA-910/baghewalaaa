"""
CSS (Cyclic Steam Stimulation) physics-informed model.

Fitted against data/css_seed.csv. Two relationships hold cleanly in the seed
data and both have real physical grounding, which is why we model them
explicitly rather than black-box regressing on 15 rows:

1. Reservoir heating: temp rise per injection-day scales with steam
   temperature and steam rate (more energy delivered -> more heating),
   fit as a linear response per day of injection.

2. Viscosity-temperature response: heavy oil viscosity follows an Arrhenius
   relationship, mu = A * exp(B / T_kelvin). Fitting ln(viscosity) vs 1/T on
   the seed data gives R^2 = 0.98 -- i.e. reservoir temperature is really the
   whole story for viscosity, which is exactly why "steam by gut feel" is
   risky: you're one lever (steam) away from the outcome that actually
   matters (viscosity -> pumpability), and it's a very nonlinear lever.

Oil production and steam consumption are then modelled as functions of the
achieved viscosity reduction, calibrated to the seed data's cycle curves.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# Arrhenius fit: ln(viscosity) = B/T_kelvin + ln(A)
_ARR_B = 3508.86065
_ARR_LNA = -1.33001310

def viscosity_from_temp(temp_c: float) -> float:
    """Predict oil viscosity (cP) from reservoir temperature (deg C) via Arrhenius fit."""
    t_k = temp_c + 273.15
    return float(np.exp(_ARR_B / t_k + _ARR_LNA))

def temp_from_viscosity(viscosity_cp: float) -> float:
    """Inverse of the Arrhenius fit: what reservoir temp gives this viscosity."""
    t_k = _ARR_B / (np.log(viscosity_cp) - _ARR_LNA)
    return float(t_k - 273.15)


@dataclass
class CSSDayResult:
    day: int
    reservoir_temp_c: float
    viscosity_cp: float
    steam_this_day_tonnes: float


@dataclass
class CSSCycleResult:
    days: list  # list[CSSDayResult]
    final_temp_c: float
    final_viscosity_cp: float
    total_steam_tonnes: float
    predicted_oil_rate_bpd: float
    sor: float  # steam-oil ratio (tonnes steam per bbl oil, illustrative units matching seed data scale)


# Calibrated from seed data via least-squares fit against the actual per-day
# temperature trajectories of all 3 observed cycles (alpha fixed at a small
# physically-sane value so the model stays monotonic when the optimizer
# extrapolates beyond the seed data's steam-temp range; beta/base fit to data):
#   delta_T_day = ALPHA * max(steam_temp - current_temp, 0) + BETA * (steam_rate - 3000) + BASE
_HEAT_ALPHA = 0.03
_HEAT_BETA = 0.005868578826993273
_HEAT_BASE = 0.24831433268140718

# Oil production: power-law fit against seed data, log(oil_rate) = a*log(viscosity_after) + b
# (R^2 = 0.96 -- post-cycle viscosity alone explains production rate very well,
# which makes physical sense: mobility ~ 1/viscosity drives inflow to the pump).
_PROD_POWER_A = -1.07622938
_PROD_POWER_B = 14.40683637

def oil_rate_from_viscosity(viscosity_cp: float) -> float:
    return float(np.exp(_PROD_POWER_A * np.log(viscosity_cp) + _PROD_POWER_B))

# Steam volume per day is an exact unit conversion confirmed against seed data:
# steam_volume_tonnes = steam_rate_kg_hr * injection_hours / 1000


def simulate_css_cycle(steam_temp_c: float, steam_rate_kg_hr: float,
                        reservoir_temp_before_c: float, injection_days: int = 5,
                        injection_hours_per_day: float = 24.0) -> CSSCycleResult:
    """
    Simulate a full CSS injection cycle day-by-day: heating -> viscosity drop
    -> steam consumed, ending with the resulting oil production potential and
    steam-oil ratio (SOR). This lets the optimizer scan steam_temp/steam_rate/
    injection_days combinations to find the cheapest (lowest-SOR) cycle that
    still hits the viscosity/production target.
    """
    temp = reservoir_temp_before_c
    days = []
    total_steam = 0.0
    for d in range(1, injection_days + 1):
        gap = max(steam_temp_c - temp, 0)
        delta_t = _HEAT_ALPHA * gap + _HEAT_BETA * (steam_rate_kg_hr - 3000) + _HEAT_BASE
        # Physical ceiling: conductive/convective heating cannot push the reservoir
        # past the temperature of the steam being injected.
        temp = min(temp + delta_t, steam_temp_c)
        steam_today = steam_rate_kg_hr * injection_hours_per_day / 1000.0
        total_steam += steam_today
        visc = viscosity_from_temp(temp)
        days.append(CSSDayResult(day=d, reservoir_temp_c=round(temp, 1),
                                  viscosity_cp=round(visc, 0), steam_this_day_tonnes=round(steam_today, 1)))

    visc_after = days[-1].viscosity_cp
    oil_rate = oil_rate_from_viscosity(visc_after)
    sor = total_steam / max(oil_rate, 1e-6)

    return CSSCycleResult(
        days=days,
        final_temp_c=days[-1].reservoir_temp_c,
        final_viscosity_cp=days[-1].viscosity_cp,
        total_steam_tonnes=round(total_steam, 1),
        predicted_oil_rate_bpd=round(oil_rate, 1),
        sor=round(sor, 3),
    )


def _natural_cycle_length(steam_temp_c: float, steam_rate_kg_hr: float,
                           reservoir_temp_before_c: float, target_viscosity_cp: float,
                           max_days: int = 20, injection_hours_per_day: float = 24.0):
    """
    Simulate day-by-day and stop injecting on the FIRST day the target viscosity
    is reached -- i.e. inject exactly as much steam as needed, not more. This is
    the direct fix for "gut feel" over-injection: the twin knows when to stop.
    Returns the full cycle result at that natural stopping length, or None if the
    target isn't reachable within max_days at this steam setting.
    """
    result = simulate_css_cycle(steam_temp_c, steam_rate_kg_hr, reservoir_temp_before_c,
                                 injection_days=max_days, injection_hours_per_day=injection_hours_per_day)
    for i, day in enumerate(result.days):
        if day.viscosity_cp <= target_viscosity_cp:
            days_needed = i + 1
            return simulate_css_cycle(steam_temp_c, steam_rate_kg_hr, reservoir_temp_before_c,
                                       injection_days=days_needed, injection_hours_per_day=injection_hours_per_day)
    return None  # never reached target within max_days


def optimize_css_cycle(reservoir_temp_before_c: float, target_viscosity_cp: float,
                        steam_temp_range=(260, 320), steam_rate_range=(2800, 3400),
                        steps: int = 8, max_days: int = 20) -> dict:
    """
    CSS optimizer: for each candidate (steam_temp, steam_rate), find the natural
    stopping day (first day the target viscosity is reached) rather than forcing
    a fixed cycle length -- then pick whichever steam setting reaches the target
    at the LOWEST total SOR (steam per barrel). This directly replaces "inject by
    gut feel" with "inject exactly as much as the reservoir needs."
    """
    temps = np.linspace(*steam_temp_range, steps)
    rates = np.linspace(*steam_rate_range, steps)

    best = None
    best_effort = None  # fallback: lowest viscosity achieved if target unreachable
    for st in temps:
        for sr in rates:
            result = _natural_cycle_length(float(st), float(sr), reservoir_temp_before_c,
                                            target_viscosity_cp, max_days=max_days)
            if result is not None:
                if best is None or result.sor < best["result"].sor:
                    best = {"steam_temp_c": round(float(st), 1), "steam_rate_kg_hr": round(float(sr), 1),
                            "injection_days": len(result.days), "result": result}
            else:
                probe = simulate_css_cycle(float(st), float(sr), reservoir_temp_before_c, injection_days=max_days)
                if best_effort is None or probe.final_viscosity_cp < best_effort["result"].final_viscosity_cp:
                    best_effort = {"steam_temp_c": round(float(st), 1), "steam_rate_kg_hr": round(float(sr), 1),
                                   "injection_days": max_days, "result": probe}

    if best is None:
        visc = best_effort["result"].final_viscosity_cp
        return {**best_effort, "css_target_reachable": False,
                "note": f"Target viscosity {target_viscosity_cp:.0f} cP is not reachable within {max_days} days "
                        f"even at max steam settings; closest achievable is {visc:.0f} cP. "
                        f"Consider a wider steam range, a longer soak, or accepting a higher target."}

    return {**best, "css_target_reachable": True,
            "note": "Lowest-SOR steam setting that reaches the target viscosity, stopping injection "
                    "as soon as the target is hit rather than over-injecting."}
