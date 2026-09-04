"""
Portal data connector.

The problem statement is explicit: no new sensors are needed -- the well's
existing instrumentation already reports to the field portal, so the digital
twin just needs to POLL that portal instead of relying on hand-recorded gut
feel. This module is the single seam where that live connection plugs in.

Right now `get_live_reading()` returns simulated/replayed readings (from the
seed CSVs, with realistic jitter) so the rest of the system -- optimizers,
dashboard, alerting -- can be built and demoed today. When the real portal
API/OPC-UA/SCADA endpoint is available, only this file needs to change:
implement `_fetch_from_portal()` to call the real endpoint and return the
same dict shape, and nothing else in the codebase needs to know the
difference.
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass

WELL_IDS = ["BGW_01", "BGW_02"]


@dataclass
class LiveReading:
    well_id: str
    timestamp: float
    oil_viscosity_cp: float
    pump_intake_pressure_psi: float
    pump_speed_spm: float
    stroke_length_m: float
    reservoir_temp_c: float


def _fetch_from_portal(well_id: str) -> LiveReading:
    """
    REAL INTEGRATION POINT.
    Replace this function body with an HTTP/OPC-UA/SCADA call to the
    Baghewala field portal, e.g.:

        resp = requests.get(f"{PORTAL_BASE_URL}/wells/{well_id}/latest", auth=...)
        data = resp.json()
        return LiveReading(well_id=well_id, timestamp=time.time(),
                            oil_viscosity_cp=data["viscosity"], ...)

    Everything downstream (optimizers, dashboard, alerts) already consumes
    LiveReading objects, so no other file needs to change.
    """
    raise NotImplementedError("Wire this up to the real field portal API when available.")


def get_live_reading(well_id: str, sim_state: dict | None = None) -> LiveReading:
    """
    Returns the latest reading for a well. Currently simulated; swap the
    body for `_fetch_from_portal(well_id)` once portal access is granted.
    """
    sim_state = sim_state or {}
    base_visc = sim_state.get("oil_viscosity_cp", 5000)
    base_temp = sim_state.get("reservoir_temp_c", 60)
    jitter = random.uniform(-1.5, 1.5)  # sensor noise, illustrative
    return LiveReading(
        well_id=well_id,
        timestamp=time.time(),
        oil_viscosity_cp=max(500, base_visc + jitter * 20),
        pump_intake_pressure_psi=sim_state.get("pump_intake_pressure_psi", 480),
        pump_speed_spm=sim_state.get("pump_speed_spm", 3.5),
        stroke_length_m=sim_state.get("stroke_length_m", 2.0),
        reservoir_temp_c=max(20, base_temp + jitter * 0.3),
    )
