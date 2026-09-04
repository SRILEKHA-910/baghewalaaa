# Baghewala Digital Twin (prototype)

A closed-loop digital twin for a heavy-oil well using **CSS** (Cyclic Steam
Stimulation) to thin the oil and an **SRP** (Sucker Rod Pump) to lift it —
built so the two stop operating blind from each other.

## The problem this solves

Today at Baghewala:
- The **CSS team** injects steam by gut feel — no model of how much heating
  is actually needed, so cycles waste steam (high SOR) or under-heat.
- The **SRP team** runs the pump at a fixed SPM and only reacts *after*
  cavitation/rod float has already started damaging equipment — several
  times a day.
- Nobody ties the two together, even though viscosity is the one variable
  that connects them: CSS controls it, SRP must react to it.

## What's in this prototype

```
baghewala_digital_twin/
├── app.py                  Streamlit dashboard (4 tabs, see below)
├── requirements.txt
├── data/
│   ├── srp_seed.csv        SRP field data (from the provided dataset)
│   └── css_seed.csv        CSS cycle data (from the provided dataset)
└── src/
    ├── css_physics.py      Reservoir heating + Arrhenius viscosity model,
    │                       CSS steam-schedule optimizer (min SOR)
    ├── srp_physics.py      SRP fillage/rod-load/efficiency model,
    │                       SPM/stroke auto-tuner (max output, safe envelope)
    ├── twin_engine.py       Ties CSS + SRP into one continuous per-well loop
    └── portal_connector.py  Single seam where the real field portal API
                              plugs in later (no new sensors needed)
```

### How the models were built (not black-box ML on 15 rows)

The seed datasets only have 15 rows each, and — importantly — the SRP data
shows SPM being bumped up *in lockstep* with rising viscosity historically,
which means a model fit directly to that data would just memorize
"high viscosity → high SPM" and be useless for actually recommending a
*different* SPM. So instead:

- **CSS side**: fit two real physical relationships out of the data —
  reservoir heating per injection day (linear response to steam
  temperature/rate, R² validated against all 3 observed cycles) and the
  **Arrhenius law** for viscosity vs. temperature (`ln(viscosity) = B/T + ln(A)`,
  **R² = 0.98** — reservoir temperature really is almost the whole story for
  viscosity). Oil production follows a power-law fit against post-cycle
  viscosity (R² = 0.96). These fits generalize far outside the 15 seed rows
  because they're physically grounded, not memorized correlations.
- **SRP side**: the viscosity-only baseline trend (fillage, rod loads, motor
  load) is fit directly from the seed data, then SPM/stroke are layered on
  as *independent* control variables with a physically-reasoned response
  (running SPM higher than the fluid can supply drags fillage down —
  worse at high viscosity — causing fluid pound / rod float; running it
  lower gives fillage more room). This is what lets the auto-tuner search
  SPM values that were **never observed** in the historical data.

### CSS optimizer logic

Given a target viscosity, it searches steam temperature × steam rate, and for
each candidate **stops injecting on the first day the target is reached**
(not a fixed cycle length) — this alone is the fix for "inject and hope":
inject exactly as much as the reservoir needs. Among candidates that reach
the target, it picks whichever has the lowest resulting SOR.

### SRP auto-tuner logic

Given current viscosity, searches SPM × stroke length and picks the point
that maximizes `oil_rate × efficiency`, subject to two hard safety
constraints:
- **no rod-float risk** (fillage above a safe threshold — this is exactly
  what causes fluid pound / impact loading)
- **no overload risk** (max rod load under the rated structural limit)

If **no** setting is safe at the current viscosity, that's a meaningful
signal in itself: SRP has run out of room to compensate, and the twin flags
that a new CSS cycle is needed upstream, instead of continuing to push the
pump into damage. `twin_engine.py` wires this trigger up automatically.

### The unified real-time loop (`twin_engine.py`)

Each well alternates **PRODUCING** (reservoir cools, SRP auto-tunes
continuously, alerts on rod-float/overload risk) and **INJECTING** (runs the
optimizer's steam schedule). When PRODUCING runs out of safe SPM options,
it automatically starts a new CSS cycle. This is the "tie it all together"
piece — CSS and SRP decisions now inform each other in one loop.

### Connecting to the real field portal

No new sensors are needed per the problem statement — `portal_connector.py`
is the single seam to wire up. Implement `_fetch_from_portal()` to call the
real portal's API/OPC-UA/SCADA endpoint and return the same `LiveReading`
shape; nothing else in the codebase needs to change.

## Running it

```bash
cd baghewala_digital_twin
pip install -r requirements.txt
streamlit run app.py
```

## Known limitations (be upfront about these in your pitch)

- Calibrated against only 15 rows per dataset (2-3 wells, 1-2 cycles) — the
  physics-informed approach generalizes better than black-box ML would here,
  but constants should be **re-fit once real portal data accumulates**.
- The CSS heating fit had a confounded predictor (steam temp and steam rate
  moved together in the seed data), so extrapolation far outside the
  observed steam-parameter range should be treated as directional, not exact.
- `portal_connector.py` currently simulates readings; it needs the real
  endpoint wired in before this can run against the live well.
- SRP physics constants (rod load formulas, rated load limit, etc.) are
  reasoned from sucker-rod-pump fundamentals + the seed data trend, not from
  a well-specific rod string design — worth validating against the actual
  BGW_01/BGW_02 rod/pump specs if available.

## Extending this

- Add a real ML layer (e.g. Gaussian Process or RandomForest) once enough
  live portal data accumulates, using the physics model's predictions as a
  prior/fallback for extrapolation.
- Add per-well persistence (currently in-memory `st.session_state`, resets
  on app restart).
- Add authentication + multi-well fleet view for a production deployment.
