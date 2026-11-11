# Solar Potential (working name)

A Home Assistant integration that knows what your solar panels **could** deliver right now, even
while the inverter is holding back (battery full, zero export, export limit, deliberate
curtailment), and learns your system from Home Assistant's own history.

Status: **Phase 0, engine port** (no Home Assistant code yet).

## Design in one paragraph

Physics that is the same everywhere (sun position, clear sky, panel plane, temperature) is code.
Only a correction factor is learned per site: actual power ÷ clear-sky power, from weather
inputs. Periods when the inverter was throttling are treated as *floors*, not targets, so throttled
systems learn their real potential instead of learning "low". Models are retrained from scratch
every night on everything collected so far (seconds of CPU), so there is no shipped model and no
online-learning library. Everything runs on numpy alone, which ships with Home Assistant.

## Layout

```
custom_components/solar_potential/engine/   numpy-only engine, no Home Assistant imports
    solar.py     sun position, clear sky (Ineichen), plane of array (Hay-Davies), IAM, temperature
    gbm.py       histogram gradient boosting with a censored squared loss, JSON persistence
    regimes.py   full output vs throttled, per system type; learns voltage band and charge limit per segment
    history.py   Home Assistant statistics rows -> aligned hourly / 5-minute arrays (units, signs, renames)
    weather.py   Open-Meteo request builders and parsers (satellite, forecasts, archives; no I/O)
    features.py  sun/season, satellite clear-sky indices (concurrent and real-time), forecast features
    models.py    targets, censored fitting with the leak guard, scoring (k = power / clear-sky power)
    daily.py     daily solar bank: target reconstruction, forecast features, ranges, today's bank
tests/                                       unit + parity tests (pvlib / LightGBM as dev-only references)
research/                                    experiments on public data (not shipped): learning curves,
                                             simulated throttling; see research/README.md
```

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pytest ruff pvlib pandas lightgbm pyarrow
.venv/bin/python -m pytest
.venv/bin/ruff check .
```
