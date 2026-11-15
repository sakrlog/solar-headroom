# Solar Headroom

A Home Assistant integration that knows what your solar panels **could** deliver right now, even
while the inverter is holding back (battery full, zero export, export limit, deliberate
curtailment), and learns your system from Home Assistant's own history.

Status: **engine done and validated (Phase 0); Home Assistant integration in progress.**

## Design in one paragraph

Physics that is the same everywhere (sun position, clear sky, panel plane, temperature) is code.
Only a correction factor is learned per site: actual power ÷ clear-sky power, from weather
inputs. Periods when the inverter was throttling are treated as *floors*, not targets, so throttled
systems learn their real potential instead of learning "low". Models are retrained from scratch
every night on everything collected so far (seconds of CPU), so there is no shipped model and no
online-learning library. Everything runs on numpy alone, which ships with Home Assistant.

## Layout

```
custom_components/solar_headroom/            the Home Assistant integration
    config_flow.py   solar sensor(s) -> panels -> history check; panel groups are subentries
    coordinator.py   history sync (recorder statistics), Open-Meteo, nightly training, outputs
    sensor.py        potential now, solar bank today / left today / tomorrow, learning status
    energy.py        hourly forecast for the Energy dashboard
    button.py, diagnostics.py, translations/
custom_components/solar_headroom/engine/    numpy-only engine, no Home Assistant imports
    solar.py     sun position, clear sky (Ineichen), plane of array (Hay-Davies), IAM, temperature
    gbm.py       histogram gradient boosting with a censored squared loss, JSON persistence
    regimes.py   full output vs throttled, per system type; learns voltage band and charge limit per segment
    history.py   Home Assistant statistics rows -> aligned hourly / 5-minute arrays (units, signs, renames)
    weather.py   Open-Meteo request builders and parsers (satellite, forecasts, archives; no I/O)
    features.py  sun/season, satellite clear-sky indices (concurrent and real-time), forecast features
    models.py    targets, censored fitting with the leak guard, scoring (k = power / clear-sky power)
    daily.py     daily solar bank: target reconstruction, forecast features, ranges, today's bank
    store.py     the integration's SQLite file: statistics rows, weather archives, models
    pipeline.py  archives to request, training with champion/challenger, outputs from the live forecast
    train_job.py nightly training as a separate low-priority process (Home Assistant's own Python)
tests/                                       unit + parity tests (pvlib / LightGBM as dev-only references)
tests/ha/                                    integration tests (Home Assistant test harness, fake Open-Meteo)
research/                                    experiments on public data (not shipped): learning curves,
                                             simulated throttling; see research/README.md
```

## Development

Two environments: the engine on any Python >= 3.12 with the reference libraries, and Home Assistant's
test harness (Home Assistant 2026.10+ needs Python 3.14; `uv` installs it without admin rights).

```bash
# engine + parity references
python3 -m venv .venv && .venv/bin/pip install numpy pytest ruff pvlib pandas lightgbm pyarrow
.venv/bin/python -m pytest            # tests/ha is skipped here
# Home Assistant
uv venv .venv-ha --python 3.14 && VIRTUAL_ENV=.venv-ha uv pip install pytest-homeassistant-custom-component ruff
.venv-ha/bin/python -m pytest         # engine tests without the parity references + tests/ha
.venv-ha/bin/ruff check .
```

## License

Apache License 2.0, see [LICENSE](LICENSE). Copyright 2026 sakrlog.

Weather and satellite data: [Open-Meteo](https://open-meteo.com) (CC BY 4.0). Research data:
NREL PVDAQ (CC BY 4.0), see `research/README.md`.
