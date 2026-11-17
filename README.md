# Solar Headroom

A Home Assistant integration that knows what your solar panels **could** deliver right now, even
while the inverter is holding back (battery full, zero export, export limit, deliberate
curtailment), and learns your system from Home Assistant's own history.

Status: **early development.** The forecast side works (learning from history, satellite real-time
estimate, Energy dashboard forecast), and so does throttling-aware learning for off-grid and
export-limited systems; the live estimate while throttled is next.

## Installation (HACS, custom repository)

Requires Home Assistant 2026.10 or newer and the recorder (on by default).

1. HACS -> three-dot menu -> *Custom repositories* -> add `https://github.com/sakrlog/solar-headroom`,
   type *Integration*; install *Solar Headroom* and restart Home Assistant.
2. *Settings -> Devices & services -> Add integration -> Solar Headroom*: pick your solar power
   sensor(s) (pre-filled from the Energy dashboard), enter the panels' kWp, tilt and direction, and
   say whether your inverter ever holds back (never / export limit or zero export / off-grid). If it
   does, pick the battery and grid sensors that show when the panels were giving all they could
   (pre-filled from the Energy dashboard too; their signs are checked against the history).
3. Forecasts appear within seconds; the learned models follow once the history and weather archives
   are read (minutes). Add the forecast to the Energy dashboard under *Solar production*.

Only [Open-Meteo](https://open-meteo.com) is contacted (no account, no key), with coordinates rounded
to 0.1 deg (~10 km): about 150-300 requests a day, plus a one-time download of archives.

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
    button.py, diagnostics.py, translations/, brand/ (icons; drawn by docs/make_brand.py)
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
python3 -m venv .venv && .venv/bin/pip install -r requirements_dev.txt
.venv/bin/python -m pytest            # tests/ha is skipped here
# Home Assistant
uv venv .venv-ha --python 3.14 && VIRTUAL_ENV=.venv-ha uv pip install -r requirements_test.txt
.venv-ha/bin/python -m pytest         # engine tests without the parity references + tests/ha
.venv-ha/bin/ruff check .
```

CI (`.github/workflows/`): ruff, the engine tests on Python 3.12-3.14, the Home Assistant tests,
hassfest and the HACS validation.

## License

Apache License 2.0, see [LICENSE](LICENSE). Copyright 2026 sakrlog.

Weather and satellite data: [Open-Meteo](https://open-meteo.com) (CC BY 4.0). Research data:
NREL PVDAQ (CC BY 4.0), see `research/README.md`.
