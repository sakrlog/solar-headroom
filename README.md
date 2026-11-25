<img src="custom_components/solar_headroom/brand/icon.png" alt="" width="96" align="right">

# Solar Headroom

[![Tests](https://github.com/sakrlog/solar-headroom/actions/workflows/tests.yml/badge.svg)](https://github.com/sakrlog/solar-headroom/actions/workflows/tests.yml)
[![Validate](https://github.com/sakrlog/solar-headroom/actions/workflows/validate.yml/badge.svg)](https://github.com/sakrlog/solar-headroom/actions/workflows/validate.yml)
[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://hacs.xyz)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Home Assistant integration that knows what your solar panels **could** deliver right now, even
while the inverter is holding back (battery full, zero export, export limit, deliberate
curtailment), and learns your system from Home Assistant's own history.

**Highlights**

* Physics (sun position, clear sky, plane of array, cell temperature) in code; only a per-site
  correction factor is learned, so it is useful from day one and improves with history.
* A gradient-boosting learner written from scratch in numpy, with a *censored* loss: hours when the
  inverter was throttling count as lower bounds, not targets, so throttled systems learn their real
  potential instead of learning "low".
* Validated on public NREL PVDAQ systems with simulated throttling: learning curves, throttling bias
  and live-estimate accuracy in [research/](research/README.md).
* Parity-tested against pvlib and LightGBM; Home Assistant integration tests; CI with ruff, hassfest
  and HACS validation.

Status: **early development.** Works: learning from history, satellite real-time estimate, Energy
dashboard forecast, throttling-aware learning for off-grid and export-limited systems, and the live
estimate while the inverter holds back.

## What you get

| sensor | |
|---|---|
| Solar potential (W) | what the panels could give right now: measured at full output, estimated while the inverter holds back (attribute `basis`: measured / estimated / the sky model's) |
| Untapped solar (W) | potential minus what the panels give now: what you could still switch on (systems that can hold back) |
| Solar wasted today (kWh) | the energy the panels could have given today but were held back from (systems that can hold back) |
| Solar bank today / left today (kWh) | today's potential: what already came (or could have) plus the forecast for the rest, with a P10–P90 range |
| Solar forecast tomorrow (kWh) | with a P10–P90 range; also the hourly forecast in the Energy dashboard |
| Learning status | collecting / learning / ready / degraded, with the models' recent accuracy |

It tells you under *Settings -> Repairs* when it needs you: a sensor that gives no reading in daylight,
a sensor without history (`state_class` missing), no forecast for hours, or battery / grid power that
looks reversed (fixed with one click).

How close the live estimate gets while held back, on public systems with simulated throttling:
[research/README.md](research/README.md#3-the-live-estimate-while-held-back-livepy).

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
    config_flow.py   solar sensor(s) -> panels -> does it hold back? -> battery / grid sensors -> history
                     check (with a sign check); panel groups are subentries
    coordinator.py   history sync (recorder statistics), Open-Meteo, nightly training, outputs
    live.py          follows the sensors: the live potential, today's log of potential and used energy
    sensor.py        potential now, untapped, wasted today, solar bank today / left / tomorrow, status
    energy.py        hourly forecast for the Energy dashboard
    repairs.py       issues for the Repairs panel (and the one-click sign fix)
    button.py, diagnostics.py, translations/, brand/ (icons; drawn by docs/make_brand.py)
custom_components/solar_headroom/engine/    numpy-only engine, no Home Assistant imports
    solar.py     sun position, clear sky (Ineichen), plane of array (Hay-Davies), IAM, temperature
    gbm.py       histogram gradient boosting with a censored squared loss, JSON persistence
    regimes.py   full output vs throttled, per system type; learns voltage band and charge limit per segment
    live.py      the same rules on live readings (its PV-voltage band learned from live windows); the
                 estimate while held back; the day's energy log
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
