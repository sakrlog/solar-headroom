# Research: does the engine work on systems it has never seen?

Two experiments on public data, run with the same engine code the integration ships. Nothing here
is installed in Home Assistant.

| script | question |
|---|---|
| `learning_curves.py` | A new user installs: how accurate is the estimate after a day, a week, a month, a year? What does loading Home Assistant's history at install buy? |
| `throttling.py` | On a system that throttles (off-grid, battery full), does learning from throttled hours as *floors* recover the hidden potential, where simpler approaches learn "low"? |
| `live.py` | While the inverter is holding back right now, how close is the live estimate to what the panels could give? |

Results land in `research/results/*.json` (small, committed). Raw downloads go to `research/data/`
(git-ignored).

## In short

* **History at install pays off.** Trained on the year Home Assistant already stored, the first week
  has a 15 % hourly error, the level that learning from zero reaches after months (physics alone: 33 %).
* **Without history the estimate still improves week by week**: a calibrated factor from week 2, the
  learned model from about week 4, picked automatically on recent hours.
* **On a throttling system, neither drop the throttled hours nor take them at face value.** Both make
  the potential look 35–70 % lower than it is. Treating throttled hours as floors recovers most of it.
* **Time-of-day features leak the throttling.** Only the darker middays run at full output, so a learner
  that knows the time can learn "midday = darker than the forecast says". The engine's leak guard
  measures this on the training data and corrects it. Bias on throttled hours: −24 % → −8 % (Arbuckle),
  −11 % → −1 % (Golden), −5 % → −2 % (Henderson).
* **The live estimate sees through the throttling.** At moments the inverter held back, the panels
  delivered 69–73 % less than they could have; the live estimate is off by 10–19 % on average, with a
  bias of −1 to −8 %. It never mistook a throttled moment for full output.

## 1. A new user's first year (`learning_curves.py`)

Each site "installs" on 2022-03-01. Every week the models are retrained on the data available at that
point and scored on the following week against the measured output. The table shows hourly MAE as % of
the mean output, the three sites pooled per kWp:

| days since install | hours | physics | factor | ML | ML + history | shown |
|---|---|---|---|---|---|---|
| 0–7 | 141 | 32.8 % | 32.8 % | 32.8 % | **15.4 %** | 32.8 % |
| 7–14 | 227 | 31.6 % | 21.4 % | 28.4 % | **11.5 %** | 21.4 % |
| 14–28 | 454 | 27.4 % | 17.2 % | 16.5 % | **13.0 %** | 17.2 % |
| 28–56 | 940 | 31.0 % | 15.2 % | 13.9 % | **12.6 %** | 14.1 % |
| 56–112 | 1,951 | 36.4 % | 14.6 % | 12.3 % | **11.5 %** | 12.6 % |
| 112–224 | 3,840 | 37.9 % | 16.6 % | 13.7 % | **12.5 %** | 13.9 % |
| 224–364 | 3,930 | 33.5 % | 24.1 % | 16.6 % | **15.8 %** | 16.8 % |

* **physics**: Open-Meteo's forecast irradiance on the panel plane × kWp, temperature-corrected, no
  losses. Physics-only forecast tools work this way until the user tunes them.
* **factor**: physics × one factor learned from the hours since install (from 10 hours on).
* **ML**: the engine's model (sun/season + forecast features) trained on the hours since install
  (from 50 hours on).
* **ML + history**: the same model, also trained on the year before install, as when Home Assistant's
  long-term statistics are loaded on day one.
* **shown**: what the integration displays without history: the factor, until the ML model has beaten
  it on the same held-out hours of the last 14 days (at least 30 hours).

Findings:

* With history, week one (15 %) is already at the level the other methods reach after months.
* Without history, the factor helps from week 2 and the ML model wins from about week 4. From then on the
  shown estimate stays within 0.3 points of the better of the two.
* Physics is biased high at every site, by +15 % to +61 % depending on site and season. It has no system
  losses (wiring, inverter, soiling, snow on the panels, ageing); the factor and the models learn them.
* Days 224–364 fall in October to February: low sun and snow raise every method's relative error.

## 2. A throttled system (`throttling.py`)

Each site's clean output is the hidden truth. On top of it, an off-grid home (no grid, no generator)
is simulated in 5-minute steps (15 minutes for Arbuckle's meter):

* **battery**: 1.6 kWh per kWp; charges at up to 500 W per kWp, tapering from 95 % state of charge;
  discharges down to 10 %; 95 % efficient each way.
* **household load**: about 10–14 kWh a day per 6 kWp, varying from day to day. It has standby,
  morning and evening peaks, daytime use, summer cooling and random 2 kW appliance bursts.

PV is cut back to what the house and the battery take. The result becomes what Home Assistant would
store: hourly mean / min / max of PV power, battery power and state of charge. `engine.regimes` labels
it exactly as in the integration, learning the battery's charge limit from the data. The learners are
then trained on held-out months (6 folds) and scored against the hidden truth.

The labels:

| | Golden | Henderson | Arbuckle |
|---|---|---|---|
| daylight hours | 8,573 | 8,744 | 20,149 |
| labelled full output | 28.8 % | 22.2 % | 32.5 % |
| actually throttled | 57.3 % | 63.3 % | 54.5 % |
| energy thrown away | 51.0 % | 58.7 % | 53.0 % |
| labelled full output but throttled | 0.0 % | 0.0 % | 0.0 % |
| labelled throttled but free | 19.6 % | 18.7 % | 19.2 % |

The rules never call a throttled hour full output. About a fifth of the hours they call throttled
were in fact free. Those become floors equal to the truth, which costs little.

The learners:

* **oracle**: trained on the true potential of every hour (an unreachable upper bound).
* **censored**: full-output hours exact, throttled hours as floors (a prediction is only penalised
  below its floor).
* **guarded**: censored + the leak guard. This is what the integration does.
* **weather only**: censored without the time-of-day and sun-geometry features.
* **excluded**: full-output hours only. Forecast tools do this when they calibrate on "clean" data.
* **naive**: every hour taken at face value.

Estimates for the throttled hours (MAE % / bias % against the hidden truth):

| learner | Golden | Henderson | Arbuckle |
|---|---|---|---|
| oracle | 14.0 / −3.6 | 7.2 / −1.1 | 12.0 / −2.8 |
| **guarded (engine default)** | **19.6 / −1.1** | **9.4 / −1.7** | **15.8 / −8.3** |
| censored, no guard | 18.4 / −10.9 | 10.2 / −4.6 | 26.2 / −24.2 |
| weather only | 20.1 / −0.5 | 10.0 / +0.4 | 15.8 / −8.3 |
| excluded | 60.8 / −60.5 | 48.5 / −48.3 | 63.6 / −63.5 |
| naive | 65.3 / −64.9 | 70.0 / −69.8 | 67.8 / −67.6 |
| physics (no losses) | 36.1 / +28.6 | 19.3 / +15.3 | 27.8 / +24.5 |

Daily solar bank: the measured energy at full output, max(measured, estimate) where throttled (MAE % /
bias %):

| learner | Golden | Henderson | Arbuckle |
|---|---|---|---|
| oracle | 7.1 / −0.5 | 3.9 / +0.2 | 8.2 / −0.3 |
| **guarded (engine default)** | **9.4 / +2.2** | **5.9 / −0.1** | **10.0 / −4.4** |
| censored, no guard | 9.7 / −5.6 | 6.2 / −2.5 | 18.2 / −16.8 |
| weather only | 9.7 / +2.6 | 6.2 / +1.8 | 10.0 / −4.4 |
| excluded | 38.4 / −38.4 | 35.6 / −35.6 | 42.0 / −42.0 |
| naive | 46.0 / −45.8 | 55.1 / −55.1 | 48.5 / −48.4 |
| physics (no losses) | 30.2 / +29.7 | 17.3 / +17.2 | 25.1 / +24.6 |

Findings:

1. **Dropping throttled hours is as wrong as taking them at face value.** An off-grid hour only runs at
   full output when the house and battery can take everything, so the full-output hours are the darker
   ones. Learning from them alone puts the throttled hours 48–64 % too low, close to the naive learner
   (65–70 %).
2. **Floors recover most of it.** Censored learning cuts the error on throttled hours from about
   50–70 % to 10–26 %.
3. **But time of day leaks the selection.** At high sun nearly every hour is throttled, and the few that
   run at full output are darker than their forecast. A learner with sun-position features learns
   "midday = darker than the forecast says" and stays 5–24 % low on throttled hours. Dropping those
   features removes the leak but also real information: on Henderson, where the leak is mild, the
   weather-only model is less accurate than the guarded one.
4. **The leak guard needs no truth, only the training data.** The integration trains both models every
   night and compares them on the throttled high-sun training hours. If the full model is lower there,
   its prediction blends toward the weather-only one, entirely so when it is 10 % lower. The weight it
   chose per fold: Arbuckle 0.99–1, Golden 0.84–1, Henderson 0.21–0.75. So it follows the weather-only
   model where the leak is strong and beats both where it is mild. Where throttling is not selected on
   the output, it stays off (unit tests: weight below 0.1).
5. **A gap to the oracle remains**: 2–6 points of MAE on throttled hours, and Arbuckle is still 8 % low.

Robustness: what if the battery starts tapering at 80 % instead of 95 % (`--taper-from 80`)? Every
guarded result moves by at most 1.5 points. Throttled-hours bias: Golden −1.7 %, Henderson −2.1 %,
Arbuckle −9.8 %. Daily bank MAE: 9.6 / 5.9 / 10.6 %. The unguarded learner gets worse: Arbuckle −26 %.
Full results: `results/throttling_taper80.json`.

## 3. The live estimate while held back (`live.py`)

The same simulated off-grid homes, now step by step at the data's own resolution (5 minutes; 15 for
Arbuckle's meter). Every step, the simulated PV power, battery power and state of charge go through
`engine.live` the way the integration feeds it from Home Assistant's sensors. It decides whether the
panels are at full output (battery helping, or taking everything below its learned charge limit) and,
if not, estimates what they could give. The sky model behind the estimate is the engine's nowcast
learner, trained on the other months' hourly statistics (forecast features only, no satellite here).

| site | throttled daylight steps | wrongly called full output | throttled steps: what the inverter delivered | live estimate (MAE / bias) | daily waste, true vs estimated |
|---|---|---|---|---|---|
| Golden, CO | 56 % | 0 % | −68.6 % | 19.4 % / −0.7 % | 14.1 vs 14.4 kWh (MAE 2.4) |
| Henderson, NV | 66 % | 0 % | −72.5 % | 9.9 % / −2.2 % | 20.1 vs 19.7 kWh (MAE 1.8) |
| Arbuckle, CA | 54 % | 0 % | −71.0 % | 15.4 % / −8.1 % | 20.6 vs 18.7 kWh (MAE 3.7) |

How the estimate uses the last full-output moment: the sky model's error at that moment is carried
forward, fading over ~40 minutes, **only upward**. On these sites the choice hardly matters (with an
hourly forecast-only sky model there is little to correct): carrying it both ways is 0.1–0.4 points
lower in error and 0.2–1.5 points more negative in bias. It matters on a real off-grid system
replayed from its 5-second log: there the panels are often at full output *because* a cloud dimmed
them (the battery has to help), and a downward correction carried that cloud into the sunnier minutes
that followed. Daily potential against the training's hindsight reconstruction: −11 % both ways,
−3.9 % upward only. Full results: `results/live.json`.

## Data

Three grid-tied systems from NREL's PVDAQ, chosen for different climates and for having two or more
years of data in the period that Open-Meteo's archived forecasts cover (2021-03 onwards):

| key | PVDAQ system | location | climate | panels | size entered | data used |
|---|---|---|---|---|---|---|
| `golden` | 50 | Golden, CO (1.8 km) | cold semi-arid, snow | 45° tilt, azimuth 158° | 6.0 kWp | 2021-03 – 2023-02, 1-2 min AC power |
| `henderson` | 1423 | Henderson, NV | hot desert | 35°, south | 6.48 kWp | 2021-03 – 2023-02, 1 min AC power (2 inverters) |
| `arbuckle` | 2107 | Arbuckle, CA | Mediterranean | 25°, south | 893 kWp, scaled to 8 kWp | 2021-03 – 2025-12, 15 min revenue meter |

"Size entered" is the nameplate from the PVDAQ metadata: what a user would type. Arbuckle is a farm
array; its power is scaled to a home-sized system (the learned factor is size-free anyway). It clips at
its inverters' limit on clear days (DC/AC about 1.27), a real-world effect the models have to learn.

Data quirks found and handled in `common.py` (each checked against the sun):

* **Henderson**: the dataset's UTC stamps are an hour early from spring to autumn. On clear days the
  output's centre of mass sat 55 min before solar noon from April to October but on time in winter: the
  logger runs on standard time and daylight saving was applied to it anyway. We use its local stamps + 8 h.
* **Arbuckle**: the 15-min meter is stamped at the interval *start*. Its cloud transients line up with
  the site's own 5-min plane-of-array sensor only that way (lag correlation peaks 10-15 min after the stamp).
* Moments when one of Henderson's two inverters is down while the other produces are dropped (outage, not
  weather); meter spikes above 105 % of the 99.9th percentile are dropped.

Weather: Open-Meteo historical forecasts (the first hours of each model run) for each site's panel
plane, requested through `engine.weather` exactly like the integration. No satellite: the archive for the
Americas starts in 2025-04, so these are forecast-style results (the integration adds satellite where
it exists).

Licences: PVDAQ is CC BY 4.0. Deline, C., Perry, K., Deceglie, M., Muller, M., Sekulic, W., & Jordan, D.
(2021). *Photovoltaic Data Acquisition (PVDAQ) Public Datasets* [Data set]. Open Energy Data Initiative
(OEDI), NREL. https://doi.org/10.25984/1846021. Weather data by Open-Meteo.com (CC BY 4.0).

## Running

```bash
.venv/bin/pip install pandas pyarrow            # or: pip install -e ".[dev,research]"
.venv/bin/python -I research/throttling.py       # ~400 MB of PVDAQ files + ~40 Open-Meteo requests on first run
.venv/bin/python -I research/throttling.py --taper-from 80     # a battery that tapers from 80 %
.venv/bin/python -I research/learning_curves.py
.venv/bin/python -I research/live.py             # ~2 minutes once the data is cached
```

Once the data is cached, each run takes about 6–7 minutes on a desktop PC.
