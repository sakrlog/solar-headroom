"""research/: the off-grid battery simulator and the period aggregation (no downloads)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from common import aggregate  # noqa: E402
from throttling import OffGrid, household_load, simulate  # noqa: E402

STEP = 300
KWP = 6.0


def test_aggregate_like_statistics():
    starts = np.array([0.0, 3600.0, 7200.0])
    ts = np.arange(0, 3 * 3600, STEP, dtype=float)
    v = np.arange(len(ts), dtype=float)
    v[12:14] = np.nan                               # hour 2 keeps 10 of 12 samples (>= 80 %)
    v[24:27] = np.nan                               # hour 3 keeps 9 of 12: missing
    a = aggregate(ts, v, STEP, starts)
    assert a["mean"][0] == pytest.approx(5.5) and a["min"][0] == 0 and a["max"][0] == 11
    assert a["mean"][1] == pytest.approx(np.mean(np.arange(14, 24))) and a["min"][1] == 14
    assert np.isnan(a["mean"][2]) and np.isnan(a["max"][2])


def test_household_load_is_deterministic_and_plausible():
    ts = 1_656_633_600 + np.arange(0, 30 * 86400, STEP, dtype=float)          # July 2022
    a = household_load(ts, STEP, "America/Denver", KWP, seed=1)
    assert np.array_equal(a, household_load(ts, STEP, "America/Denver", KWP, seed=1))
    assert a.min() > 0
    kwh_per_day = a.sum() * STEP / 3.6e6 / 30
    assert 8 < kwh_per_day < 16
    assert np.allclose(household_load(ts, STEP, "America/Denver", 2 * KWP, seed=1), 2 * a)


def sunny_days(n_days=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_days * 288
    hour = (np.arange(n) * STEP / 3600) % 24
    potential = np.clip(5000 * np.sin(np.pi * (hour - 6) / 12), 0, None) * rng.uniform(0.5, 1.0, n)
    return potential, rng.uniform(200, 1500, n)


def test_simulate_conserves_energy_and_only_throttles_when_it_must():
    potential, load = sunny_days()
    sys_ = OffGrid()
    out = simulate(potential, load, STEP, KWP, sys_)
    pv, bat, soc = out["pv"], out["bat"], out["soc"]
    assert np.all(pv <= potential + 1e-9)
    assert np.all((soc >= sys_.soc_min - 1e-6) & (soc <= 100 + 1e-9))
    cap = sys_.battery_kwh_per_kwp * KWP * 3.6e6
    # stored energy follows the battery power (charging stores eff * |b|, discharging drains b / eff)
    delta = np.where(bat < 0, -bat * sys_.efficiency, -bat / sys_.efficiency) * STEP
    e = sys_.soc_start / 100 * cap + np.cumsum(delta)
    assert np.allclose(100 * e / cap, soc)
    # PV is held back only when the house and the battery (tapering near full) could not take more
    held = pv < potential - 1e-6
    assert held.any() and (pv >= potential - 1e-6).any()
    soc_before = np.r_[sys_.soc_start, soc[:-1]]
    accept = sys_.charge_w_per_kwp * KWP * np.clip((100 - soc_before) / (100 - sys_.taper_from), 0, 1)
    assert np.allclose(pv[held] - load[held], np.minimum(accept[held], -bat[held] + 1e-9))
    assert np.all(bat[held] <= 0)                                              # never discharging then
    # the battery covers the load when the sun cannot
    night = (potential == 0) & (soc > sys_.soc_min + 5)
    assert night.any() and np.allclose(bat[night], load[night])


def test_simulate_leaves_data_gaps_missing():
    potential, load = sunny_days(1)
    potential[100:110] = np.nan
    out = simulate(potential, load, STEP, KWP, OffGrid())
    gap = np.isnan(potential)
    for k in ("pv", "bat", "soc"):
        assert np.isnan(out[k][gap]).all() and not np.isnan(out[k][~gap]).any()


def test_nrel_day_reduction_clock_fix_and_outages():
    import io

    import pandas as pd
    from common import _reduce_nrel_day

    local = pd.date_range("2022-06-01 12:00", periods=10, freq="1min")          # logger clock (standard time)
    rows = []
    for i, t in enumerate(local):
        for metric, value in ((1, 1000.0), (2, 1000.0 if i < 5 else 50.0)):   # inverter 2 fails at 12:05
            rows.append({"measured_on": t, "utc_measured_on": t + pd.Timedelta(hours=7), "metric_id": metric,
                         "value": value})
    rows.append({"measured_on": local[0], "utc_measured_on": local[0], "metric_id": 99, "value": 5.0})
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf)
    ts, v = _reduce_nrel_day(buf.getvalue(), (1, 2))
    assert list(ts) == [pd.Timestamp("2022-06-01 19:00", tz="UTC").timestamp()]   # outage minutes dropped
    assert v[0] == 2000.0
    ts, _ = _reduce_nrel_day(buf.getvalue(), (1, 2), clock_utc_offset_h=-8.0)
    assert ts[0] == pd.Timestamp("2022-06-01 20:00", tz="UTC").timestamp()       # local + 8 h, not the stamp
    assert _reduce_nrel_day(buf.getvalue(), (7,))[0].size == 0
