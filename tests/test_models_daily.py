"""engine.models and engine.daily on synthetic sites (no real data)."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from engine import daily as D
from engine import features as F
from engine import models as M
from engine import solar as S
from engine.gbm import GBMParams

FAST = GBMParams(rounds=60, learning_rate=0.1, min_leaf=10)
SITE = S.Site(45.0, 7.0, 300.0)
PLANES = (S.Plane(25, 170, 4.0),)


def ts(*args) -> float:
    return datetime(*args, tzinfo=UTC).timestamp()


# ---------------------------------------------------------------- models
def test_targets_and_config():
    cfg = M.ModelConfig.for_kwp(8.0)
    assert cfg.min_p_cs_w == 320 and cfg.p_ref_w == 1000
    t = {"ts": np.arange(5.0), "pv_w": np.array([500, 900, 100, np.nan, 2000.0]),
         "p_cs": np.array([1000, 1000, 200, 1000, 2000.0]),
         "regime": np.array(["mpp", "limited", "mpp", "mpp", "night"], dtype=object)}
    tg = M.targets(t, cfg)
    assert list(tg["k_obs"][:3]) == [0.5, 0.9, 0.5] and list(tg["cens"][:2]) == [False, True]
    assert list(tg["trainable"]) == [True, True, False, False, False]      # low sun / no PV / night
    assert tg["w"][4] == pytest.approx(4.0)


def test_group_folds_cover_each_group_once_and_balance():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 30, 3000)
    folds = M.group_folds(groups, 6)
    assert np.all(np.sum(folds, axis=0) == 1)
    for f in folds:                                       # a group never straddles folds
        assert not set(groups[f]) & set(groups[~f])
    sizes = [f.sum() for f in folds]
    assert max(sizes) - min(sizes) < 0.1 * len(groups)
    assert all(np.array_equal(a, b) for a, b in zip(folds, M.group_folds(groups, 6), strict=True))
    months = M.month_ids([ts(2025, 1, 31, 23), ts(2025, 2, 1, 0), ts(2026, 1, 1)])
    assert list(np.diff(months)) == [1, 11]


def test_fit_predict_and_metrics():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.uniform(0, 1, n)
    k_true = 0.2 + 0.8 * x
    p_cs = rng.uniform(500, 3000, n)
    cens = rng.random(n) < 0.5
    pv = np.where(cens, k_true * p_cs * rng.uniform(0.3, 1.0, n), k_true * p_cs)
    t = {"ts": np.arange(n, dtype=float), "x": x, "pv_w": pv, "p_cs": p_cs,
         "regime": np.where(cens, "limited", "mpp").astype(object)}
    cfg = M.ModelConfig(min_p_cs_w=300, p_ref_w=1000, params=FAST)
    t.update(M.targets(t, cfg))
    m = M.fit(t, t["trainable"], ["x"], cfg)
    k = M.predict_k(m, t)
    assert np.mean(np.abs(k - k_true)) < 0.05             # censored rows did not drag it down
    assert k.min() >= 0 and k.max() <= M.K_MAX
    r = M.metrics(k * p_cs, pv, cens, np.where(cens, "", "voltage"))
    assert r["all_full_output_n"] == (~cens).sum() == r["voltage_rule_n"]
    assert r["pct_below_known_floor"] < 5


# ---------------------------------------------------------------- daily
def test_local_days_follow_dst():
    t = np.array([ts(2025, 3, 29, 22, 30), ts(2025, 3, 29, 23, 30), ts(2025, 6, 30, 21, 30), ts(2025, 6, 30, 22, 30)])
    d = D.local_day(t, "Europe/Berlin")
    assert [str(D.day_date(x)) for x in d] == ["2025-03-29", "2025-03-30", "2025-06-30", "2025-07-01"]
    assert D.day_number("2025-03-30") == d[1]
    assert D.local_midnight(D.day_number("2025-07-01"), "Europe/Berlin") == ts(2025, 6, 30, 22)


def test_hour_grid_daylight_hours_of_local_days():
    g = D.hour_grid("2025-06-01", "2025-06-03", "Europe/Rome", SITE, PLANES)
    assert set(g["day"]) == {D.day_number(d) for d in ("2025-06-01", "2025-06-02", "2025-06-03")}
    assert np.all(g["p_cs"] > D.MIN_GRID_P_CS) and np.all(np.diff(g["ts"]) > 0)
    assert 14 <= np.sum(g["day"] == g["day"][0]) <= 18           # ~16 daylight hours in June at 45 N


def synthetic_hours(n_days=60, seed=2):
    """Hour grid + a hidden truth k(x) + battery-full throttling in the afternoon."""
    rng = np.random.default_rng(seed)
    g = D.hour_grid("2025-04-01", D.day_date(D.day_number("2025-04-01") + n_days - 1), "Europe/Rome", SITE, PLANES)
    n = len(g["ts"])
    cloud = np.repeat(rng.uniform(0, 1, n_days), np.bincount((g["day"] - g["day"].min()).astype(int)))
    g["sat_kgti"] = 1 - 0.8 * cloud + rng.normal(0, 0.03, n)          # what the satellite saw
    k_true = np.clip(1.0 - 0.8 * cloud, 0.05, 1.2)
    truth = k_true * g["p_cs"]
    throttled = (g["hour"] > 13) & (rng.random(n) < 0.8)
    g["pv_w"] = np.where(throttled, truth * rng.uniform(0.2, 0.9, n), truth)
    g["regime"] = np.where(g["pv_w"] < 20, "night", np.where(throttled, "limited", "mpp")).astype(object)
    return g, truth


def test_reconstruct_fills_throttled_hours_from_hindsight():
    g, truth = synthetic_hours()
    cfg = M.ModelConfig(min_p_cs_w=150, p_ref_w=500, params=FAST)
    g.update(M.targets(g, cfg))
    gap = np.zeros(len(g["ts"]), bool)
    gap[100:110] = True                                       # HA was down: no data
    g["regime"][gap], g["pv_w"][gap] = "", np.nan
    g.update(M.targets(g, cfg))
    folds = M.group_folds(M.month_ids(g["ts"]), 2)
    r = D.reconstruct(g, ["sun_el", "poa_cs", "hour", "sat_kgti"], folds, cfg)
    mpp, lim = g["regime"] == "mpp", g["regime"] == "limited"
    assert np.array_equal(r["pot"][mpp], g["pv_w"][mpp])
    assert np.all(r["pot"][lim] >= g["pv_w"][lim])
    big = lim & (g["p_cs"] > 1000)
    assert abs(np.sum(r["pot"][big]) / np.sum(truth[big]) - 1) < 0.08     # the hidden truth is recovered
    assert r["missing"][gap & (g["p_cs"] >= cfg.min_p_cs_w)].all() and not r["missing"][mpp].any()
    d = D.day_table(r)
    assert len(d["day"]) == 60 and d["complete"].sum() < 60
    assert np.allclose(d["e_cs"].sum(), r["p_cs"].sum() / 1000)
    assert np.all(d["e_exact"] <= d["e_pot"] + 1e-9)


def test_forecast_features_and_day_summaries():
    g = D.hour_grid("2025-06-01", "2025-06-02", "Europe/Rome", SITE, PLANES)
    hours = np.arange(g["ts"][0] - 7200, g["ts"][-1] + 7200, 3600.0)
    gti = 400 + 100 * np.sin(np.arange(len(hours)))
    bm = {"ts": hours + 3600, "global_tilted_irradiance": gti, "shortwave_radiation": gti,
          "temperature_2m": np.full(len(hours), 22.0), "cloud_cover": np.full(len(hours), 30.0),
          "precipitation": np.full(len(hours), 0.1)}
    gfs = {"ts": hours + 3600, "global_tilted_irradiance": gti * 0.5, "cloud_cover": np.full(len(hours), 70.0)}
    f = D.forecast_features(g, {"best_match": bm, "gfs_seamless": gfs})
    lit = g["poa_cs"] >= F.SAT_MIN_POA
    first_day = g["day"] == g["day"][0]
    ref = np.nansum(f["fc_gti_raw"][first_day]) / np.sum(g["poa_cs"][first_day])
    assert f["day_kgti"][0] == pytest.approx(ref)
    assert np.isnan(f["fc_kgti_prev"][0]) and f["fc_kgti_prev"][1] == pytest.approx(f["fc_kgti"][0], nan_ok=True)
    last = np.flatnonzero(first_day)[-1]
    assert np.isnan(f["fc_kgti_next"][last])                     # never looks into the next day
    both = lit & ~np.isnan(f["fc_kgti"])
    assert np.allclose(f["ens_kgti_mean"][both], 0.75 * f["fc_kgti"][both])
    assert np.allclose(f["ens_kgti_std"][both], np.std([f["fc_kgti"][both], f["gfs_kgti"][both]], axis=0, ddof=1))
    assert np.isnan(f["icon_kgti"]).all() and f["ens_cloud_mean"][both][0] == pytest.approx(50.0)
    assert f["day_precip"][0] == pytest.approx(0.1 * first_day.sum())
    assert np.isnan(f["fc_snow_depth"]).all()


def test_range_bins_tables_and_bank():
    assert list(D.range_bin([0.0, 0.3, 0.55, 0.56, 0.9, 9.5, np.nan])) == [-1, 0, 0, 1, 3, -1, -1]
    rng = np.random.default_rng(3)
    clear = rng.uniform(0.1, 1.0, 400)
    ratio = rng.normal(1.0, 0.1, 400)
    tab = D.range_table(ratio, clear)
    assert set(tab) == {0, 1, 2, 3} and all(v[0] < v[1] < v[2] for v in tab.values())
    lo, hi = D.apply_range([10.0, 10.0], [np.nan, 0.95], {3: [0.8, 1.0, 1.2, 50]})
    assert list(lo) == [8.0, 8.0] and list(hi) == pytest.approx([12.0, 12.0])  # missing bins fall back
    hs = ts(2025, 6, 1, 4) + np.arange(10) * 3600.0
    p = np.full(10, 1000.0)
    before = D.bank_today(hs, p, now=hs[0] - 3600, clear=0.9)
    assert before["day_estimate_kwh"] == before["forecast_kwh"] == 10.0 and before["left_kwh"] == 10.0
    mid = hs[0] + 4 * 3600 + 1800                              # 4.5 h into the day
    known = {"covered_s": np.r_[np.full(4, 3600.0), 1800.0, np.zeros(5)],
             "potential_wh": np.r_[np.full(4, 1500.0), 750.0, np.zeros(5)],
             "used_wh": np.r_[np.full(4, 500.0), 250.0, np.zeros(5)]}
    b = D.bank_today(hs, p, now=mid, clear=0.9, known=known)
    assert b["left_kwh"] == 5.5 and b["so_far_potential_kwh"] == 6.8          # 6.75 rounded
    assert b["day_estimate_kwh"] == pytest.approx(12.2, abs=0.06)             # 6.75 known + 5.5 still to come
    assert b["so_far_wasted_kwh"] == 4.5 and b["log_coverage_pct"] == 100
    gap = D.bank_today(hs, p, now=mid, clear=0.9)                             # no log: forecast fills the past
    assert gap["day_estimate_kwh"] == 10.0 and gap["log_coverage_pct"] == 0


def test_physics_power_baseline():
    p = D.physics_power([1000.0, np.nan], [25.0, np.nan], PLANES, ts=np.zeros(2), t_air=20.0)
    assert p[0] == pytest.approx(4000 * (1 - 0.0035 * (25 + 1000 * 25 / 800 - 25)))
    assert p[1] == 0.0
