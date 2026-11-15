"""engine.features on small hand-checkable cases (parity with the pandas originals is checked
privately on real data)."""
from __future__ import annotations

import numpy as np
import pytest

from engine import features as F
from engine import solar as S

SITE = S.Site(48.0, 11.0, 500.0)
PLANES = (S.Plane(30, 180, 5.0),)
DAY = 1_750_000_000 - 1_750_000_000 % 86400           # a UTC midnight in June 2025


def test_helpers():
    assert list(F.lookup([0, 10, 20], [1, 2, 3], [10, 15, 20, 30])[[0, 2]]) == [2, 3]
    assert np.isnan(F.lookup([0, 10], [1, 2], [5])[0]) and np.isnan(F.lookup([], [], [1])[0])
    assert list(np.nan_to_num(F.ffill([np.nan, 1, np.nan, 3, np.nan]), nan=-1)) == [-1, 1, 1, 3, 3]
    assert list(np.nan_to_num(F.shift([1, 2, 3]), nan=-1)) == [-1, 1, 2]
    assert list(np.nan_to_num(F.shift([1, 2, 3], -1), nan=-1)) == [2, 3, -1]
    m = F.matrix({"ts": np.zeros(2), "a": np.array([1.0, 2.0])}, ["a", "missing"])
    assert m.shape == (2, 2) and np.isnan(m[:, 1]).all()


def test_sky_features_solar_time_and_segments():
    starts = DAY + np.arange(24) * 3600.0
    X = F.sky_features(starts, 3600, 12, SITE, PLANES, change_points=[DAY + 12 * 3600])
    noon = int(np.argmax(X["sun_el"]))
    assert abs(X["hour"][noon] - 12) < 0.6                     # solar noon at solar time ~12
    assert X["p_cs"][noon] > 3000 and X["p_cs"][0] == 0
    assert list(X["segment"][:12]) == [0] * 12 and list(X["segment"][12:]) == [1] * 12
    assert np.all(np.abs(X["doy_sin"] ** 2 + X["doy_cos"] ** 2 - 1) < 1e-12)


def sat_frame(step, n, gti=500.0, start=DAY + 6 * 3600):
    ts = start + np.arange(n) * step
    return {"ts": ts.astype(float), "ghi": np.full(n, gti), "dni": np.full(n, 600.0), "dhi": np.full(n, 100.0),
            "gti": np.full(n, gti)}


def test_sat_with_clearsky_indices():
    s = F.sat_with_clearsky(sat_frame(1800, 30), 1800, SITE, PLANES)
    day = s["poa_cs"] >= F.SAT_MIN_POA
    assert np.allclose(s["k"][day], 500.0 / s["poa_cs"][day])
    assert np.isnan(s["k"][~day]).all()


def test_quantile_map_and_harmonise():
    rng = np.random.default_rng(0)
    src = rng.uniform(0.1, 1.0, 5000)
    qm = F.fit_quantile_map(src, 0.8 * src)
    x = np.array([0.2, 0.5, 0.9, 2.0, np.nan])
    y = F.apply_quantile_map(x, qm)
    assert np.allclose(y[:3], 0.8 * x[:3], atol=0.01)
    assert y[3] == pytest.approx(2.0 * qm["ref"][-1] / qm["src"][-1]) and np.isnan(y[4])
    sat = {"ts": np.arange(3.0), "k": np.array([0.5, np.nan, 0.9]), "kd": np.array([0.5, 0.5, np.nan]),
           "poa_cs": np.array([100.0, 100.0, 100.0]), "dni_cs": np.array([800.0] * 3),
           "gti": np.array([50.0, 7.0, 90.0]), "dni": np.array([400.0] * 3)}
    h = F.harmonise(sat, {"k": qm, "kd": qm})
    assert h["gti"][0] == pytest.approx(100 * h["k"][0]) and h["gti"][1] == 7.0
    assert F.harmonise(sat, {"n_pairs": 3})["gti"][0] == 50.0


def test_sat_concurrent_hourly_and_short_periods():
    sat = {"ts": DAY + np.array([0, 1800, 3600, 5400, 7200]) * 1.0, "gti": np.array([100, 300, np.nan, 200, 50.0]),
           "poa_cs": np.array([200, 400, 400, 400, 40.0]), "dni": np.full(5, 100.0), "dni_cs": np.full(5, 200.0)}
    sat["k"] = sat["gti"] / sat["poa_cs"]
    out = F.sat_concurrent(sat, 1800, DAY + np.array([0, 3600, 7200]) * 1.0, 3600)
    assert out["sat_kgti"][0] == pytest.approx(400 / 600)
    assert out["sat_kgti_min"][0] == pytest.approx(0.5) and out["sat_kgti_max"][0] == pytest.approx(0.75)
    assert out["sat_kgti_std"][0] == pytest.approx(0.125)
    assert out["sat_kgti"][1] == pytest.approx(200 / 400)     # one valid interval of two is enough, and the
    #                                                           missing one does not dilute the index
    assert np.isnan(out["sat_kgti"][2])                         # below the clear-sky threshold
    assert out["sat_kdni"][0] == pytest.approx(0.5)
    short = F.sat_concurrent(sat, 1800, DAY + np.array([300, 2100, 3900]) * 1.0, 300)
    assert short["sat_kgti"][0] == pytest.approx(0.5) and short["sat_kgti"][1] == pytest.approx(0.75)
    assert np.isnan(short["sat_kgti"][2])                       # interval without GTI: unknown, not dark


def test_sat_lagged_uses_only_published_intervals():
    step = 600
    ts = DAY + np.arange(12) * step * 1.0
    k = np.linspace(0.1, 1.2, 12)
    sat = {"ts": ts, "gti": 1000 * k, "poa_cs": np.full(12, 1000.0), "dni": 500 * k, "dni_cs": np.full(12, 500.0)}
    t = np.array([DAY + 7200 + 300.0])                           # 02:05: intervals ending <= 01:45 are out
    out = F.sat_lagged(sat, step, t)
    newest = (7200 + 300 - 1200) // step * step / step - 1        # start index of the newest published interval
    i = int(newest)
    assert out["sat_kgti_lag_last"][0] == pytest.approx(k[i])
    assert out["sat_kgti_lag"][0] == pytest.approx(k[i - 5:i + 1].mean())       # 60-min window = 6 intervals
    assert out["sat_kgti_lag_trend"][0] == pytest.approx(k[i] - k[i - 3])
    assert out["sat_kdni_lag"][0] == pytest.approx(k[i - 5:i + 1].mean())
    assert out["sat_lag_min"][0] == pytest.approx(25.0)
    empty = F.sat_lagged({"ts": np.zeros(0), "gti": np.zeros(0), "poa_cs": np.zeros(0), "dni": np.zeros(0),
                          "dni_cs": np.zeros(0)}, step, t)
    assert np.isnan(empty["sat_kgti_lag"][0])


def test_satellite_features_priority():
    a = {"ts": DAY + np.array([0.0, 1800.0]), "gti": np.array([100.0, np.nan]), "poa_cs": np.array([200.0, 200.0]),
         "dni": np.array([1.0, 1.0]), "dni_cs": np.array([100.0, 100.0])}
    a["k"] = a["gti"] / a["poa_cs"]
    b = {k: v.copy() for k, v in a.items()}
    b["gti"] = np.array([180.0, 180.0])
    b["k"] = b["gti"] / b["poa_cs"]
    out = F.satellite_features([(a, 1800), (b, 1800)], DAY + np.array([0.0, 1800.0]), 1800)
    assert out["sat_kgti"][0] == pytest.approx(0.5) and out["sat_kgti"][1] == pytest.approx(0.9)


def test_wx_for_periods_interval_and_instant_variables():
    t = DAY + np.arange(0, 6) * 3600.0
    wx = {"ts": t, "global_tilted_irradiance": np.arange(6) * 100.0, "temperature_2m": np.array([10, 12, 14, np.nan,
                                                                                                  np.nan, 20.0])}
    out = F.wx_for_periods(wx, DAY + np.array([0, 3600, 7200 + 1800, 3 * 3600]) * 1.0, 3600)
    assert out["fc_gti"][0] == 100 and out["fc_gti"][1] == 200          # stamp T describes (T-1h, T]
    assert out["fc_temperature_2m"][0] == pytest.approx(11.0)           # interpolated to the period middle
    assert np.isnan(out["fc_temperature_2m"][3])                        # never across a 3 h gap
    assert "fc_ghi" not in out
    assert F.forecast_kgti({"fc_gti": np.array([50.0, 50.0])}, np.array([100.0, 10.0]))[0] == 0.5


def test_history_features():
    ts = np.arange(8) * 300.0
    k = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    mpp = np.array([True, False, False, True, False, False, False, False])
    out = F.history_features(ts, k, mpp, max_gap_s=1200)
    assert np.isnan(out["k_last_mpp"][0])
    assert out["k_last_mpp"][1] == 0.5 and out["min_since_mpp"][2] == 10
    assert out["k_last_mpp"][4] == 0.8 and out["min_since_mpp"][7] == 20
    assert out["k_last_mpp"][7] == 0.8
    assert np.isnan(F.history_features(ts, k, mpp, max_gap_s=600)["k_last_mpp"][7])
