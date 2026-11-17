"""engine.live: instantaneous regime, the estimate while throttled, the day log."""
from __future__ import annotations

import math

import numpy as np
import pytest

from engine import live as L
from engine import regimes as R


def _est(kind="off_grid", **kw) -> L.LiveEstimator:
    return L.LiveEstimator(R.RegimeConfig(kind=kind, **kw))


def test_unlimited_is_always_full_output():
    e = _est("unlimited")
    e.update("pv_w", 2500.0, 0.0)
    assert e.regime(1.0) == (L.FULL, "unlimited")
    out = e.estimate(1.0, p_cs=4000.0, k_sky=0.9)
    assert out["w"] == 2500.0 and out["basis"] == "measured" and out["untapped_w"] == 0.0


def test_discharging_needs_to_hold():
    e = _est()
    e.update("pv_w", 800.0, 0.0)
    e.update("bat_w", 400.0, 0.0)                 # helping
    assert e.regime(10.0)[0] == L.THROTTLED       # not yet held long enough
    assert e.regime(25.0) == (L.FULL, "discharging")
    e.update("bat_w", -200.0, 26.0)               # charging again, battery full (no SOC): held back
    assert e.regime(27.0) == (L.THROTTLED, "held_back")


def test_hungry_battery_and_charge_limit():
    e = _est(charge_limit_w=3000.0)
    e.update("pv_w", 2000.0, 0.0)
    e.update("soc", 60.0, 0.0)
    e.update("bat_w", -1000.0, 0.0)               # charging well below its limit
    assert e.regime(1.0) == (L.FULL, "battery_hungry")
    e.update("bat_w", -2900.0, 2.0)               # near its limit: maybe the battery is the bottleneck
    assert e.regime(3.0)[0] == L.THROTTLED
    e.update("bat_w", -1000.0, 4.0)
    e.update("soc", 95.0, 4.0)                    # nearly full: tapering, can't tell
    assert e.regime(5.0)[0] == L.THROTTLED


def test_external_source_blocks_the_battery_rule_off_grid():
    e = _est()
    e.update("pv_w", 800.0, 0.0)
    e.update("bat_w", 400.0, 0.0)
    e.update("grid_v", 230.0, 0.0)                # grid present: it may be what helps
    assert e.regime(30.0) == (L.THROTTLED, "external_source")
    e.update("grid_v", 30.0, 31.0)
    assert e.regime(32.0) == (L.FULL, "discharging")


@pytest.mark.parametrize(("limit", "grid", "full"), [(5000.0, -3000.0, True), (5000.0, -4950.0, False),
                                                      (0.0, 200.0, True), (0.0, -10.0, False)])
def test_export_limit(limit, grid, full):
    e = _est("export_limited", export_limit_w=limit)
    e.update("pv_w", 6000.0, 0.0)
    e.update("grid_w", grid, 0.0)
    assert (e.regime(1.0)[0] == L.FULL) is full


def test_voltage_band_needs_a_full_window():
    e = _est(v_full_max=400.0)
    e.update("pv_w", 1500.0, 0.0)
    e.update("pv_v", 380.0, 0.0)
    assert e.regime(30.0)[0] == L.THROTTLED       # window not yet known
    assert e.regime(61.0) == (L.FULL, "voltage")
    e.update("pv_v", 450.0, 70.0)                 # pushed toward open circuit: held back
    e.update("pv_v", 385.0, 80.0)
    assert e.regime(100.0)[0] == L.THROTTLED      # the 450 V is still inside the window
    assert e.regime(141.0) == (L.FULL, "voltage")


def test_estimate_carries_the_sky_model_error_and_fades():
    e = _est()
    e.update("pv_w", 2000.0, 0.0)
    e.update("bat_w", 300.0, 0.0)
    full = e.estimate(30.0, p_cs=4000.0, k_sky=0.4)       # measured k 0.5: the sky model was 25 % low
    assert full["basis"] == "measured" and full["w"] == 2000.0
    e.update("bat_w", -50.0, 31.0)
    e.update("pv_w", 1200.0, 31.0)                        # held back
    soon = e.estimate(60.0, p_cs=4000.0, k_sky=0.4)
    assert soon["basis"] == "estimated" and soon["regime"] == L.THROTTLED
    assert soon["w"] == pytest.approx(0.4 * 1.25 ** math.exp(-30 / L.TAU_S) * 4000.0)
    assert soon["untapped_w"] == pytest.approx(soon["w"] - 1200.0)
    later = e.estimate(30.0 + 80 * 60, p_cs=4000.0, k_sky=0.4)
    assert 1600.0 < later["w"] < soon["w"]                # most of the correction has faded
    gone = e.estimate(30.0 + 91 * 60, p_cs=4000.0, k_sky=0.4)
    assert gone["basis"] == "sky_model" and gone["w"] == pytest.approx(1600.0)
    e.update("pv_w", 1900.0, 30.0 + 92 * 60)              # never below what the panels give
    assert e.estimate(30.0 + 92 * 60, p_cs=4000.0, k_sky=0.4)["w"] == 1900.0


def test_a_dim_full_output_moment_does_not_lower_the_estimate():
    e = _est()
    e.update("pv_w", 1000.0, 0.0)
    e.update("bat_w", 300.0, 0.0)
    e.estimate(25.0, p_cs=4000.0, k_sky=0.5)              # k 0.25: a cloud while the battery helped
    e.update("bat_w", 0.0, 26.0)
    e.update("pv_w", 900.0, 26.0)
    assert e.estimate(60.0, p_cs=4000.0, k_sky=0.5)["w"] == pytest.approx(2000.0)   # the sky model's value


def test_voltage_band_learned_from_live_windows():
    rng = np.random.default_rng(0)
    band = L.VoltageBand()
    for v in rng.normal(400, 20, 300):
        band.add("full", v)
    assert band.edge is None                                     # nothing surely held back yet
    for v in rng.normal(450, 8, 300):
        band.add("held", v)
    assert band.edge == pytest.approx(np.quantile(np.asarray(band.held), L.BAND_MAX_LEAK), abs=0.1)
    assert 425 < band.edge < 440
    assert L.VoltageBand.from_dict(band.to_dict()).edge == band.edge
    blurred = L.VoltageBand()                                    # voltage says nothing on this system
    for v in rng.normal(450, 8, 300):
        blurred.add("full", v)
        blurred.add("held", v)
    assert blurred.edge is None


def test_estimator_samples_the_band_and_uses_it():
    e = _est()
    t = 0.0

    def run(minutes, pv_v, bat, soc):
        nonlocal t
        e.update("bat_w", bat, t)
        e.update("soc", soc, t)
        for _ in range(minutes):
            e.update("pv_v", pv_v, t)
            e.estimate(t, p_cs=5000.0, k_sky=0.8)
            t += 60.0

    e.update("pv_w", 2000.0, t)
    run(L.BAND_MIN + 10, 395.0, 500.0, 60.0)                     # discharging: full output, MPP voltage
    run(L.BAND_MIN + 10, 455.0, 0.0, 100.0)                      # full and idle: held back, high voltage
    assert min(len(e.band.full), len(e.band.held)) >= L.BAND_MIN and e.band.edge == pytest.approx(455.0)
    held = len(e.band.held)
    run(2, 395.0, 0.0, 100.0)
    assert e.regime(t - 60.0) == (L.FULL, "voltage")
    assert len(e.band.held) == held + 2                          # still sampled: else the edge would creep up
    assert _est(v_full_max=380.0).cfg.v_full_max == 380.0        # a configured edge wins over the live one
    e2 = _est()
    e2.restore(e.to_dict())
    assert e2.band.edge == e.band.edge


def test_night_unknown_and_restore():
    e = _est()
    assert e.estimate(0.0, p_cs=4000.0, k_sky=0.5)["basis"] == "sky_model"   # no PV reading yet
    assert e.estimate(0.0, p_cs=4000.0, k_sky=0.5)["untapped_w"] is None
    assert e.estimate(0.0, p_cs=0.0, k_sky=None)["basis"] == "night"
    e.update("pv_w", 1000.0, 0.0)
    e.update("bat_w", 200.0, 0.0)
    e.estimate(25.0, p_cs=2000.0, k_sky=0.4)
    e2 = _est()
    e2.restore(e.to_dict())
    assert e2.last_full == pytest.approx((25.0, 0.5, 0.4))
    e.update("pv_w", None, 26.0)                          # sensor unavailable
    assert e.regime(27.0) == (L.UNKNOWN, "no_pv_reading")


def test_day_log_buckets_and_known():
    log = L.DayLog()
    t0 = 1_700_000_000.0 // 3600 * 3600
    log.step(t0, 3000.0, 1000.0)
    for i in range(1, 121):                                # 60 s steps for two hours
        log.step(t0 + 60 * i, 3000.0, 1000.0)
    log.step(t0 + 7200 + 600, 3000.0, 1000.0)              # a 10-minute gap: not counted
    k = log.known(np.array([t0, t0 + 3600, t0 + 7200]))
    assert k["covered_s"].tolist() == [3600.0, 3600.0, 0.0]
    assert k["potential_wh"][0] == pytest.approx(3000.0) and k["used_wh"][0] == pytest.approx(1000.0)
    assert log.wasted_kwh(t0, t0 + 86400) == pytest.approx(4.0)
    again = L.DayLog.from_dict(log.to_dict())
    assert again.known([t0])["potential_wh"][0] == pytest.approx(3000.0)


def test_merge_known():
    live = {"covered_s": np.array([600.0, 3600.0]), "potential_wh": np.array([100.0, 900.0]),
            "used_wh": np.array([100.0, 900.0])}
    stats = {"covered_s": np.array([3600.0, 3600.0]), "potential_wh": np.array([700.0, 800.0]),
             "used_wh": np.array([700.0, 800.0])}
    m = L.merge_known("unlimited", live, stats)
    assert m["potential_wh"].tolist() == [700.0, 900.0]
    assert L.merge_known("off_grid", live, stats)["potential_wh"].tolist() == [100.0, 900.0]
