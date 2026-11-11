"""engine.regimes on synthetic periods (signs: battery + = discharging, grid + = importing)."""
from __future__ import annotations

import numpy as np
import pytest
from engine import regimes as R


def periods(**cols):
    n = len(next(iter(cols.values())))
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    out["pv_w"] = np.asarray(cols.get("pv_w", [1000] * n), float)
    return out


def test_unlimited_every_daylight_period_is_full_output():
    s = periods(pv_w=[0, 5, 300, 4000])
    out = R.label(s, R.RegimeConfig(kind="unlimited"))
    assert list(out["regime"]) == ["night", "night", "mpp", "mpp"]


def test_off_grid_battery_rule():
    s = periods(pv_w=[2000, 2000, 2000, 2000, 2000],
                bat_w_min=[200, -1500, -1500, 0, -6000],      # helping / charging / charging / idle / charging fast
                soc_max=[50, 60, 95, 100, 60])
    out = R.label(s, R.RegimeConfig(kind="off_grid", charge_limit_w=8000))
    assert list(out["regime"]) == ["mpp", "mpp", "limited", "limited", "limited"]
    assert list(out["reason"][:2]) == ["discharging", "battery_hungry"]


def test_off_grid_battery_rule_needs_external_source_absent():
    s = periods(pv_w=[2000, 2000, 2000], bat_w_min=[200, 200, 200], soc_max=[50, 50, 50],
                grid_v_max=[25, 230, np.nan], gen_w_max=[0, 0, 900])
    out = R.label(s, R.RegimeConfig(kind="off_grid"))
    assert list(out["regime"]) == ["mpp", "limited", "limited"]
    assert list(out["ext_any"]) == [False, True, True]


def test_export_limited():
    s = periods(pv_w=[3000] * 4, grid_w_min=[-1000, -4950, 300, -200])
    out = R.label(s, R.RegimeConfig(kind="export_limited", export_limit_w=5000))
    assert list(out["regime"]) == ["mpp", "limited", "mpp", "mpp"]
    zero = R.label(s, R.RegimeConfig(kind="export_limited", export_limit_w=0))
    assert list(zero["regime"]) == ["limited", "limited", "mpp", "limited"]   # only importing proves full output


def test_voltage_rule_and_curtailment_override():
    s = periods(pv_w=[2000, 2000, 2000, 50], pv_v_max=[410, 470, 410, 300], curtail_frac=[0, 0, 0.2, 0])
    out = R.label(s, R.RegimeConfig(kind="off_grid", v_full_max=430))
    assert list(out["regime"]) == ["mpp", "limited", "limited", "limited"]   # 50 W: too little current to judge
    assert out["reason"][2] == "curtailed"


def test_bad_data_flags():
    s = periods(pv_w=[800, 900, 60], pv_w_min=[800, 100, 60], pv_w_max=[800, 1500, 60])
    out = R.label(s, R.RegimeConfig(kind="unlimited"), sun_el=np.array([30, 30, -10]))
    assert list(out["regime"]) == ["bad", "mpp", "bad"]          # frozen value; PV at night


def test_learn_charge_limit():
    rng = np.random.default_rng(0)
    b = -np.minimum(rng.exponential(1500, 2000), 5000)              # charging saturates at 5 kW
    assert R.learn_charge_limit(b) == pytest.approx(5000, rel=0.01)
    assert R.learn_charge_limit(b[:10]) is None


def test_learn_voltage_band_separates_throttled_from_full_output():
    rng = np.random.default_rng(1)
    n_full, n_thr = 600, 300
    v = np.concatenate([rng.normal(405, 8, n_full), rng.normal(465, 10, n_thr)])
    regime = np.array(["mpp"] * n_full + ["limited"] * n_thr, dtype=object)
    reason = np.array(["discharging"] * n_full + [""] * n_thr, dtype=object)
    pv = np.full(n_full + n_thr, 2000.0)
    soc = np.concatenate([np.full(n_full, 60.0), np.full(n_thr, 100.0)])
    bmin = np.concatenate([np.full(n_full, 100.0), np.full(n_thr, -20.0)])
    bmax = np.concatenate([np.full(n_full, 500.0), np.full(n_thr, 10.0)])
    out = R.learn_voltage_band(v, regime, reason, pv, soc, bmin, bmax)
    assert 425 < out["v_full_max"] < 445 and out["coverage"] > 0.9
    # overlapping bands: the rule must refuse rather than mislabel
    v_bad = np.concatenate([rng.normal(440, 15, n_full), rng.normal(445, 15, n_thr)])
    assert R.learn_voltage_band(v_bad, regime, reason, pv, soc, bmin, bmax)["v_full_max"] is None


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        R.RegimeConfig(kind="solar_farm")


def test_auto_label_learns_thresholds_per_segment():
    """Two eras with different string voltages (e.g. before/after a rewiring): each learns its own band."""
    rng = np.random.default_rng(4)
    parts, segs = [], []
    for s, (v_full, v_thr) in enumerate(((205, 250), (405, 465))):
        n_full, n_thr, n_mid = 300, 200, 100
        parts.append({
            # battery-rule full output | battery full and idle (throttled) | grid on: only the voltage rule can tell
            "pv_v_max": np.concatenate([rng.normal(v_full, 5, n_full), rng.normal(v_thr, 6, n_thr),
                                        rng.normal(v_full, 5, n_mid)]),
            "soc_max": np.concatenate([np.full(n_full, 60.0), np.full(n_thr, 100.0), np.full(n_mid, 95.0)]),
            "bat_w_min": np.concatenate([np.full(n_full, 100.0), np.full(n_thr, -20.0), np.full(n_mid, -20.0)]),
            "bat_w_max": np.concatenate([np.full(n_full, 500.0), np.full(n_thr, 10.0), np.full(n_mid, 10.0)]),
            "grid_v_max": np.concatenate([np.zeros(n_full + n_thr), np.full(n_mid, 230.0)]),
        })
        segs.append(np.full(n_full + n_thr + n_mid, float(s)))
    stats = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    stats["pv_w"] = np.full(len(stats["soc_max"]), 2000.0)
    out, learned = R.auto_label(stats, R.RegimeConfig(kind="off_grid"), segment=np.concatenate(segs))
    # the edge sits just under the throttled cluster (its 1 % quantile)
    assert 225 < learned[0.0]["v_full_max"] < 245 and 440 < learned[1.0]["v_full_max"] < 460
    for s in (0, 1):
        o = s * 600
        assert (out["regime"][o:o + 300] == "mpp").all()
        assert (out["regime"][o + 300:o + 500] == "limited").mean() >= 0.98    # max_leak: ~1 % by design
        assert (out["reason"][o + 500:o + 600] == "voltage").mean() > 0.95
    fixed, _ = R.auto_label(stats, R.RegimeConfig(kind="off_grid", v_full_max=300), segment=np.concatenate(segs))
    assert (fixed["regime"][1100:1200] == "limited").all()        # a given threshold is kept, not learned
