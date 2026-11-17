"""Does the live estimate find the hidden potential while the inverter holds back? (plan Phase 2 exit)

The simulated off-grid homes of throttling.py, on real output of public systems (the hidden truth).
At every step of the data (5 or 15 minutes) the simulated PV, battery power and state of charge go
through engine.live as the integration feeds it, with the charge limit learned from the hourly
statistics. The sky model is the engine's nowcast learner (guarded, forecast features, no satellite)
trained on the other months' hourly statistics. Rules compared against the truth:

  measured  what the inverter delivered (the floor everyone can see)
  sky       the sky model alone
  two_way   the sky model corrected by its error at the last full-output step, both directions
  upward    engine default: the correction only upward (engine.live)

Scored on the steps the truth calls throttled (PV > 2 % short), on all steps the estimate calls
held back, and as daily wasted energy (potential - used).

    .venv/bin/python -I research/live.py [--sites golden henderson arbuckle]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SITES, load_power, load_weather, site_and_planes, t_air_from, write_result  # noqa: E402
from throttling import FEATURES, HOME_KWP, THROTTLED_FRAC, OffGrid, household_load, prepare, simulate  # noqa: E402

from engine import daily as D  # noqa: E402
from engine import features as F  # noqa: E402
from engine import live as L  # noqa: E402
from engine import models as M  # noqa: E402
from engine import regimes as R  # noqa: E402

RULES = {"sky": None, "two_way": (0.25, 4.0), "upward": (1.0, 4.0)}


def sky_k(tab: dict, cfg: M.ModelConfig, folds_n: int = 6) -> np.ndarray:
    """Held-out-month clearness of the engine's guarded nowcast learner, per hour."""
    k = np.full(len(tab["ts"]), np.nan)
    for test in M.group_folds(M.month_ids(tab["ts"]), folds_n):
        model = M.fit(tab, ~test & tab["trainable"], FEATURES, cfg)
        k[test] = M.predict_k(model, tab, test)
    return k


def replay(steps: dict, cfg: R.RegimeConfig, rule: str) -> dict[str, np.ndarray]:
    """Feed the steps through engine.live; one estimate at the end of every step."""
    saved = L.RATIO_BOUNDS, L.TAU_S
    if RULES[rule] is None:
        L.TAU_S = 1e-9                                     # the correction fades at once: the sky model alone
    else:
        L.RATIO_BOUNDS = RULES[rule]
    try:
        est = L.LiveEstimator(cfg)
        n = len(steps["ts"])
        w, full = np.full(n, np.nan), np.zeros(n, bool)
        ts, step = steps["ts"], steps["step"]
        for i in range(n):
            if np.isnan(steps["pv"][i]):
                continue
            for role in ("pv_w", "bat_w", "soc"):
                est.update(role, float(steps[role][i]), ts[i])
            k = steps["k_sky"][i]
            e = est.estimate(ts[i] + step, float(steps["p_cs"][i]), None if np.isnan(k) else float(k))
            w[i] = np.nan if e["w"] is None else e["w"]
            full[i] = e["regime"] == L.FULL
        return {"w": w, "full": full}
    finally:
        L.RATIO_BOUNDS, L.TAU_S = saved


def err(e, truth) -> dict:
    ok = ~np.isnan(e) & ~np.isnan(truth)
    if not ok.any():
        return {"n": 0}
    return {"n": int(ok.sum()), "mae_pct": round(float(100 * np.mean(np.abs(e[ok])) / np.mean(truth[ok])), 1),
            "bias_pct": round(float(100 * np.mean(e[ok]) / np.mean(truth[ok])), 1)}


def site_run(spec, sys_: OffGrid) -> dict:
    t0 = time.time()
    prep = prepare(spec, sys_)
    tab, cfg, kwp = prep["tab"], prep["cfg"], prep["kwp"]
    k_hour = sky_k(tab, cfg)
    # the native steps, simulated exactly as prepare() did for its hourly statistics
    ts, p, step = load_power(spec)
    p = p * kwp / spec.kwp
    sim = simulate(p, household_load(ts, step, spec.tz, kwp), step, kwp, sys_)
    site, planes = site_and_planes(spec, kwp)
    t_air = t_air_from(load_weather(spec))
    sky = F.sky_features(ts, float(step), 3, site, planes, t_air=t_air)
    hour = np.searchsorted(tab["ts"], ts, side="right") - 1
    inside = (hour >= 0) & (ts < tab["ts"][np.maximum(hour, 0)] + 3600)
    steps = {"ts": np.asarray(ts, float), "step": float(step), "pv": sim["pv"], "pv_w": sim["pv"], "bat_w": sim["bat"],
             "soc": sim["soc"], "p_cs": sky["p_cs"], "k_sky": np.where(inside, k_hour[np.maximum(hour, 0)], np.nan)}
    learned = prep["learned"].get(0.0, {})
    rcfg = R.RegimeConfig(kind="off_grid", charge_limit_w=learned.get("charge_limit_w"))

    truth, pv = p, sim["pv"]
    day = (sky["p_cs"] > 0.05 * 1000 * kwp) & ~np.isnan(truth) & ~np.isnan(pv) & ~np.isnan(steps["k_sky"])
    short = day & ((truth - pv) > np.maximum(THROTTLED_FRAC * truth, 20.0))
    days = D.local_day(ts, spec.tz)
    out = {"location": spec.location, "step_min": step / 60, "kwp_model": kwp, "learned": learned,
           "daylight_steps": int(day.sum()), "throttled_pct": round(float(100 * short.sum() / day.sum()), 1),
           "rules": {}}
    runs = {r: replay(steps, rcfg, r) for r in RULES}
    full = runs["upward"]["full"]
    out["regime"] = {"called_full_pct": round(float(100 * full[day].mean()), 1),
                     "called_full_but_throttled_pct": round(float(100 * short[day & full].mean()), 1),
                     "throttled_caught_pct": round(float(100 * (~full[short]).mean()), 1)}
    true_waste = np.where(day, truth - pv, 0.0)
    u, inv = np.unique(days[day], return_inverse=True)
    tw = np.bincount(inv, weights=true_waste[day], minlength=len(u)) * step / 3.6e6
    for name, w in [("measured", pv)] + [(r, runs[r]["w"]) for r in RULES]:
        held = day & ~full
        ew = np.bincount(inv, weights=np.nan_to_num(w - pv)[day], minlength=len(u)) * step / 3.6e6
        busy = tw > 0.5                                            # days with real waste
        out["rules"][name] = {"throttled_steps": err((w - truth)[short], truth[short]),
                              "held_back_steps": err((w - truth)[held], truth[held]),
                              "daily_waste_kwh": {"days": int(busy.sum()),
                                                  "true": round(float(tw[busy].mean()), 2),
                                                  "estimated": round(float(ew[busy].mean()), 2),
                                                  "mae": round(float(np.mean(np.abs(ew[busy] - tw[busy]))), 2)}}
    out["run_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    args = ap.parse_args()
    sys_ = OffGrid()
    report = {"features": FEATURES, "throttled_frac": THROTTLED_FRAC, "home_kwp": HOME_KWP, "tau_min": L.TAU_S / 60,
              "sites": {}}
    for key in args.sites:
        r = site_run(SITES[key], sys_)
        report["sites"][key] = r
        g = r["regime"]
        print(f"\n== {key} ({r['location']}), {r['step_min']:.0f}-min steps, {r['run_s']} s: throttled "
              f"{r['throttled_pct']} % of daylight steps; learned {r['learned']}")
        print(f"   called full output {g['called_full_pct']} % (of those throttled "
              f"{g['called_full_but_throttled_pct']} %); throttled steps caught {g['throttled_caught_pct']} %")
        print(f"   {'rule':<10} {'throttled steps':>18} {'held-back steps':>18}   daily waste kWh (true / est / MAE)")
        for name, s in r["rules"].items():
            cells = [f"{s[g].get('mae_pct', np.nan):>8.1f} / {s[g].get('bias_pct', np.nan):>+6.1f}"
                     for g in ("throttled_steps", "held_back_steps")]
            dw = s["daily_waste_kwh"]
            print(f"   {name:<10} " + " ".join(cells) + f"   {dw['true']:.2f} / {dw['estimated']:.2f} / {dw['mae']:.2f}"
                  f" ({dw['days']} days)")
    print(f"\nwritten {write_result('live', report)}")


if __name__ == "__main__":
    main()
