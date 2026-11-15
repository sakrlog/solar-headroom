"""How fast does a new install become accurate? (README material, plan Phase 0 item 7)

A user installs on INSTALL with no model. Every week the models are retrained on what is known by
then and scored on the following week's full-output hours (hourly mean power):

  physics      kWp x forecast plane irradiance x temperature factor (what Forecast.Solar-type tools do)
  factor       physics x one learned calibration factor (sum of actual / sum of physics since install)
  ml           the sky model (sun/season + forecast features, censored GBM) trained since install only
  ml_history   the same, plus the Home Assistant history recorded before the install (here one year)
  auto         champion/challenger: the factor, until the trained ml beat it on the same out-of-sample
               hours of the last 14 days (what the integration will run)

Forecast-style features only (no satellite: the US archive starts 2025-04). Grid-tied sites, so every
daylight hour is full output; errors are relative to the measured power.

    .venv/bin/python -I research/learning_curves.py [--sites golden henderson arbuckle] [--step-days 7]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SITES, aggregate, hour_table, load_power, site_and_planes, write_result  # noqa: E402

from engine import daily as D  # noqa: E402
from engine import features as F  # noqa: E402
from engine import models as M  # noqa: E402
from engine import regimes as R  # noqa: E402

INSTALL = "2022-03-01"
HORIZON_DAYS = 364
BUCKETS = ((0, 7), (7, 14), (14, 28), (28, 56), (56, 112), (112, 224), (224, 364))
FEATURES = F.SKY + F.FC
MIN_ML_ROWS = 50            # below this the ML ladder step keeps the calibrated factor
CHAMPION_WINDOW_DAYS = 14   # the challenger is judged on its own out-of-sample record of the last 14 days
CHAMPION_MIN_HOURS = 30     # ... and needs at least this many hours of it
HOME_KWP = 8.0              # large arrays are scaled to a home-sized system (k is size-free anyway)
METHODS = ("physics", "factor", "ml", "ml_history", "auto")


def site_table(spec) -> tuple[dict, float]:
    kwp = spec.kwp if spec.kwp <= 30 else HOME_KWP
    scale = kwp / spec.kwp
    tab = hour_table(spec, kwp=kwp)
    ts, p, step = load_power(spec)
    agg = aggregate(ts, p * scale, step, tab["ts"])
    stats = {"pv_w": agg["mean"], "pv_w_min": agg["min"], "pv_w_max": agg["max"]}
    lab = R.label(stats, R.RegimeConfig(kind="unlimited"), sun_el=tab["sun_el"])
    tab.update(pv_w=agg["mean"], regime=lab["regime"])
    _, planes = site_and_planes(spec, kwp)
    tab["physics_w"] = D.physics_power(tab["fc_gti"], tab["fc_temperature_2m"], planes)
    return tab, kwp


def curve(tab: dict, kwp: float, install: str, tz: str, step_days: int) -> dict[str, np.ndarray]:
    cfg = M.ModelConfig.for_kwp(kwp)
    tab.update(M.targets(tab, cfg))
    t_inst = pd.Timestamp(install, tz=tz).timestamp()
    ts, ok = tab["ts"], tab["trainable"]
    pv, phys = tab["pv_w"], tab["physics_w"]
    rows = {"day": [], "pv": [], **{m: [] for m in METHODS}}
    hist_err = {"factor": [], "ml": []}           # abs errors of earlier out-of-sample weeks
    hist_ts: list[np.ndarray] = []
    hist_real: list[np.ndarray] = []               # was ML a trained model (not the factor fallback)?
    for w in range(0, HORIZON_DAYS, step_days):
        a = t_inst + w * 86400
        b = a + step_days * 86400
        test = ok & (ts >= a) & (ts < b)
        if not test.any():
            continue
        since = ok & (ts >= t_inst) & (ts < a)
        before = ok & (ts < a)
        pred = {"physics": phys[test]}
        denom = phys[since].sum()
        factor = pv[since].sum() / denom if since.sum() >= 10 and denom > 0 else 1.0
        pred["factor"] = phys[test] * factor
        ml_real = since.sum() >= MIN_ML_ROWS
        if ml_real:
            pred["ml"] = M.predict_k(M.fit(tab, since, FEATURES, cfg), tab, test) * tab["p_cs"][test]
        else:
            pred["ml"] = pred["factor"]
        pred["ml_history"] = M.predict_k(M.fit(tab, before, FEATURES, cfg), tab, test) * tab["p_cs"][test]
        # champion/challenger: the factor stays unless the trained ML beat it on the same recent hours
        pred["auto"] = pred["factor"]
        if hist_ts and ml_real:
            recent = (np.concatenate(hist_ts) >= a - CHAMPION_WINDOW_DAYS * 86400) & np.concatenate(hist_real)
            if recent.sum() >= CHAMPION_MIN_HOURS and \
                    np.concatenate(hist_err["ml"])[recent].mean() < np.concatenate(hist_err["factor"])[recent].mean():
                pred["auto"] = pred["ml"]
        hist_ts.append(ts[test])
        hist_real.append(np.full(test.sum(), ml_real))
        hist_err["factor"].append(np.abs(pred["factor"] - pv[test]))
        hist_err["ml"].append(np.abs(pred["ml"] - pv[test]))
        rows["day"].append((ts[test] - t_inst) / 86400)
        rows["pv"].append(pv[test])
        for m in METHODS:
            rows[m].append(pred[m])
    return {k: np.concatenate(v) for k, v in rows.items()}


def bucket_table(res: dict[str, np.ndarray], kwp: float) -> list[dict]:
    out = []
    for lo, hi in BUCKETS:
        m = (res["day"] >= lo) & (res["day"] < hi)
        if not m.any():
            continue
        row = {"days": f"{lo}-{hi}", "hours": int(m.sum()),
               "mean_w_per_kwp": round(float(res["pv"][m].mean() / kwp), 1)}
        for meth in METHODS:
            e = res[meth][m] - res["pv"][m]
            row[f"{meth}_mae_w_per_kwp"] = round(float(np.mean(np.abs(e)) / kwp), 1)
            row[f"{meth}_mae_pct"] = round(float(100 * np.sum(np.abs(e)) / np.sum(res["pv"][m])), 1)
            row[f"{meth}_bias_pct"] = round(float(100 * np.sum(e) / np.sum(res["pv"][m])), 1)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--step-days", type=int, default=7)
    args = ap.parse_args()
    report = {"install": INSTALL, "step_days": args.step_days, "features": FEATURES, "sites": {}}
    pooled = []
    for key in args.sites:
        spec = SITES[key]
        t0 = time.time()
        tab, kwp = site_table(spec)
        n_bad = int(np.sum(tab["regime"] == R.BAD))
        res = curve(tab, kwp, INSTALL, spec.tz, args.step_days)
        table = bucket_table(res, kwp)
        report["sites"][key] = {"location": spec.location, "kwp_model": kwp, "kwp_listed": spec.kwp,
                                "tilt": spec.tilt, "azimuth": spec.azimuth, "climate": spec.climate,
                                "bad_hours": n_bad, "note": spec.note, "buckets": table}
        pooled.append({k: (v / kwp if k not in ("day",) else v) for k, v in res.items()})
        print(f"\n== {key} ({spec.location}), {time.time() - t0:.0f} s; MAE % of mean output")
        print_table(table)
    res = {k: np.concatenate([p[k] for p in pooled]) for k in pooled[0]}
    report["pooled"] = bucket_table(res, 1.0)
    print("\n== pooled (per kWp)")
    print_table(report["pooled"])
    path = write_result("learning_curves", report)
    print(f"\nwritten {path}")


def print_table(table: list[dict]) -> None:
    print(f"  {'days':>8} {'hours':>6}  " + "  ".join(f"{m:>10}" for m in METHODS))
    for r in table:
        print(f"  {r['days']:>8} {r['hours']:>6}  " + "  ".join(f"{r[m + '_mae_pct']:>9.1f}%" for m in METHODS))


if __name__ == "__main__":
    main()
