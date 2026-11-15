"""Does the censored learner recover the hidden potential of a throttled system? (plan Phase 0 item 7)

Clean output of real grid-tied systems is the hidden truth. An off-grid home is simulated on top of
it in 5-minute steps (15 for a meter): a battery that tapers near full, a charge-power limit and a varying
household load; with nowhere to send it, PV is cut back to what the house and battery take. Home
Assistant-style hourly statistics (mean/min/max of PV, battery power and state of charge) are labelled
by engine.regimes exactly as in the integration, then these learners are compared on held-out months:

  oracle        trained on the true potential of every hour (unreachable upper bound)
  censored      full-output hours exact, throttled hours as floors, all features, no leak guard
  guarded       engine default: censored + the leak guard (engine.models: blends toward weather_only
                when the time-of-day features carry the throttling selection)
  weather_only  censored without the time-of-day / sun-geometry features
  excluded      full-output hours only, throttled hours dropped (what forecast tools do when calibrating)
  naive         every daylight hour taken at face value

Scores are against the hidden truth on every daylight hour, on the throttled hours alone, and for the
daily "solar bank" (measured where full output, max(measured, estimate) where throttled).

    .venv/bin/python -I research/throttling.py [--sites golden henderson arbuckle]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SITES, aggregate, hour_table, load_power, site_and_planes, write_result  # noqa: E402

from engine import daily as D  # noqa: E402
from engine import features as F  # noqa: E402
from engine import models as M  # noqa: E402
from engine import regimes as R  # noqa: E402

FEATURES = F.SKY + F.FC
HOME_KWP = 8.0
LEARNERS = ("oracle", "censored", "guarded", "weather_only", "excluded", "naive")
THROTTLED_FRAC = 0.02          # an hour counts as throttled when PV fell >2 % short of the truth


@dataclass(frozen=True)
class OffGrid:
    """A home off-grid system sized relative to its kWp (no grid, no generator)."""
    battery_kwh_per_kwp: float = 1.6
    charge_w_per_kwp: float = 500.0
    discharge_w_per_kwp: float = 1000.0
    soc_min: float = 10.0
    taper_from: float = 95.0       # charge acceptance falls linearly to 0 between here and 100 %
    efficiency: float = 0.95       # one way
    soc_start: float = 60.0


def household_load(ts, step_s: int, tz: str, kwp: float, seed: int = 0) -> np.ndarray:
    """A varying household load (W): standby, morning and evening peaks, daytime use, appliance
    bursts and summer cooling; ~10-14 kWh/day for a 6 kWp home."""
    rng = np.random.default_rng(seed)
    ts = np.asarray(ts, float)
    loc = pd.DatetimeIndex(pd.to_datetime(ts + step_s / 2, unit="s", utc=True)).tz_convert(tz)
    h = (loc.hour + loc.minute / 60).to_numpy(float)
    month = loc.month.to_numpy()
    day = ((ts - ts[0]) // 86400).astype(int)
    n_days = int(day.max()) + 1

    def bump(center, width):
        return np.exp(-0.5 * ((h - center) / width) ** 2)
    level = rng.uniform(0.8, 1.2, n_days)[day]
    load = 180 + 500 * bump(7.5, 1.0) + 180 * ((h >= 9) & (h < 17)) + 800 * bump(20.0, 1.8)
    cooling = np.where(np.isin(month, [6, 7, 8, 9]) & (h >= 12) & (h < 19), 700 * rng.uniform(0, 1, n_days)[day], 0)
    awake = (h >= 7) & (h < 22)
    bursts = (rng.random(len(ts)) < 0.25 * step_s / 3600) & awake
    return (kwp / 6.0) * (load * level + cooling + bursts * 2000 * min(1.0, 300 / step_s))


def simulate(potential_w, load_w, step_s: int, kwp: float, sys_: OffGrid) -> dict[str, np.ndarray]:
    """Step-by-step battery simulation. Missing potential (data gaps) counts as no sun for the battery
    and leaves the outputs missing. Battery power: + discharging, - charging (HA Energy convention)."""
    p_in = np.asarray(potential_w, float)
    load = np.asarray(load_w, float)
    n = len(p_in)
    cap = sys_.battery_kwh_per_kwp * kwp * 3.6e6            # J
    charge_w, discharge_w = sys_.charge_w_per_kwp * kwp, sys_.discharge_w_per_kwp * kwp
    e_min = sys_.soc_min / 100 * cap
    eff = sys_.efficiency
    e = sys_.soc_start / 100 * cap
    pv, bat, soc = np.empty(n), np.empty(n), np.empty(n)
    p_list, l_list = p_in.tolist(), load.tolist()          # plain floats: ~10x faster loop
    for i in range(n):
        p = p_list[i]
        gap = p != p
        p = 0.0 if gap else p
        s = 100.0 * e / cap
        taper = min(1.0, max(0.0, (100.0 - s) / (100.0 - sys_.taper_from)))
        accept = min(charge_w * taper, (cap - e) / (eff * step_s))
        L = l_list[i]
        if p >= L:
            used = min(p, L + accept)
            b = -(used - L)
            e += -b * eff * step_s
        else:
            used = p
            b = min(L - p, discharge_w, max(e - e_min, 0.0) * eff / step_s)
            e -= b / eff * step_s
        pv[i], bat[i], soc[i] = (np.nan, np.nan, np.nan) if gap else (used, b, 100.0 * e / cap)
    return {"pv": pv, "bat": bat, "soc": soc}


def prepare(spec, sys_: OffGrid) -> dict:
    """Hour table of a simulated throttled system: features, HA-style statistics, engine labels,
    hidden truth, training targets and the physics baseline."""
    kwp = spec.kwp if spec.kwp <= 30 else HOME_KWP
    tab = hour_table(spec, kwp=kwp)
    ts, p, step = load_power(spec)
    p = p * kwp / spec.kwp
    load = household_load(ts, step, spec.tz, kwp)
    sim = simulate(p, load, step, kwp, sys_)
    truth = aggregate(ts, p, step, tab["ts"])["mean"]
    st = {}
    for name, v in (("pv_w", sim["pv"]), ("bat_w", sim["bat"]), ("soc", sim["soc"])):
        a = aggregate(ts, v, step, tab["ts"])
        st[name], st[f"{name}_min"], st[f"{name}_max"] = a["mean"], a["min"], a["max"]
    lab, learned = R.auto_label(st, R.RegimeConfig(kind="off_grid"), sun_el=tab["sun_el"])
    tab.update(st)
    tab.update(regime=lab["regime"], reason=lab["reason"], truth=truth)
    cfg = M.ModelConfig.for_kwp(kwp)
    tab.update(M.targets(tab, cfg))
    _, planes = site_and_planes(spec, kwp)
    tab["physics_w"] = D.physics_power(tab["fc_gti"], tab["fc_temperature_2m"], planes)
    tab["daylight"] = np.isin(tab["regime"], [R.MPP, R.LIMITED]) & ~np.isnan(truth)
    tab["short"] = (truth - tab["pv_w"]) > np.maximum(THROTTLED_FRAC * truth, 20.0)
    tab["eval"] = tab["daylight"] & (tab["p_cs"] >= cfg.min_p_cs_w)
    tab["day"] = D.local_day(tab["ts"], spec.tz)
    return {"tab": tab, "cfg": cfg, "kwp": kwp, "learned": learned}


def label_quality(tab: dict, learned: dict) -> dict:
    daylight, short, regime = tab["daylight"], tab["short"], tab["regime"]
    return {"daylight_hours": int(daylight.sum()),
            "mpp_pct": pct(regime[daylight] == R.MPP),
            "limited_pct": pct(regime[daylight] == R.LIMITED),
            "throttled_pct": pct(short[daylight]),
            "mpp_but_throttled_pct": pct(short[daylight & (regime == R.MPP)]),
            "limited_but_free_pct": pct(~short[daylight & (regime == R.LIMITED)]),
            "wasted_energy_pct": round(float(100 * np.nansum((tab["truth"] - tab["pv_w"])[daylight])
                                             / np.nansum(tab["truth"][daylight])), 1),
            "learned": learned}


def run_learners(tab: dict, cfg: M.ModelConfig, folds_n: int = 6,
                 features=FEATURES) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Held-out-month predictions (W) of every learner, and the leak guard of each fold."""
    # the oracle sees the truth everywhere; the others see what the inverter delivered
    orc = dict(tab, pv_w=tab["truth"], regime=np.where(tab["daylight"], R.MPP, tab["regime"]).astype(object))
    orc.update(M.targets(orc, cfg))
    folds = M.group_folds(M.month_ids(tab["ts"]), folds_n)
    pred = {k: np.full(len(tab["ts"]), np.nan) for k in LEARNERS}
    weather = [f for f in features if f not in M.CLOCK]
    guards = []
    for test in folds:
        train = ~test
        rows = train & tab["trainable"]
        pred["oracle"][test] = M.predict_k(M.fit(orc, train & orc["trainable"], features, cfg), tab, test)
        pred["censored"][test] = M.predict_k(M.fit(tab, rows, features, cfg, guard=False), tab, test)
        guarded = M.fit(tab, rows, features, cfg)
        pred["guarded"][test] = M.predict_k(guarded, tab, test)
        if isinstance(guarded, M.Guarded):
            guards.append({"ratio": round(guarded.ratio, 3), "weight": round(guarded.weight, 2)})
        pred["weather_only"][test] = M.predict_k(M.fit(tab, rows, weather, cfg), tab, test)
        mpp_rows = train & tab["trainable"] & (tab["regime"] == R.MPP)
        pred["excluded"][test] = M.predict_k(M.fit(tab, mpp_rows, features, cfg), tab, test)
        naive = M.fit(tab, train & tab["trainable"], features, cfg, censored=False)
        pred["naive"][test] = M.predict_k(naive, tab, test)
    out = {k: v * tab["p_cs"] for k, v in pred.items()}
    out["physics"] = tab["physics_w"]
    return out, guards


def score(tab: dict, pred_w: np.ndarray) -> dict:
    """Against the hidden truth: every daylight hour, the throttled hours, and the daily solar bank
    (measured where full output, max(measured, estimate) where throttled)."""
    truth, ev = tab["truth"], tab["eval"]
    e = pred_w - truth
    thr = ev & tab["short"]
    bank = np.where(tab["regime"] == R.MPP, tab["pv_w"], np.fmax(tab["pv_w"], pred_w))
    return {"all": err(e[ev], truth[ev]), "throttled": err(e[thr], truth[thr]),
            "daily_bank": daily_err(bank, truth, tab["daylight"], tab["day"])}


def site_run(spec, sys_: OffGrid, folds_n: int = 6) -> dict:
    prep = prepare(spec, sys_)
    tab, cfg = prep["tab"], prep["cfg"]
    t0 = time.time()
    pred, guards = run_learners(tab, cfg, folds_n)
    fit_s = time.time() - t0
    return {"location": spec.location, "kwp_model": prep["kwp"], "climate": spec.climate, "system": asdict(sys_),
            "labels": label_quality(tab, prep["learned"]), "fit_s": round(fit_s, 1), "guard_by_fold": guards,
            "scores": {k: score(tab, v) for k, v in pred.items()}}


def pct(mask) -> float:
    mask = np.asarray(mask, bool)
    return round(float(100 * mask.mean()), 1) if mask.size else float("nan")


def err(e, truth) -> dict:
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "mae_pct": round(float(100 * np.mean(np.abs(e)) / np.mean(truth)), 1),
            "bias_pct": round(float(100 * np.mean(e) / np.mean(truth)), 1)}


def daily_err(bank, truth, daylight, days) -> dict:
    """Days whose every daylight hour has data: estimated vs true potential energy."""
    ok = np.where(daylight, ~np.isnan(bank), True)
    u, inv = np.unique(days, return_inverse=True)
    complete = np.bincount(inv, weights=(~ok).astype(float), minlength=len(u)) == 0
    est = np.bincount(inv, weights=np.where(daylight, np.nan_to_num(bank), 0.0), minlength=len(u))
    tru = np.bincount(inv, weights=np.where(daylight, np.nan_to_num(truth), 0.0), minlength=len(u))
    m = complete & (tru > 0)
    e = est[m] - tru[m]
    return {"days": int(m.sum()), "mae_pct": round(float(100 * np.mean(np.abs(e)) / np.mean(tru[m])), 1),
            "bias_pct": round(float(100 * np.mean(e) / np.mean(tru[m])), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--taper-from", type=float, default=95.0, help="SOC where charging starts to taper")
    args = ap.parse_args()
    sys_ = OffGrid(taper_from=args.taper_from)
    report = {"features": FEATURES, "throttled_frac": THROTTLED_FRAC, "sites": {}}
    for key in args.sites:
        t0 = time.time()
        r = site_run(SITES[key], sys_)
        report["sites"][key] = r
        lb = r["labels"]
        print(f"\n== {key} ({r['location']}), {time.time() - t0:.0f} s (fits {r['fit_s']} s)")
        print(f"  daylight hours {lb['daylight_hours']}: full output {lb['mpp_pct']}%, throttled by the truth "
              f"{lb['throttled_pct']}%, energy wasted {lb['wasted_energy_pct']}%")
        print(f"  labelled full output but throttled: {lb['mpp_but_throttled_pct']}%; labelled throttled but "
              f"free: {lb['limited_but_free_pct']}%; learned {lb['learned']}")
        print("  leak guard per fold (ratio, weight): "
              + ", ".join(f"({g['ratio']}, {g['weight']})" for g in r["guard_by_fold"]))
        print(f"  {'learner':<12} {'all hours':>18} {'throttled hours':>18} {'daily bank':>18}   (MAE% / bias%)")
        for k, s in r["scores"].items():
            cells = [f"{s[g].get('mae_pct', np.nan):>8.1f} / {s[g].get('bias_pct', np.nan):>+6.1f}"
                     for g in ("all", "throttled", "daily_bank")]
            print(f"  {k:<12} " + " ".join(cells))
    suffix = "" if args.taper_from == 95.0 else f"_taper{int(args.taper_from)}"
    print(f"\nwritten {write_result('throttling' + suffix, report)}")


if __name__ == "__main__":
    main()
