"""What the integration does with the engine, without Home Assistant.

  archive_requests  which Open-Meteo archive chunks a system needs (calendar quarters, per panel group)
  train             stored statistics + archives -> nowcast and daily models, physics factors, report
  outputs           models + the live forecast + today's measurements -> what the sensors show

Runs in the training subprocess, in Home Assistant's executor and in tests. Weather frames keep
Open-Meteo's variable names and raw timestamps (engine.weather); panel groups after the first only
fetch their plane irradiance, which is combined by kWp.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from . import daily as D
from . import features as F
from . import history as H
from . import models as M
from . import regimes as R
from . import solar as S
from . import weather as W
from .regimes import LIMITED

HOUR = 3600.0
NOWCAST = F.SKY + F.FC
HINDSIGHT = F.SKY + F.FC                  # fills throttled / missing hours of past days
HF_START = date(2021, 3, 1)               # Open-Meteo historical-forecast archive
PR_START = date.fromisoformat(W.PREVIOUS_RUNS_START)
REFRESH_OPEN_S = 20 * 3600                # archive chunks that may still change are re-fetched after this
DEFAULT_FACTOR = 0.86                     # typical system losses (PVWatts: 14 %) until the site has data
FACTOR_DAYS = 60                          # physics factor: measured / physics over this many recent days
MIN_FACTOR_HOURS = 10
MIN_ML_HOURS = 50
HOLDOUT_DAYS = 14                         # champion/challenger: last days held out of a trial fit
MIN_HOLDOUT_HOURS = 30
MIN_HOLDOUT_DAYS = 5
RANGE_MIN_DAYS = 60                       # site-specific P10/P90 table from this many complete days
RANGE_MAX_AGE_S = 7 * 86400
GTI = "global_tilted_irradiance"
LIVE_VARS = tuple(dict.fromkeys((*W.WX_VARS, *W.DAILY_VARS)))
ENSEMBLE_VARS = (GTI, "cloud_cover")


# ---------------------------------------------------------------- the system
@dataclass(frozen=True)
class System:
    """One PV system as the integration knows it (JSON round-trip with to_dict / from_dict).

    roles: role -> parts -> sources, each {"statistic_id", "scale"=1, "since"=None, "until"=None}
    (engine.history). Phase 1 uses "pv_w"; battery, grid and voltage roles feed the throttling rules."""
    latitude: float
    longitude: float
    altitude: float
    time_zone: str
    planes: tuple[S.Plane, ...]
    roles: Mapping[str, Sequence[Sequence[Mapping]]]
    kind: str = "unlimited"
    change_points: tuple[float, ...] = ()

    @property
    def kwp(self) -> float:
        return float(sum(p.kwp for p in self.planes))

    @property
    def site(self) -> S.Site:
        return S.Site(self.latitude, self.longitude, self.altitude)

    def history_roles(self) -> dict[str, list[list[H.Source]]]:
        return {role: [[H.Source(**src) for src in part] for part in parts] for role, parts in self.roles.items()}

    def statistic_ids(self) -> list[str]:
        return H.statistic_ids(self.history_roles())

    def model_config(self) -> M.ModelConfig:
        return M.ModelConfig.for_kwp(self.kwp)

    def to_dict(self) -> dict:
        return {"latitude": self.latitude, "longitude": self.longitude, "altitude": self.altitude,
                "time_zone": self.time_zone, "kind": self.kind, "change_points": list(self.change_points),
                "planes": [{"tilt": p.tilt, "azimuth": p.azimuth, "kwp": p.kwp} for p in self.planes],
                "roles": {r: [[dict(s) for s in part] for part in parts] for r, parts in self.roles.items()}}

    @classmethod
    def from_dict(cls, d: Mapping) -> System:
        return cls(float(d["latitude"]), float(d["longitude"]), float(d.get("altitude", 0.0)), d["time_zone"],
                   tuple(S.Plane(float(p["tilt"]), float(p["azimuth"]), float(p["kwp"])) for p in d["planes"]),
                   {r: [[dict(s) for s in part] for part in parts] for r, parts in d["roles"].items()},
                   d.get("kind", "unlimited"), tuple(float(t) for t in d.get("change_points", ())))


def local_date(ts: float, tz: str) -> date:
    return datetime.fromtimestamp(ts, ZoneInfo(tz)).date()


# ---------------------------------------------------------------- Open-Meteo requests
@dataclass(frozen=True)
class Request:
    key: str            # store key; the same key is replaced when an open chunk is fetched again
    kind: str           # "hf" historical forecasts (nowcast inputs), "pr" previous runs (daily inputs), "live"
    url: str
    params: dict = field(hash=False)
    first_day: str | None = None
    last_day: str | None = None
    final: bool = False


def quarters(first: date, last: date) -> list[tuple[date, date, date]]:
    """(start, end, quarter end) of the calendar quarters overlapping [first, last], clipped to it."""
    out = []
    q = date(first.year, 3 * ((first.month - 1) // 3) + 1, 1)
    while q <= last:
        nxt = date(q.year + 1, 1, 1) if q.month == 10 else date(q.year, q.month + 3, 1)
        q_end = nxt - timedelta(days=1)
        out.append((max(q, first), min(q_end, last), q_end))
        q = nxt
    return out


def plane_tag(i: int, p: S.Plane) -> str:
    return f"{p.tilt:g}_{p.azimuth:g}/{'all' if i == 0 else 'gti'}"


def archive_requests(system: System, first_day: date, today: date) -> list[Request]:
    """Every archive chunk the system needs from first_day to yesterday, stored or not."""
    last = today - timedelta(days=1)
    lat, lon = system.latitude, system.longitude
    out: list[Request] = []
    for i, p in enumerate(system.planes):
        tag = plane_tag(i, p)
        for a, b, q_end in quarters(max(first_day, HF_START), last):
            url, params = W.historical_forecast_request(lat, lon, a.isoformat(), b.isoformat(), p.tilt, p.azimuth,
                                                        W.WX_VARS if i == 0 else (GTI,))
            out.append(Request(f"hf/{tag}/{a}", "hf", url, params, a.isoformat(), b.isoformat(),
                               W.is_final(q_end.isoformat(), today.isoformat())))
        for model, variables in W.PREVIOUS_RUNS_MODELS.items():
            for a, b, q_end in quarters(max(first_day, PR_START), last):
                url, params = W.previous_runs_request(lat, lon, a.isoformat(), b.isoformat(), p.tilt, p.azimuth,
                                                      model, variables if i == 0 else (GTI,), (0, 1))
                out.append(Request(f"pr/{model}/{tag}/{a}", "pr", url, params, a.isoformat(), b.isoformat(),
                                   W.is_final(q_end.isoformat(), today.isoformat())))
    return out


def missing(requests: Sequence[Request], index: Mapping[str, tuple[bool, float]], now: float) -> list[Request]:
    """Requests not stored yet, or stored while still open and older than REFRESH_OPEN_S."""
    out = []
    for r in requests:
        have = index.get(r.key)
        if have is None or (not have[0] and now - have[1] > REFRESH_OPEN_S):
            out.append(r)
    return out


def live_requests(system: System) -> list[Request]:
    """The current forecast (yesterday to the day after tomorrow): best match + the ensemble spread."""
    out = []
    for i, p in enumerate(system.planes):
        tag = plane_tag(i, p)
        url, params = W.forecast_request(system.latitude, system.longitude, p.tilt, p.azimuth,
                                         LIVE_VARS if i == 0 else (GTI,), past_days=1, forecast_days=3)
        out.append(Request(f"live/best_match/{tag}", "live", url, params))
        url, params = W.forecast_request(system.latitude, system.longitude, p.tilt, p.azimuth,
                                         ENSEMBLE_VARS if i == 0 else (GTI,), model=",".join(W.ENSEMBLE),
                                         past_days=1, forecast_days=3)
        out.append(Request(f"live/ensemble/{tag}", "live", url, params))
    return out


# ---------------------------------------------------------------- frames from responses
def _on(ts: np.ndarray, frame: Mapping[str, np.ndarray], col: str) -> np.ndarray:
    """frame[col] at times ts (NaN where the frame has no such time or column)."""
    out = np.full(len(ts), np.nan)
    if col not in frame or not len(frame["ts"]):
        return out
    f_ts = np.asarray(frame["ts"], float)
    i = np.clip(np.searchsorted(f_ts, ts), 0, len(f_ts) - 1)
    hit = f_ts[i] == ts
    out[hit] = np.asarray(frame[col], float)[i[hit]]
    return out


def combine_planes(frames: Sequence[Mapping[str, np.ndarray]], kwp: Sequence[float]) -> dict[str, np.ndarray]:
    """First group's frame with every plane-irradiance column (GTI, also per lead or model) replaced by
    the kWp-weighted mean over the groups."""
    base = dict(frames[0])
    if len(frames) == 1:
        return base
    ts = np.asarray(base["ts"], float)
    w = np.asarray(kwp, float) / float(np.sum(kwp))
    for col in [c for c in base if c.startswith(GTI)]:
        base[col] = sum(wi * _on(ts, f, col) for f, wi in zip(frames, w, strict=True))
    return base


def split_models(frame: Mapping[str, np.ndarray], models: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    """A multi-model response ('<var>_<model>' columns) -> {model: frame with base names}."""
    out = {}
    for m in models:
        sfx = f"_{m}"
        out[m] = {"ts": frame["ts"], **{k[: -len(sfx)]: v for k, v in frame.items() if k.endswith(sfx)}}
    return out


def archive_frames(system: System, responses: Callable[[str], list[tuple[str, dict]]]) -> dict:
    """{"hf": frame, "pr": {lead: {model: frame}}} from stored archive responses (store.responses)."""
    kwp = [p.kwp for p in system.planes]

    def stitched(prefix: str, rename=None) -> list[dict]:
        return [W.concat([W.parse(b, rename=rename) for _, b in responses(prefix.format(tag=plane_tag(i, p)))])
                for i, p in enumerate(system.planes)]
    hf = combine_planes(stitched("hf/{tag}/"), kwp)
    pr: dict[int, dict[str, dict]] = {0: {}, 1: {}}
    for model, variables in W.PREVIOUS_RUNS_MODELS.items():
        frame = combine_planes(stitched(f"pr/{model}/" + "{tag}/", W.lead_names(variables, (0, 1))), kwp)
        for lead, fr in W.split_leads(frame, (0, 1)).items():
            pr[lead][model] = fr
    return {"hf": hf, "pr": pr}


def live_frames(system: System, bodies: Mapping[str, Mapping]) -> dict[str, dict[str, np.ndarray]]:
    """{model: frame} (best_match + ensemble members) from live_requests' responses keyed like them."""
    kwp = [p.kwp for p in system.planes]
    tags = [plane_tag(i, p) for i, p in enumerate(system.planes)]
    if not all(f"live/best_match/{t}" in bodies for t in tags):
        return {}
    out = {"best_match": combine_planes([W.parse(bodies[f"live/best_match/{t}"]) for t in tags], kwp)}
    if all(f"live/ensemble/{t}" in bodies for t in tags):
        per_plane = [split_models(W.parse(bodies[f"live/ensemble/{t}"]), list(W.ENSEMBLE)) for t in tags]
        for m in W.ENSEMBLE:
            out[m] = combine_planes([pp[m] for pp in per_plane], kwp)
    return out


def t_air_from(*frames: Mapping[str, np.ndarray]) -> Callable | float:
    """Air temperature (C) at any time from forecast frames (20 C where none is known)."""
    ts, tv = [], []
    for f in frames:
        if f and "temperature_2m" in f:
            t = np.asarray(f["temperature_2m"], float)
            ok = ~np.isnan(t)
            ts.append(np.asarray(f["ts"], float)[ok])
            tv.append(t[ok])
    if not ts or not sum(len(x) for x in ts):
        return 20.0
    t_all, v_all = np.concatenate(ts), np.concatenate(tv)
    order = np.argsort(t_all, kind="stable")
    t_all, v_all = t_all[order], v_all[order]
    return lambda x: np.interp(np.asarray(x, float), t_all, v_all)


# ---------------------------------------------------------------- training
def _fc_columns(table: dict, wx: Mapping[str, np.ndarray]) -> None:
    fc = F.wx_for_periods(wx, table["ts"], HOUR) if len(wx.get("ts", ())) else {"ts": table["ts"]}
    fc["fc_kgti"] = F.forecast_kgti(fc, table["poa_cs"])
    n = len(table["ts"])
    for c in {*F.FC, "fc_gti"}:
        table[c] = fc.get(c, np.full(n, np.nan))


def label(system: System, table: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    stats = {k: np.asarray(v) for k, v in table.items() if k.split("_")[0] in ("pv", "soc", "bat", "grid", "gen",
                                                                             "curtail")}
    cfg = R.RegimeConfig(kind=system.kind)
    if system.kind == "unlimited":
        return R.label(stats, cfg, sun_el=table["sun_el"]), {}
    return R.auto_label(stats, cfg, segment=table["segment"], sun_el=table["sun_el"])


def hours_table(system: System, rows: Mapping[str, list], wx: Mapping[str, np.ndarray], t0: float, t1: float,
                t_air=None) -> dict[str, np.ndarray]:
    """Hourly periods [t0, t1): measured roles, sun/clear-sky, labels, nowcast forecast features."""
    tab = H.period_table(system.history_roles(), rows, t0, t1, int(HOUR))
    sky = F.sky_features(tab["ts"], HOUR, 12, system.site, system.planes,
                         t_air=t_air if t_air is not None else t_air_from(wx), change_points=system.change_points)
    tab.update({k: v for k, v in sky.items() if k != "ts"})
    lab, _ = label(system, tab)
    tab["regime"], tab["reason"] = lab["regime"], lab["reason"]
    _fc_columns(tab, wx)
    return tab


def _folds(ts: np.ndarray, n_splits: int) -> list[np.ndarray]:
    """Month-blocked folds; weeks while there are too few months; none with fewer than 2 groups."""
    months = M.month_ids(ts)
    groups = months if len(np.unique(months)) >= n_splits else np.floor(np.asarray(ts) / (7 * 86400))
    n = min(n_splits, len(np.unique(groups)))
    return M.group_folds(groups, n) if n >= 2 else []


def _factor(measured, physics, mask, min_n: int) -> float | None:
    m = mask & ~np.isnan(measured) & ~np.isnan(physics) & (physics > 0)
    if m.sum() < min_n or physics[m].sum() <= 0:
        return None
    return float(np.clip(measured[m].sum() / physics[m].sum(), 0.3, 1.3))


def _mae(pred, truth) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth))))


def train(system: System, rows: Mapping[str, list], frames: Mapping, now: float, previous: Mapping | None = None,
          log: Callable[[str], None] = lambda _: None) -> dict:
    """Train everything on all stored data. previous: the last result (its range table is reused while
    fresh). Returns a JSON-ready dict: models, physics factors, champions, range table, report."""
    t_start = time.process_time()
    cfg = system.model_config()
    hf, pr = frames["hf"], frames["pr"]
    starts = [r[0]["start"] for r in rows.values() if r]
    ends = [r[-1]["start"] for r in rows.values() if r]
    out: dict = {"version": 1, "trained_at": now, "system": system.to_dict(), "nowcast": None, "daily_d0": None,
                 "daily_d1": None, "factor": DEFAULT_FACTOR, "factor_daily": DEFAULT_FACTOR,
                 "champion": {"nowcast": "physics", "daily": "physics"}, "range": None, "range_at": None,
                 "report": {}}
    rep = out["report"]
    if not starts:
        rep["status"] = "no_data"
        return out
    t0, t1 = float(min(starts)), float(max(ends)) + HOUR
    t_air = t_air_from(hf)
    tab = hours_table(system, rows, hf, t0, t1, t_air)
    tab.update(M.targets(tab, cfg))
    ts, pv = tab["ts"], tab["pv_w"]
    trainable = tab["trainable"]
    exact = trainable & ~tab["cens"]
    phys = D.physics_power(tab["fc_gti"], tab["fc_temperature_2m"], system.planes, ts, t_air)
    recent = ts >= t1 - FACTOR_DAYS * 86400
    out["factor"] = _factor(pv, phys, exact & recent, MIN_FACTOR_HOURS) or DEFAULT_FACTOR
    rep |= {"hours": int(len(ts)), "hours_with_pv": int(np.sum(~np.isnan(pv))), "trainable_hours": int(trainable.sum()),
            "full_output_hours": int(exact.sum()), "throttled_hours": int((trainable & tab["cens"]).sum()),
            "first_hour": t0, "last_hour": t1 - HOUR,
            "days_of_data": round(float(np.sum(~np.isnan(pv))) / 24.0, 1)}
    log(f"hours table: {len(ts)} hours, {int(trainable.sum())} trainable")

    # nowcast: champion/challenger on the last HOLDOUT_DAYS, then the final fit on everything
    if trainable.sum() >= MIN_ML_HOURS:
        cut = t1 - HOLDOUT_DAYS * 86400
        before, hold = trainable & (ts < cut), exact & (ts >= cut)
        if before.sum() >= MIN_ML_HOURS and hold.sum() >= MIN_HOLDOUT_HOURS:
            trial = M.fit(tab, before, NOWCAST, cfg)
            pred = M.predict_k(trial, tab, hold) * tab["p_cs"][hold]
            f_before = _factor(pv, phys, exact & (ts < cut) & (ts >= cut - FACTOR_DAYS * 86400),
                               MIN_FACTOR_HOURS) or DEFAULT_FACTOR
            err_ml, err_phys = _mae(pred, pv[hold]), _mae(f_before * phys[hold], pv[hold])
            rep["nowcast_holdout"] = {"hours": int(hold.sum()), "ml_mae_w": round(err_ml, 1),
                                      "physics_mae_w": round(err_phys, 1)}
            if err_ml < err_phys:
                out["champion"]["nowcast"] = "ml"
        out["nowcast"] = M.fit(tab, trainable, NOWCAST, cfg).to_dict()
        log("nowcast model trained")

    # daily: reconstruct each past day's potential, then learn it from forecasts as issued
    tz = system.time_zone
    first_day, last_day = local_date(t0, tz) + timedelta(days=1), local_date(t1 - HOUR, tz) - timedelta(days=1)
    if last_day > first_day:
        g = D.hour_grid(first_day, last_day, tz, system.site, system.planes, t_air=t_air,
                        change_points=system.change_points)
        idx = np.clip(np.searchsorted(ts, g["ts"]), 0, len(ts) - 1)
        on = ts[idx] == g["ts"]
        g["pv_w"] = np.where(on, pv[idx], np.nan)
        g["regime"] = np.where(on, tab["regime"][idx], "").astype(object)
        g["reason"] = np.where(on, tab["reason"][idx], "").astype(object)
        g.update(M.targets(g, cfg))
        _fc_columns(g, hf)
        throttled = bool(np.any(g["regime"] == LIMITED))
        g = D.reconstruct(g, HINDSIGHT, _folds(g["ts"], 6) if throttled else [], cfg)
        dt = D.day_table(g)
        rep["days"] = int(len(dt["day"]))
        rep["complete_days"] = int(dt["complete"].sum())
        if pr[1].get("best_match") is not None and len(pr[1]["best_match"]["ts"]):
            daily = _train_daily(system, g, dt, pr, cfg, now, previous, t_air, rep)
            out["champion"]["daily"] = daily.pop("champion_daily", "physics")
            out |= daily
        log("daily models trained")
    rep["cpu_s"] = round(time.process_time() - t_start, 1)
    rep["status"] = "ready" if out["nowcast"] or out["daily_d1"] else "collecting"
    return out


def _train_daily(system: System, g: dict, dt: dict, pr: Mapping, cfg: M.ModelConfig, now: float,
                 previous: Mapping | None, t_air, rep: dict) -> dict:
    out: dict = {}
    day = g["day"]
    FF = {lead: D.forecast_features(g, pr[lead]) for lead in (1, 0)}
    X = {lead: {**g, **{k: v for k, v in FF[lead].items() if k != "ts"}} for lead in (1, 0)}
    have_fc = ~np.isnan(FF[1]["fc_gti_raw"])
    rows = g["trainable"] & have_fc & ~np.isnan(g["k_fill"])
    # physics per day from the day-ahead forecast; days count when complete and fully forecast
    phys_h = D.physics_power(FF[1]["fc_gti_raw"], FF[1]["fc_t_raw"], system.planes, g["ts"], t_air)
    days, phys_d = D.day_sum(np.where(have_fc, phys_h, np.nan), day)
    _, fc_hours = D.day_sum(have_fc.astype(float), day)
    usable = dt["complete"] & (fc_hours >= dt["hours"] - 2) & (phys_d > 0)
    recent = days >= days.max() - FACTOR_DAYS
    f_daily = _factor(dt["e_pot"], phys_d / 1000, usable & recent, 5)
    out["factor_daily"] = f_daily or DEFAULT_FACTOR
    if rows.sum() < MIN_ML_HOURS:
        return out
    # champion/challenger on the last HOLDOUT_DAYS days
    cut = days.max() - HOLDOUT_DAYS
    hold_days = usable & (days > cut)
    if hold_days.sum() >= MIN_HOLDOUT_DAYS and (rows & (day <= cut)).sum() >= MIN_ML_HOURS:
        trial = D.fit_daily(X[1], rows & (day <= cut), cfg)
        sel = day > cut
        pred_h = np.full(len(day), np.nan)
        pred_h[sel] = D.predict_hourly_w(trial, X[1], sel)
        _, pred_d = D.day_sum(pred_h, day, min_count=1)
        f_before = _factor(dt["e_pot"], phys_d / 1000, usable & (days <= cut) & (days > cut - FACTOR_DAYS), 5) \
            or DEFAULT_FACTOR
        truth = dt["e_pot"][hold_days]
        err_ml = _mae(pred_d[hold_days] / 1000, truth)
        err_phys = _mae(f_before * phys_d[hold_days] / 1000, truth)
        rep["daily_holdout"] = {"days": int(hold_days.sum()), "ml_mae_kwh": round(err_ml, 2),
                                "physics_mae_kwh": round(err_phys, 2), "mean_kwh": round(float(truth.mean()), 2)}
        out["champion_daily"] = "ml" if err_ml < err_phys else "physics"
    for lead in (1, 0):
        out[f"daily_d{lead}"] = D.fit_daily(X[lead], rows, cfg).to_dict()
    # P10/P90 of truth / prediction: out-of-fold over complete days, refreshed weekly
    prev_range = (previous or {}).get("range")
    if prev_range and now - float((previous or {}).get("range_at") or 0) < RANGE_MAX_AGE_S:
        out["range"], out["range_at"] = prev_range, previous["range_at"]
    elif usable.sum() >= RANGE_MIN_DAYS:
        pred_h = np.full(len(day), np.nan)
        for test in _folds(g["ts"], 3):
            m = D.fit_daily(X[1], rows & ~test, cfg)
            pred_h[test] = D.predict_hourly_w(m, X[1], test)
        _, pred_d = D.day_sum(pred_h, day, min_count=1)
        _, first = np.unique(day, return_index=True)
        clear = FF[1]["day_kgti"][first]
        ok = usable & (pred_d > 0)
        table = D.range_table(dt["e_pot"][ok] / (pred_d[ok] / 1000), clear[ok])
        out["range"] = {str(k): v for k, v in table.items()} or None
        out["range_at"] = now
        rep["range_days"] = int(ok.sum())
    return out


# ---------------------------------------------------------------- live outputs
def load_models(result: Mapping | None) -> dict:
    """Model objects from a train() result (None where absent)."""
    result = result or {}
    return {k: (M.load_model(result[k]) if result.get(k) else None) for k in ("nowcast", "daily_d0", "daily_d1")}


def range_table(result: Mapping | None) -> dict[int, list[float]] | None:
    r = (result or {}).get("range")
    return {int(k): v for k, v in r.items()} if r else None


def today_known(system: System, rows5: Mapping[str, list], hour_starts: np.ndarray) -> dict[str, np.ndarray]:
    """What the measurements already say about today's hours (5-minute statistics): seconds covered,
    energy used and potential energy (= used while the system is never throttled)."""
    n = len(hour_starts)
    zero = {"covered_s": np.zeros(n), "used_wh": np.zeros(n), "potential_wh": np.zeros(n)}
    if n == 0:
        return zero
    roles = {"pv_w": system.history_roles()["pv_w"]}
    t0, t1 = float(hour_starts.min()), float(hour_starts.max()) + HOUR
    tab = H.period_table(roles, rows5, t0, t1, H.FIVE_MIN)
    pv = tab["pv_w"]
    hour = np.floor(tab["ts"] / HOUR) * HOUR
    i = np.searchsorted(hour_starts, hour)
    ok = (i < n) & (hour_starts[np.minimum(i, n - 1)] == hour) & ~np.isnan(pv)
    covered = np.bincount(i[ok], minlength=n)[:n] * float(H.FIVE_MIN)
    used = np.bincount(i[ok], weights=np.clip(pv[ok], 0, None), minlength=n)[:n] * H.FIVE_MIN / HOUR
    return {"covered_s": covered, "used_wh": used, "potential_wh": used.copy()}


def outputs(system: System, result: Mapping | None, live: Mapping[str, Mapping], now: float,
            known_rows5: Mapping[str, list] | None = None, models: Mapping | None = None) -> dict:
    """Everything the sensors and the Energy dashboard show, from the newest forecast."""
    result = result or {}
    models = models if models is not None else load_models(result)
    tz = system.time_zone
    today = local_date(now, tz)
    out: dict = {"computed_at": now, "basis": {}, "hourly": {}}
    if "best_match" not in live:
        out["basis"]["forecast"] = "unavailable"
        return out
    t_air = t_air_from(live["best_match"])
    g = D.hour_grid(today, today + timedelta(days=2), tz, system.site, system.planes, t_air=t_air,
                    change_points=system.change_points)
    FF = D.forecast_features(g, live)
    X = {**g, **{k: v for k, v in FF.items() if k != "ts"}}
    day, hs = g["day"], g["ts"]
    d_today = D.day_number(today)
    have = ~np.isnan(FF["fc_gti_raw"])
    phys = result.get("factor_daily", DEFAULT_FACTOR) * D.physics_power(FF["fc_gti_raw"], FF["fc_t_raw"],
                                                                          system.planes, hs, t_air)
    pred = np.where(have, phys, np.nan)
    use_ml = result.get("champion", {}).get("daily") == "ml" and models.get("daily_d1") is not None
    if use_ml:
        for rows, name in ((day == d_today, "daily_d0"), (day > d_today, "daily_d1")):
            m = models.get(name) or models["daily_d1"]
            sel = rows & have
            if sel.any():
                pred[sel] = D.predict_hourly_w(m, X, sel)
    out["basis"]["daily"] = "learned" if use_ml else "physics"
    _, first = np.unique(day, return_index=True)
    clear_by_day = dict(zip(day[first].astype(int), FF["day_kgti"][first], strict=True))
    rng = range_table(result)
    m_today = day == d_today
    known = today_known(system, known_rows5, hs[m_today]) if known_rows5 else None
    clear_today = clear_by_day.get(d_today, np.nan)
    out["today"] = D.bank_today(hs[m_today], np.nan_to_num(pred[m_today]), now, clear_today, rng, known)
    out["today"]["clearness"] = None if np.isnan(clear_today) else round(float(clear_today), 3)
    for offset, name in ((1, "tomorrow"), (2, "day_after")):
        sel = day == d_today + offset
        if sel.any() and have[sel].mean() > 0.9:
            total = float(np.nansum(pred[sel]) / 1000)
            c = clear_by_day.get(d_today + offset, np.nan)
            lo, hi = D.apply_range([total], [c], rng)
            out[name] = {"kwh": round(total, 1), "p10_kwh": round(float(lo[0]), 1), "p90_kwh": round(float(hi[0]), 1),
                         "clearness": None if np.isnan(c) else round(float(c), 3)}
    out["hourly"] = {float(t): round(float(w), 1) for t, w in zip(hs, pred, strict=True) if not np.isnan(w)}
    out["now"] = potential_now(system, result, models, live["best_match"], now, t_air)
    return out


def potential_now(system: System, result: Mapping, models: Mapping, wx: Mapping[str, np.ndarray], now: float,
                  t_air, period_s: float = 900.0) -> dict:
    """Sky-model estimate of the panels' power for the current quarter hour (W)."""
    start = np.floor(now / period_s) * period_s
    tab = F.sky_features([start], period_s, 3, system.site, system.planes, t_air=t_air,
                         change_points=system.change_points)
    fc = F.wx_for_periods(wx, tab["ts"], period_s)
    fc["fc_kgti"] = F.forecast_kgti(fc, tab["poa_cs"])
    for c in {*F.FC, "fc_gti"}:
        tab[c] = fc.get(c, np.full(1, np.nan))
    p_cs = float(tab["p_cs"][0])
    gti = float(tab["fc_gti"][0])
    phys = result.get("factor", DEFAULT_FACTOR) * float(
        D.physics_power([gti], tab["fc_temperature_2m"], system.planes, tab["ts"], t_air)[0])
    if p_cs <= 1:
        return {"w": 0.0, "basis": "night", "clear_sky_w": round(p_cs, 1)}
    if result.get("champion", {}).get("nowcast") == "ml" and models.get("nowcast") is not None:
        k = float(M.predict_k(models["nowcast"], tab)[0])
        return {"w": round(k * p_cs, 1), "basis": "learned", "clear_sky_w": round(p_cs, 1)}
    return {"w": round(0.0 if np.isnan(phys) else phys, 1), "basis": "physics", "clear_sky_w": round(p_cs, 1)}


__all__ = ["System", "Request", "archive_requests", "archive_frames", "live_frames", "live_requests", "missing",
           "outputs", "train", "today_known", "load_models", "quarters"]
