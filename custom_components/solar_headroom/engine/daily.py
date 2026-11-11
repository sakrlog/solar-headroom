"""Daily solar bank: how much energy (kWh) the panels could deliver over a local day.

Target   the potential energy of a day = sum over its daylight hours of the potential power:
         measured at full output (exact); throttled hours: max(measured floor, hindsight estimate
         from the satellite of that hour); missing hours: hindsight estimate.
Model    hourly k from forecasts as issued before the day (lead 1 = the day before, lead 0 = newest
         run), summed over the day. Inputs: sun/season, forecast irradiance on the panel plane, cloud,
         rain, snow, ..., the spread between weather models, and whole-day summaries.
Range    P10..P90 of truth / prediction on held-out days, per bin of forecast whole-day clearness.
Today    what the live log already knows + the forecast for what is still to come.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from . import solar as S
from .features import SAT_MIN_POA, SKY, Table, forecast_kgti, sky_features, wx_for_periods
from .models import K_MAX, ModelConfig, fit, predict_k
from .regimes import LIMITED, MPP, NIGHT
from .weather import ENSEMBLE

F_HOUR = ["fc_kgti", "fc_kghi", "fc_kdni", "fc_diffuse_frac", "fc_cloud_cover", "fc_temperature_2m",
          "fc_relative_humidity_2m", "fc_wind_speed_10m", "fc_precipitation", "fc_sunshine_frac", "fc_cape",
          "fc_snowfall", "fc_snow_depth", "fc_kgti_prev", "fc_kgti_next", "gfs_kgti", "icon_kgti", "mf_kgti",
          "ens_kgti_mean", "ens_kgti_std", "ens_cloud_mean", "ens_cloud_std"]
F_DAY = ["day_kgti", "day_kgti_am", "day_kgti_pm", "day_cloud", "day_precip", "day_sunshine_frac", "day_ens_std"]
DAILY = SKY + F_HOUR + F_DAY
RANGE_BINS = (0.0, 0.55, 0.75, 0.88, 9.0)     # forecast whole-day clearness
MIN_GRID_P_CS = 5.0                            # W: hours with less clear-sky power are not daylight


# ---------------------------------------------------------------- local days
def local_day(ts, tz: str) -> np.ndarray:
    """Local calendar day of epoch seconds as days since 1970-01-01 (int)."""
    ts = np.asarray(ts, float)
    zone = ZoneInfo(tz)
    hours = np.floor(ts / 3600.0)
    uh, inv = np.unique(hours, return_inverse=True)
    off = np.array([datetime.fromtimestamp(h * 3600.0, zone).utcoffset().total_seconds() for h in uh])
    return np.floor((ts + off[inv]) / 86400.0).astype(np.int64)


def day_number(d: date | str) -> int:
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return (d - date(1970, 1, 1)).days


def day_date(n: int) -> date:
    return date(1970, 1, 1) + timedelta(days=int(n))


def local_midnight(n: int, tz: str) -> float:
    d = day_date(n)
    return datetime(d.year, d.month, d.day, tzinfo=ZoneInfo(tz)).timestamp()


def hour_grid(first_day: date | str, last_day: date | str, tz: str, site: S.Site, planes: Sequence[S.Plane],
              t_air=None, change_points: Sequence[float] = ()) -> Table:
    """Daylight hours (UTC hour starts, clear-sky power > 5 W) of local days first..last with sun and
    clear-sky features and the local day number ("day")."""
    a, b = day_number(first_day), day_number(last_day)
    t0, t1 = local_midnight(a, tz), local_midnight(b + 1, tz)
    starts = np.arange(np.floor(t0 / 3600) * 3600, t1, 3600.0)
    g = sky_features(starts, 3600, 12, site, planes, t_air, change_points)
    g["day"] = local_day(starts, tz).astype(float)
    keep = (g["p_cs"] > MIN_GRID_P_CS) & (g["day"] >= a) & (g["day"] <= b)
    return {k: v[keep] for k, v in g.items()}


# ---------------------------------------------------------------- grouping helpers
def _day_groups(day) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    day = np.asarray(day)
    u, idx, inv = np.unique(day, return_index=True, return_inverse=True)
    if np.any(np.diff(day) < 0):
        raise ValueError("rows must be sorted by time")
    return u, idx, inv


def day_sum(values, day, min_count: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(days, sums skipping NaN); days with fewer than min_count values -> NaN."""
    u, idx, _ = _day_groups(day)
    v = np.asarray(values, float)
    s = np.add.reduceat(np.nan_to_num(v, nan=0.0), idx) if len(v) else np.zeros(0)
    c = np.add.reduceat((~np.isnan(v)).astype(np.int64), idx) if len(v) else np.zeros(0)
    return u, np.where(c >= min_count, s, np.nan)


def _transform(values, day, how: str) -> np.ndarray:
    """Per-day aggregate broadcast back to the rows (pandas groupby().transform)."""
    u, idx, inv = _day_groups(day)
    v = np.asarray(values, float)
    valid = ~np.isnan(v)
    s = np.add.reduceat(np.where(valid, v, 0.0), idx)
    if how == "sum":
        return s[inv]
    c = np.add.reduceat(valid.astype(np.int64), idx)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / np.where(c > 0, c, np.nan))[inv]


def _shift_in_day(values, day, n: int) -> np.ndarray:
    v, d = np.asarray(values, float), np.asarray(day)
    out = np.full(len(v), np.nan)
    if n > 0:
        same = d[n:] == d[:-n]
        out[n:] = np.where(same, v[:-n], np.nan)
    else:
        same = d[:n] == d[-n:]
        out[:n] = np.where(same, v[-n:], np.nan)
    return out


def _div(a, b) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(a, float) / np.asarray(b, float)


# ---------------------------------------------------------------- target reconstruction
def reconstruct(g: Table, hind_features: Sequence[str], test_masks: Sequence[np.ndarray],
                cfg: ModelConfig) -> Table:
    """Hourly potential (W) for every grid hour: exact at full output, max(floor, out-of-fold hindsight
    estimate) while throttled, the estimate where data is missing. g needs pv_w, regime, p_cs and the
    hindsight features; k_obs / cens / w / trainable as from models.targets."""
    n = len(g["ts"])
    k = np.full(n, np.nan)
    for test in test_masks:
        rows = ~test & g["trainable"]
        if rows.sum() < 2 * cfg.params.min_leaf:
            continue
        m = fit(g, rows, hind_features, cfg)
        k[test] = predict_k(m, g, test)
    out = dict(g)
    out["k_hind"] = k
    est = k * g["p_cs"]
    pv, reg = np.asarray(g["pv_w"], float), np.asarray(g["regime"], dtype=object)
    mpp, lim = reg == MPP, reg == LIMITED
    dawn_dusk = (reg == NIGHT) & ~np.isnan(pv) & (g["p_cs"] < cfg.min_p_cs_w)
    pot = np.select([mpp, lim, dawn_dusk], [pv, np.fmax(pv, est), pv], est)
    out["pot"] = pot
    out["pot_exact"] = np.where(mpp | dawn_dusk, pot, 0.0)
    out["k_fill"] = _div(pot, g["p_cs"])
    out["missing"] = ~(mpp | lim | dawn_dusk) & (g["p_cs"] >= cfg.min_p_cs_w)
    return out


def day_table(g: Table) -> Table:
    """Per local day: clear-sky, potential, used and exactly measured energy (kWh), missing hours."""
    day = g["day"]
    u, e_cs = day_sum(g["p_cs"], day)
    _, e_pot = day_sum(g["pot"], day)
    _, e_pv = day_sum(g["pv_w"], day)
    _, e_exact = day_sum(g["pot_exact"], day)
    _, n_missing = day_sum(np.asarray(g["missing"], float), day)
    _, hours = day_sum(np.ones(len(day)), day)
    d = {"day": u.astype(float), "e_cs": e_cs / 1000, "e_pot": e_pot / 1000, "e_pv": e_pv / 1000,
         "e_exact": e_exact / 1000, "n_missing": n_missing, "hours": hours}
    d["exact_share"] = _div(d["e_exact"], d["e_pot"])
    d["complete"] = d["n_missing"] == 0
    return d


# ---------------------------------------------------------------- forecast features
def forecast_features(g: Mapping[str, np.ndarray], frames: Mapping[str, Mapping[str, np.ndarray]]) -> Table:
    """Hourly + whole-day forecast features for the hours of g. frames: model -> forecast frame at raw
    Open-Meteo times with base variable names ("best_match" required, ENSEMBLE models optional)."""
    ts = np.asarray(g["ts"], float)
    n = len(ts)
    day = np.asarray(g["day"])
    f = wx_for_periods(frames["best_match"], ts, 3600)

    def col(c: str) -> np.ndarray:
        return f.get(c, np.full(n, np.nan))
    poa_cs = np.asarray(g["poa_cs"], float)
    poa = np.where(poa_cs >= SAT_MIN_POA, poa_cs, np.nan)
    out: Table = {"ts": ts}
    out["fc_kgti"] = forecast_kgti(f, poa_cs)
    out["fc_kghi"] = _div(col("fc_ghi"), np.where(g["ghi_cs"] >= 50, g["ghi_cs"], np.nan))
    out["fc_kdni"] = _div(col("fc_dni"), np.where(g["dni_cs"] > 20, g["dni_cs"], np.nan))
    ghi = col("fc_ghi")
    out["fc_diffuse_frac"] = _div(col("fc_dhi"), np.where(ghi > 20, ghi, np.nan))
    for c in ("cloud_cover", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "cape",
              "snowfall", "snow_depth"):
        out[f"fc_{c}"] = col(f"fc_{c}")
    out["fc_sunshine_frac"] = col("fc_sunshine_duration") / 3600.0
    out["fc_kgti_prev"] = _shift_in_day(out["fc_kgti"], day, 1)
    out["fc_kgti_next"] = _shift_in_day(out["fc_kgti"], day, -1)
    ks, cs = [out["fc_kgti"]], [out["fc_cloud_cover"]]
    for model, short in ENSEMBLE.items():
        if model in frames:
            e = wx_for_periods(frames[model], ts, 3600)
            out[f"{short}_kgti"] = _div(e.get("fc_gti", np.full(n, np.nan)), poa)
            cs.append(e.get("fc_cloud_cover", np.full(n, np.nan)))
        else:
            out[f"{short}_kgti"] = np.full(n, np.nan)
            cs.append(np.full(n, np.nan))
        ks.append(out[f"{short}_kgti"])
    K, C = np.vstack(ks), np.vstack(cs)
    out["ens_kgti_mean"], out["ens_kgti_std"] = _nanmean(K), _nanstd1(K)
    out["ens_cloud_mean"], out["ens_cloud_std"] = _nanmean(C), _nanstd1(C)
    # whole-day context: clearness as forecast for the day, morning vs afternoon (solar time), rain, ...
    gti = col("fc_gti")
    ok = ~np.isnan(gti)
    am = np.asarray(g["hour"], float) < 12

    def ratio(mask):
        num = _transform(np.where(ok & mask, gti, np.nan), day, "sum")
        den = _transform(np.where(ok & mask, poa_cs, np.nan), day, "sum")
        return _div(num, np.where(den > 0, den, np.nan))
    out["day_kgti"] = ratio(np.ones(n, bool))
    out["day_kgti_am"], out["day_kgti_pm"] = ratio(am), ratio(~am)
    lit = ~np.isnan(poa)
    out["day_cloud"] = _transform(np.where(lit, out["fc_cloud_cover"], np.nan), day, "mean")
    out["day_precip"] = _transform(out["fc_precipitation"], day, "sum")
    out["day_sunshine_frac"] = _transform(np.where(lit, out["fc_sunshine_frac"], np.nan), day, "mean")
    out["day_ens_std"] = _transform(np.where(lit, out["ens_kgti_std"], np.nan), day, "mean")
    out["fc_gti_raw"], out["fc_t_raw"] = gti, col("fc_temperature_2m")
    return out


def _nanmean(M: np.ndarray) -> np.ndarray:
    c = np.sum(~np.isnan(M), axis=0)
    return _div(np.nansum(M, axis=0), np.where(c > 0, c, np.nan))


def _nanstd1(M: np.ndarray) -> np.ndarray:
    """Sample std (ddof=1) over rows skipping NaN; NaN with fewer than 2 values."""
    c = np.sum(~np.isnan(M), axis=0)
    m = _nanmean(M)
    ss = np.nansum((M - m) ** 2, axis=0)
    return np.where(c >= 2, np.sqrt(_div(ss, np.where(c >= 2, c - 1, np.nan))), np.nan)


def physics_power(fc_gti, fc_t, planes: Sequence[S.Plane], ts=None,
                  t_air: Callable | float | None = None) -> np.ndarray:
    """Forecast plane irradiance x system size, temperature-corrected (what typical PV-forecast
    services do; the cold-start baseline). Missing temperatures fall back to t_air (default 20 C)."""
    gti = np.nan_to_num(np.asarray(fc_gti, float), nan=0.0)
    t = np.asarray(fc_t, float)
    if callable(t_air):
        fallback = t_air(ts)
    else:
        fallback = np.full(len(t), 20.0 if t_air is None else float(t_air))
    t = np.where(np.isnan(t), fallback, t)
    kwp = sum(p.kwp for p in planes)
    return kwp * gti * S.temp_derate(gti, t)


# ---------------------------------------------------------------- model, range, bank
def fit_daily(g: Table, rows, cfg: ModelConfig, features: Sequence[str] = DAILY):
    """Daily model: hourly k_fill (reconstructed potential / clear sky) as exact targets."""
    return fit(g, rows, features, cfg, target="k_fill", censored=False)


def predict_hourly_w(model, table: Table, rows=None) -> np.ndarray:
    p_cs = np.asarray(table["p_cs"], float) if rows is None else np.asarray(table["p_cs"], float)[rows]
    return predict_k(model, table, rows) * p_cs


def range_bin(clear) -> np.ndarray:
    """Bin index of forecast whole-day clearness (right-closed bins), -1 when outside / NaN."""
    x = np.asarray(clear, float)
    i = np.searchsorted(np.asarray(RANGE_BINS), x, side="left") - 1
    return np.where((i >= 0) & (i < len(RANGE_BINS) - 1) & ~np.isnan(x), i, -1)


def range_table(ratio, clear, min_days: int = 15) -> dict[int, list[float]]:
    """P10 / P50 / P90 of truth / prediction per clearness bin (+ number of days)."""
    ratio, b = np.asarray(ratio, float), range_bin(clear)
    out = {}
    for i in range(len(RANGE_BINS) - 1):
        r = ratio[(b == i) & ~np.isnan(ratio)]
        if len(r) >= min_days:
            out[i] = [round(float(q), 3) for q in np.quantile(r, [0.1, 0.5, 0.9])] + [int(len(r))]
    return out


# until a site has ~60 days of its own: truth / prediction quantiles measured on a reference site
DEFAULT_RANGE = {0: [0.54, 0.93, 1.42, 0], 1: [0.64, 0.95, 1.49, 0], 2: [0.72, 0.99, 1.29, 0],
                 3: [0.82, 1.02, 1.17, 0]}


def apply_range(pred, clear, table: Mapping[int, Sequence[float]] | None) -> tuple[np.ndarray, np.ndarray]:
    """(p10, p90) for predictions; bins without enough history use the lowest bin that has one
    (unknown clearness counts as the cloudiest bin)."""
    table = table or DEFAULT_RANGE
    pred = np.asarray(pred, float)
    b = np.maximum(range_bin(clear), 0)
    first = table[min(table)]
    lo = np.array([table.get(int(i), first)[0] for i in b])
    hi = np.array([table.get(int(i), first)[2] for i in b])
    return pred * lo, pred * hi


def bank_today(hour_starts, pred_w, now: float, clear: float, range_tab=None,
               known: Mapping[str, np.ndarray] | None = None) -> dict:
    """Today's bank from the hourly forecast (W, hourly means) and what the live log already knows.
    known: per hour of hour_starts, "covered_s" (seconds the log covers), "potential_wh", "used_wh".
    Hours (or parts) that are past but not covered by the log count with the forecast."""
    hs, p = np.asarray(hour_starts, float), np.asarray(pred_w, float)
    ahead = np.clip((hs + 3600 - now) / 3600, 0, 1)
    total, left = float(p.sum() / 1000), float((p * ahead).sum() / 1000)

    def rng(x):
        lo, hi = apply_range([x], [clear], range_tab)
        return [round(float(lo[0]), 1), round(float(hi[0]), 1)]
    out = {"forecast_kwh": round(total, 1), "forecast_range_kwh": rng(total), "left_kwh": round(left, 1),
           "left_range_kwh": rng(left)}
    if len(hs) == 0:
        return out
    t0, t1 = hs.min(), min(now, hs.max() + 3600)
    st = hs - t0
    past_s = np.clip(np.minimum(st + 3600, t1 - t0) - st, 0, 3600)
    zero = np.zeros(len(hs))
    k = {c: np.asarray(known[c], float) if known else zero for c in ("covered_s", "potential_wh", "used_wh")}
    covered = np.minimum(k["covered_s"], past_s)
    so_far = float(k["potential_wh"].sum() / 1000)
    unknown = float((p * (past_s - covered) / 3600).sum() / 1000) + left
    r = rng(unknown)
    used = float(k["used_wh"].sum() / 1000)
    out |= {"day_estimate_kwh": round(so_far + unknown, 1),
            "day_range_kwh": [round(so_far + r[0], 1), round(so_far + r[1], 1)],
            "so_far_potential_kwh": round(so_far, 1), "so_far_used_kwh": round(used, 1),
            "so_far_wasted_kwh": round(max(0.0, so_far - used), 1),
            "log_coverage_pct": round(100 * covered.sum() / past_s.sum()) if past_s.sum() else None}
    return out


__all__ = ["DAILY", "F_DAY", "F_HOUR", "K_MAX", "RANGE_BINS", "apply_range", "bank_today", "day_table",
           "fit_daily", "forecast_features", "hour_grid", "local_day", "physics_power", "range_table",
           "reconstruct"]
