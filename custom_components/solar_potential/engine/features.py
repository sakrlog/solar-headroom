"""Model inputs, numpy only: sun/season, satellite clear-sky indices, weather forecasts.

Tables are dicts of equal-length float arrays with a "ts" key (period starts, UTC epoch seconds).
Satellite and weather frames come from engine.weather (radiation indexed by interval start).

Feature groups
  SKY      sun position, clear-sky power, season, solar time, system segment (changes over time)
  SAT_NOW  satellite during the period itself (hindsight: daily target reconstruction)
  SAT_LAG  what the satellite showed in the last hour that was already published (real time)
  FC       weather forecast for the period (cloud layers, temperature, rain, snow, ...)

Every clear-sky index is relative to the panels' own plane (GTI / clear-sky POA), so learned
corrections transfer between seasons and systems sizes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from . import solar as S
from .weather import INTERVAL_VARS, SHORT

SKY = ["sun_el", "sun_el_max", "sun_az", "poa_cs", "p_cs", "doy_sin", "doy_cos", "hour", "segment"]
SAT_NOW = ["sat_kgti", "sat_kdni", "sat_kgti_min", "sat_kgti_max", "sat_kgti_std"]
SAT_LAG = ["sat_kgti_lag", "sat_kgti_lag_last", "sat_kgti_lag_trend", "sat_kdni_lag"]
FC = ["fc_cloud_cover", "fc_cloud_cover_low", "fc_cloud_cover_mid", "fc_cloud_cover_high", "fc_temperature_2m",
      "fc_relative_humidity_2m", "fc_wind_speed_10m", "fc_precipitation", "fc_sunshine_duration", "fc_kgti",
      "fc_snowfall", "fc_snow_depth"]
SAT_MIN_POA = 50.0          # W/m2: below this a clear-sky index is meaningless (dawn/dusk)
SAT_LATENCY_S = 1200        # live satellite intervals appear ~20 min after they end
MAX_INTERP_GAP_S = 7200     # never interpolate forecasts across gaps longer than this

Table = dict[str, np.ndarray]


# ---------------------------------------------------------------- small helpers
def lookup(src_ts, values, dst_ts) -> np.ndarray:
    """values (aligned with sorted unique src_ts) at dst_ts by exact time match; NaN elsewhere."""
    src_ts, dst_ts = np.asarray(src_ts, float), np.asarray(dst_ts, float)
    out = np.full(len(dst_ts), np.nan)
    if len(src_ts) == 0:
        return out
    i = np.clip(np.searchsorted(src_ts, dst_ts), 0, len(src_ts) - 1)
    ok = src_ts[i] == dst_ts
    out[ok] = np.asarray(values, float)[i[ok]]
    return out


def _div(a, b) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(a, float) / np.asarray(b, float)


def take(table: Mapping[str, np.ndarray], rows) -> Table:
    return {k: np.asarray(v)[rows] for k, v in table.items()}


def matrix(table: Mapping[str, np.ndarray], features: Sequence[str]) -> np.ndarray:
    """Feature matrix in the given order (missing columns are all-NaN)."""
    n = len(table["ts"])
    return np.column_stack([np.asarray(table[f], float) if f in table else np.full(n, np.nan) for f in features])


def ffill(values) -> np.ndarray:
    """Forward-fill NaN (leading NaN stay)."""
    v = np.asarray(values, float)
    idx = np.where(~np.isnan(v), np.arange(len(v)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = v[idx]
    out[np.cumsum(~np.isnan(v)) == 0] = np.nan
    return out


def shift(values, n: int = 1) -> np.ndarray:
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    if n > 0:
        out[n:] = v[:-n]
    elif n < 0:
        out[:n] = v[-n:]
    else:
        out[:] = v
    return out


# ---------------------------------------------------------------- sun and season
def sky_features(starts, period_s: float, n_sub: int, site: S.Site, planes: Sequence[S.Plane], t_air=None,
                 change_points: Sequence[float] = ()) -> Table:
    """Sun, clear-sky and calendar features for periods [start, start + period).

    hour: apparent solar time at the period middle (no time zones or daylight saving needed).
    segment: how many system changes (change_points, epoch s) happened before the period; lets one
    model use data from before a change (new panels, rewiring) while predicting for the current setup.
    """
    starts = np.atleast_1d(np.asarray(starts, float))
    sol = S.period_solar(starts, period_s, n_sub, site, planes, t_air)
    mid = starts + period_s / 2
    doy, _ = S.day_of_year(mid)
    pos = S.sun_position(mid, site.latitude, site.longitude, site.altitude)
    out = {"ts": starts, **{k: sol[k] for k in ("sun_el", "sun_el_max", "sun_az", "poa_cs", "p_cs", "p_cs_max",
                                                 "ghi_cs", "dni_cs")}}
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["hour"] = pos["solar_time"]
    out["segment"] = np.searchsorted(np.sort(np.asarray(change_points, float)), mid, side="right").astype(float)
    return out


# ---------------------------------------------------------------- satellite
def sat_with_clearsky(sat: Mapping[str, np.ndarray], step_s: float, site: S.Site,
                      planes: Sequence[S.Plane]) -> Table:
    """Clear-sky references at every native satellite interval (computed at its middle) and the
    clear-sky indices k = GTI / clear-sky POA and kd = DNI / clear-sky DNI."""
    out = {k: np.asarray(v, float) for k, v in sat.items()}
    s = S.solar(out["ts"] + step_s / 2, site)
    kwp = sum(p.kwp for p in planes)
    out["poa_cs"] = sum(S.poa_clearsky(s, p) * p.kwp for p in planes) / kwp
    out["ghi_cs"], out["dni_cs"] = s["ghi_cs"], s["dni_cs"]
    ok = out["poa_cs"] >= SAT_MIN_POA
    out["k"] = np.where(ok, _div(out["gti"], out["poa_cs"]), np.nan)
    out["kd"] = np.where(ok, _div(out["dni"], np.where(out["dni_cs"] > 20, out["dni_cs"], np.nan)), np.nan)
    return out


def fit_quantile_map(src, ref, n: int = 99) -> dict:
    """Monotone map giving `src` values the distribution of `ref` (paired periods both sources cover)."""
    q = np.linspace(0.005, 0.995, n)
    return {"src": np.quantile(np.asarray(src, float), q).round(4).tolist(),
            "ref": np.quantile(np.asarray(ref, float), q).round(4).tolist()}


def apply_quantile_map(x, qm: Mapping) -> np.ndarray:
    xs, ys = np.asarray(qm["src"], float), np.asarray(qm["ref"], float)
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xs, ys)
    y = np.where(x > xs[-1], x * ys[-1] / xs[-1], y)               # beyond the fitted range: keep the end ratio
    y = np.where(x < xs[0], x * (ys[0] / xs[0] if xs[0] > 0 else 1.0), y)
    return np.where(np.isnan(x), np.nan, y)


def fit_source_maps(ref: Mapping[str, np.ndarray], ref_step_s: float, src: Mapping[str, np.ndarray],
                    src_step_s: float, min_poa: float = 150.0) -> dict:
    """Quantile maps src -> ref for k and kd, on ref-length slots both sources cover completely
    (src intervals summed per slot; every interval of the slot must be there)."""
    keep = np.asarray(src["poa_cs"]) >= min_poa
    slot = np.floor(np.asarray(src["ts"]) / ref_step_s) * ref_step_s
    per = max(1, int(round(ref_step_s / src_step_s)))
    agg = {}
    for c in ("gti", "poa_cs", "dni", "dni_cs"):
        v = np.where(keep, np.asarray(src[c], float), np.nan)
        agg[c] = _group(slot, v, "sum", min_count=per)
    u = agg["gti"][0]
    d_k = _div(agg["gti"][1], agg["poa_cs"][1])
    d_kd = _div(agg["dni"][1], np.where(agg["dni_cs"][1] > 20, agg["dni_cs"][1], np.nan))
    rk = np.where(np.asarray(ref["poa_cs"]) >= min_poa, np.asarray(ref["k"], float), np.nan)
    rkd = np.where(np.asarray(ref["poa_cs"]) >= min_poa, np.asarray(ref["kd"], float), np.nan)
    s_k, s_kd = lookup(ref["ts"], rk, u), lookup(ref["ts"], rkd, u)
    both = ~np.isnan(s_k) & ~np.isnan(s_kd) & ~np.isnan(d_k) & ~np.isnan(d_kd)
    if both.sum() < 200:
        return {"n_pairs": int(both.sum())}
    return {"k": fit_quantile_map(d_k[both], s_k[both]), "kd": fit_quantile_map(d_kd[both], s_kd[both]),
            "n_pairs": int(both.sum()), "period": [float(u[both].min()), float(u[both].max())]}


def _group(keys, values, how: str, min_count: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(unique sorted keys, aggregate) skipping NaN; groups with fewer than min_count values -> NaN."""
    keys, values = np.asarray(keys, float), np.asarray(values, float)
    order = np.argsort(keys, kind="stable")
    k, v = keys[order], values[order]
    uk, idx = np.unique(k, return_index=True)
    if len(uk) == 0:
        return uk, np.zeros(0)
    valid = ~np.isnan(v)
    cnt = np.add.reduceat(valid.astype(np.int64), idx)
    if how == "sum":
        agg = np.add.reduceat(np.where(valid, v, 0.0), idx)
    elif how == "mean":
        agg = _div(np.add.reduceat(np.where(valid, v, 0.0), idx), cnt)
    elif how == "min":
        agg = np.fmin.reduceat(v, idx)
    elif how == "max":
        agg = np.fmax.reduceat(v, idx)
    elif how == "std":                                               # population std (ddof=0)
        s1 = np.add.reduceat(np.where(valid, v, 0.0), idx)
        s2 = np.add.reduceat(np.where(valid, v * v, 0.0), idx)
        m = _div(s1, cnt)
        agg = np.sqrt(np.maximum(_div(s2, cnt) - m * m, 0.0))
    elif how == "count":
        agg = cnt.astype(float)
    else:
        raise ValueError(how)
    return uk, np.where(cnt >= min_count, agg, np.nan)


def harmonise(sat: Mapping[str, np.ndarray], maps: Mapping) -> Table:
    """Express a satellite source on a reference source's scale (via its k and kd maps)."""
    out = dict(sat)
    if "k" not in maps:
        return out
    out["k"] = apply_quantile_map(sat["k"], maps["k"])
    out["kd"] = apply_quantile_map(sat["kd"], maps["kd"])
    out["gti"] = np.where(~np.isnan(out["k"]), out["k"] * sat["poa_cs"], sat["gti"])
    out["dni"] = np.where(~np.isnan(out["kd"]), out["kd"] * sat["dni_cs"], sat["dni"])
    return out


def sat_concurrent(sat: Mapping[str, np.ndarray], step_s: float, starts, period_s: float) -> Table:
    """Clear-sky indices over each period itself (known only afterwards: daily target reconstruction).
    Periods shorter than the native step use the interval that contains them."""
    starts = np.asarray(starts, float)
    keep = np.asarray(sat["poa_cs"], float) >= SAT_MIN_POA
    ts = np.asarray(sat["ts"], float)[keep]
    key = np.floor(ts / period_s) * period_s if period_s > step_s else ts
    col = {c: np.asarray(sat[c], float)[keep] for c in ("gti", "poa_cs", "dni", "dni_cs", "k")}
    # ratios of sums over the intervals where numerator and denominator both exist
    g_ok = ~np.isnan(col["gti"])
    d_ok = ~np.isnan(col["dni"]) & ~np.isnan(col["dni_cs"])
    u, s_gti = _group(key, np.where(g_ok, col["gti"], np.nan), "sum")
    _, s_poa = _group(key, np.where(g_ok, col["poa_cs"], np.nan), "sum")
    _, s_dni = _group(key, np.where(d_ok, col["dni"], np.nan), "sum")
    _, s_dcs = _group(key, np.where(d_ok, col["dni_cs"], np.nan), "sum")
    _, n = _group(key, col["gti"], "count", min_count=0)
    agg = {"sat_kgti": _div(s_gti, np.where(s_poa > 0, s_poa, np.nan)),
           "sat_kdni": _div(s_dni, np.where(s_dcs > 0, s_dcs, np.nan)),
           "sat_kgti_min": _group(key, col["k"], "min")[1], "sat_kgti_max": _group(key, col["k"], "max")[1],
           "sat_kgti_std": _group(key, col["k"], "std")[1]}
    if period_s > step_s:
        ok = n >= max(1, int(period_s // step_s) // 2)
        at = starts
    else:
        ok = n >= 1
        at = np.floor(starts / step_s) * step_s
    return {"ts": starts, **{k: lookup(u[ok], v[ok], at) for k, v in agg.items()}}


def _rolling_sum(a: np.ndarray, n: int) -> np.ndarray:
    """Sum of the last n values (inclusive), skipping NaN; NaN when the window has no value."""
    v = np.concatenate([[0.0], np.cumsum(np.nan_to_num(a, nan=0.0))])
    c = np.concatenate([[0], np.cumsum(~np.isnan(a))])
    i = np.arange(len(a))
    lo = np.maximum(i - n + 1, 0)
    return np.where(c[i + 1] - c[lo] > 0, v[i + 1] - v[lo], np.nan)


def sat_lagged(sat: Mapping[str, np.ndarray], step_s: float, times, window_s: float = 3600,
               latency_s: float = SAT_LATENCY_S) -> Table:
    """What the satellite showed in the latest `window_s` that was AVAILABLE at `times` (intervals that
    ended at least `latency_s` earlier): clear-sky index over the window, of the newest interval, its
    trend, the beam index, and how old the newest interval is (minutes)."""
    times = np.asarray(times, float)
    ts = np.asarray(sat["ts"], float)
    if len(ts) == 0:
        nan = np.full(len(times), np.nan)
        return {"ts": times, **{k: nan.copy() for k in (*SAT_LAG, "sat_lag_min")}}
    g0 = ts[0]
    size = int(round((ts[-1] - g0) / step_s)) + 1
    pos = np.round((ts - g0) / step_s).astype(np.int64)
    on = np.abs(g0 + pos * step_s - ts) < 1e-6
    keep = on & (np.asarray(sat["poa_cs"], float) >= SAT_MIN_POA)
    G = {}
    for c in ("gti", "poa_cs", "dni", "dni_cs"):
        a = np.full(size, np.nan)
        a[pos[keep]] = np.asarray(sat[c], float)[keep]
        G[c] = a
    for num, den in (("gti", "poa_cs"), ("dni", "dni_cs")):          # window sums over complete pairs only
        gap = np.isnan(G[num]) | np.isnan(G[den])
        G[num][gap], G[den][gap] = np.nan, np.nan
    n = max(1, int(window_s // step_s))
    r = {c: _rolling_sum(G[c], n) for c in G}
    k_win = _div(r["gti"], np.where(r["poa_cs"] > 0, r["poa_cs"], np.nan))
    kd_win = _div(r["dni"], np.where(r["dni_cs"] > 20, r["dni_cs"], np.nan))
    k_one = _div(G["gti"], G["poa_cs"])
    last_end = np.floor((times - latency_s) / step_s) * step_s
    first = last_end - step_s
    prev = first - step_s * max(1, n // 2)

    def at(arr, t):
        i = np.round((t - g0) / step_s).astype(np.int64)
        ok = (i >= 0) & (i < size) & (np.abs(g0 + i * step_s - t) < 1e-6)
        out = np.full(len(t), np.nan)
        out[ok] = arr[i[ok]]
        return out
    last = at(k_one, first)
    return {"ts": times, "sat_kgti_lag": at(k_win, first), "sat_kgti_lag_last": last,
            "sat_kgti_lag_trend": last - at(k_one, prev), "sat_kdni_lag": at(kd_win, first),
            "sat_lag_min": (times - last_end) / 60.0}


def satellite_features(sources: Sequence[tuple[Mapping[str, np.ndarray], float]], starts, period_s: float) -> Table:
    """SAT_NOW + SAT_LAG for periods from several satellite sources (already on one scale, see
    harmonise), in priority order: each value comes from the first source that has it."""
    starts = np.asarray(starts, float)
    mid = starts + period_s / 2
    out: Table = {"ts": starts}
    for sat, step in sources:
        now = sat_concurrent(sat, step, starts, period_s)
        lag = sat_lagged(sat, step, mid)
        for k in (*SAT_NOW, *SAT_LAG, "sat_lag_min"):
            v = now[k] if k in now else lag[k]
            out[k] = v if k not in out else np.where(np.isnan(out[k]), v, out[k])
    return out


# ---------------------------------------------------------------- weather forecasts
def wx_for_periods(wx: Mapping[str, np.ndarray], starts, period_s: float) -> Table:
    """Forecast for periods [start, start + period): interval variables (radiation, rain, sunshine,
    snowfall: preceding-hour values) from the hour that contains the period start, instantaneous ones
    interpolated to the period middle (never across gaps > 2 h). Columns fc_<name>, radiation
    shortened (fc_ghi, fc_dni, fc_dhi, fc_gti)."""
    starts = np.asarray(starts, float)
    mid = starts + period_s / 2
    t_wx = np.asarray(wx["ts"], float)
    hour_start = np.floor(starts / 3600.0) * 3600.0
    out: Table = {"ts": starts}
    for c, v in wx.items():
        if c == "ts":
            continue
        v = np.asarray(v, float)
        name = f"fc_{SHORT.get(c, c)}"
        if c in INTERVAL_VARS:
            out[name] = lookup(t_wx - 3600.0, v, hour_start)
            continue
        ok = ~np.isnan(v)
        tv, vv = t_wx[ok], v[ok]
        y = np.full(len(mid), np.nan)
        if len(tv) >= 2:
            y = np.interp(mid, tv, vv, left=np.nan, right=np.nan)
            i = np.clip(np.searchsorted(tv, mid), 1, len(tv) - 1)
            y[(tv[i] - tv[i - 1]) > MAX_INTERP_GAP_S] = np.nan
        out[name] = y
    return out


def forecast_kgti(fc: Mapping[str, np.ndarray], poa_cs) -> np.ndarray:
    """Forecast clear-sky index on the panel plane (NaN at dawn/dusk)."""
    poa_cs = np.asarray(poa_cs, float)
    gti = np.asarray(fc["fc_gti"], float) if "fc_gti" in fc else np.full(len(poa_cs), np.nan)
    return _div(gti, np.where(poa_cs >= SAT_MIN_POA, poa_cs, np.nan))


# ---------------------------------------------------------------- live history
def history_features(ts, k_obs, is_mpp, max_gap_s: float = 5400) -> Table:
    """For each row: k at the most recent EARLIER full-output row and minutes since then (NaN when
    older than max_gap_s)."""
    ts = np.asarray(ts, float)
    is_mpp = np.asarray(is_mpp, bool)
    last_k = ffill(shift(np.where(is_mpp, k_obs, np.nan)))
    last_t = ffill(shift(np.where(is_mpp, ts, np.nan)))
    mins = (ts - last_t) / 60.0
    stale = mins > max_gap_s / 60.0
    return {"ts": ts, "k_last_mpp": np.where(stale, np.nan, last_k), "min_since_mpp": np.where(stale, np.nan, mins)}
