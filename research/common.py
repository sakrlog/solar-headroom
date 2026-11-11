"""Shared pieces of the research scripts (not shipped with the integration).

Public data only, downloaded on first use and cached under research/data/ (git-ignored):
* NREL PVDAQ systems from the Open Energy Data Initiative data lake (s3://oedi-data-lake/pvdaq/)
* Open-Meteo historical forecasts (CC BY 4.0), requested through engine.weather like the integration

Run the scripts with the repo's dev venv from the repo root, e.g.
    .venv/bin/python -I research/learning_curves.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "custom_components" / "solar_headroom"))

from engine import features as F  # noqa: E402
from engine import solar as S  # noqa: E402
from engine import weather as W  # noqa: E402

DATA = ROOT / "research" / "data"
RESULTS = ROOT / "research" / "results"
OEDI = "https://oedi-data-lake.s3.amazonaws.com/"
USER_AGENT = "solar-headroom-research/0.1"      # generic client name; no contact details
HOUR = 3600


@dataclass(frozen=True)
class SiteSpec:
    key: str
    system_id: int
    location: str
    lat: float
    lon: float
    alt: float
    tz: str                   # for local days only; all data is handled in UTC
    tilt: float
    azimuth: float            # deg clockwise from north
    kwp: float                # nameplate DC size from the PVDAQ metadata: what a user would enter
    climate: str              # Koppen
    source: str               # "nrel_parquet" (per-day files) or "prize_meter" (15-min meter CSVs)
    metrics: tuple[int, ...] = ()
    # the logger's clock runs on fixed standard time at this UTC offset (h) and the dataset's own UTC
    # stamps applied daylight saving to it: take the local stamps instead (None: trust the UTC stamps)
    clock_utc_offset_h: float | None = None
    first_day: str = "2021-03-01"
    last_day: str = "2023-02-28"
    note: str = ""


SITES = {
    "golden": SiteSpec("golden", 50, "Golden, CO", 39.742, -105.1727, 1994.7, "America/Denver", 45.0, 158.0, 6.0,
                       "BSk", "nrel_parquet", (752,),
                       note="45 deg tilt facing SSE (158 deg), cold semi-arid at 1.8 km with snow"),
    "henderson": SiteSpec("henderson", 1423, "Henderson, NV", 36.0275, -114.9215, 697.0, "America/Los_Angeles",
                          35.0, 180.0, 6.48, "BWh", "nrel_parquet", (4854, 4860), clock_utc_offset_h=-8.0,
                          note="24 x 270 W modules on two inverters (hot desert); its UTC stamps are an hour early "
                               "in summer (output centroid 55 min before solar noon Apr-Oct, on time in winter)"),
    "arbuckle": SiteSpec("arbuckle", 2107, "Arbuckle, CA", 38.996306, -122.134111, 10.0, "America/Los_Angeles",
                         25.0, 180.0, 893.0, "Csa", "prize_meter", last_day="2025-12-31",
                         note="farm array, DC/AC ~1.27 (clips near 705 kW on clear days); scaled to a home size"),
}


def site_and_planes(spec: SiteSpec, kwp: float | None = None) -> tuple[S.Site, tuple[S.Plane, ...]]:
    return S.Site(spec.lat, spec.lon, spec.alt), (S.Plane(spec.tilt, spec.azimuth, kwp or spec.kwp),)


# ---------------------------------------------------------------- downloads
def http_get(url: str, timeout: float = 180.0, retries: int = 10) -> bytes | None:
    """GET with patient retries (busy servers answer 429/5xx); None for a missing object (404/403 from S3)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if e.code != 429 and e.code < 500:
                raise
            why = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            why = type(e).__name__
        wait = min(20 * (attempt + 1), 120)
        print(f"  ({why}; retry {attempt + 1}/{retries - 1} in {wait} s: {url.split('?')[0]})", file=sys.stderr,
              flush=True)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def om_get(url: str, params: dict, pause_s: float = 1.0) -> dict:
    """Open-Meteo request, cached on disk for good (only finished archive periods are requested)."""
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
    path = DATA / "openmeteo" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    body = http_get(url + "?" + urllib.parse.urlencode(params))
    if body is None:
        raise RuntimeError(f"Open-Meteo returned nothing for {params}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    time.sleep(pause_s)                      # be gentle with the free API
    return json.loads(body)


def days(first: str, last: str) -> list[date]:
    d, e = date.fromisoformat(first), date.fromisoformat(last)
    return [d + timedelta(days=i) for i in range((e - d).days + 1)]


# ---------------------------------------------------------------- PVDAQ power
def _nrel_day_url(sid: int, d: date) -> str:
    return (f"{OEDI}pvdaq/parquet/pvdata/system_id={sid}/year={d.year}/month={d.month}/day={d.day}/"
            f"system_{sid}__date_{d:%Y_%m_%d}.snappy.000.parquet")


def epoch_s(index) -> np.ndarray:
    """DatetimeIndex (naive = UTC) -> float epoch seconds, whatever its resolution."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.as_unit("s").asi8.astype(float)


def _reduce_nrel_day(raw: bytes, metrics: tuple[int, ...],
                     clock_utc_offset_h: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One file of long-format PVDAQ data (a LOCAL day) -> five-minute mean AC power (W) by UTC
    interval start, only intervals with data.

    With several inverters a sample counts only if all report, and moments where one inverter is down
    while the other produces are dropped (an outage, not weather)."""
    t = pd.read_parquet(io.BytesIO(raw), columns=["measured_on", "utc_measured_on", "metric_id", "value"])
    t = t[t["metric_id"].isin(metrics)]
    if t.empty:
        return np.zeros(0), np.zeros(0)
    if clock_utc_offset_h is None:
        t["utc"] = t["utc_measured_on"]
    else:
        t["utc"] = t["measured_on"] - pd.Timedelta(hours=clock_utc_offset_h)
    w = t.pivot_table(index="utc", columns="metric_id", values="value", aggfunc="mean")
    w = w.reindex(columns=list(metrics))
    x = w.to_numpy(float)
    total = np.where(np.isnan(x).any(axis=1), np.nan, np.nansum(x, axis=1))
    if len(metrics) > 1:
        hi, lo = np.nanmax(x, axis=1), np.nanmin(x, axis=1)
        total[(hi > 200) & (lo < 0.2 * hi)] = np.nan
    s = pd.Series(np.clip(total, 0, None), index=pd.DatetimeIndex(w.index)).dropna()
    five = s.resample("5min").agg(["mean", "count"])
    five = five[five["count"] >= 2]
    return epoch_s(five.index), five["mean"].to_numpy(float)


def _nrel_power(spec: SiteSpec) -> tuple[np.ndarray, np.ndarray, int]:
    """Five-minute UTC series on a regular grid from the per-day files, each file cached reduced."""
    clock = spec.clock_utc_offset_h
    cache = DATA / "pvdaq" / str(spec.system_id) / ("days" if clock is None else f"days_clock{clock:+g}h")
    cache.mkdir(parents=True, exist_ok=True)
    dl = days(spec.first_day, spec.last_day)
    dl = [dl[0] - timedelta(days=1), *dl, dl[-1] + timedelta(days=1)]

    def one(d: date) -> tuple[np.ndarray, np.ndarray]:
        f = cache / f"{d.isoformat()}.npz"
        if f.exists():
            z = np.load(f)
            return z["ts"], z["v"]
        raw = http_get(_nrel_day_url(spec.system_id, d))
        ts, v = (np.zeros(0), np.zeros(0)) if raw is None else _reduce_nrel_day(raw, spec.metrics, clock)
        np.savez(f, ts=ts, v=v)
        return ts, v

    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(one, dl))
    t0 = pd.Timestamp(dl[0], tz="UTC").timestamp()
    grid = t0 + np.arange(288 * len(dl)) * 300.0
    out = np.full(len(grid), np.nan)
    for ts, v in parts:
        i = np.round((ts - t0) / 300).astype(int)
        ok = (i >= 0) & (i < len(grid))
        out[i[ok]] = v[ok]
    return grid, out, 300


def _prize_power(spec: SiteSpec) -> tuple[np.ndarray, np.ndarray, int]:
    """15-minute revenue-meter data (kW, local clock time US/Pacific stamped at the interval START:
    checked against the site's 5-minute plane-of-array sensor, whose cloud transients line up only
    with the meter window that begins at the stamp)."""
    folder = DATA / "pvdaq" / str(spec.system_id)
    files = [f"{spec.system_id}_meter_15m_data.csv", f"{spec.system_id}_meter_15m_data_2024.csv",
             f"{spec.system_id}_meter_15m_data_2025.csv"]
    frames = []
    for name in files:
        f = folder / name
        if not f.exists():
            raw = http_get(f"{OEDI}pvdaq/2023-solar-data-prize/{spec.system_id}_OEDI/data/{name}")
            if raw is None:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            f.write_bytes(raw)
        frames.append(pd.read_csv(f))
    d = pd.concat(frames, ignore_index=True)
    local = pd.to_datetime(d["measured_on"])
    utc = local.dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    kw = d.iloc[:, 1].to_numpy(float)
    ok = utc.notna().to_numpy()
    s = pd.Series(kw[ok], index=pd.DatetimeIndex(utc[ok]))
    s = s[~s.index.duplicated(keep="first")].sort_index()
    cap = 1.05 * s.quantile(0.999)
    s[s > cap] = np.nan                                                     # meter spikes
    first = pd.Timestamp(spec.first_day, tz="UTC") - pd.Timedelta(days=1)
    last = pd.Timestamp(spec.last_day, tz="UTC") + pd.Timedelta(days=2)
    grid = pd.date_range(first, last, freq="15min", inclusive="left")
    s = s.reindex(grid)
    return epoch_s(grid), 1000.0 * s.to_numpy(float), 900


def load_power(spec: SiteSpec) -> tuple[np.ndarray, np.ndarray, int]:
    """(interval starts UTC epoch s, AC power W, step s) for the site's data window."""
    return _nrel_power(spec) if spec.source == "nrel_parquet" else _prize_power(spec)


# ---------------------------------------------------------------- periods and weather
def aggregate(ts, values, step_s: int, starts, period_s: int = HOUR, min_frac: float = 0.8) -> dict[str, np.ndarray]:
    """Sub-period samples -> per-period mean/min/max (like Home Assistant statistics); a period needs
    `min_frac` of its samples, else NaN."""
    ts, v, starts = np.asarray(ts, float), np.asarray(values, float), np.asarray(starts, float)
    i = np.floor((ts - starts[0]) / period_s).astype(int)
    ok = (i >= 0) & (i < len(starts)) & ~np.isnan(v)
    n = len(starts)
    cnt = np.bincount(i[ok], minlength=n)
    tot = np.bincount(i[ok], weights=v[ok], minlength=n)
    mn, mx = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(mn, i[ok], v[ok])
    np.maximum.at(mx, i[ok], v[ok])
    full = cnt >= min_frac * (period_s / step_s)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"mean": np.where(full, tot / cnt, np.nan), "min": np.where(full, mn, np.nan),
                "max": np.where(full, mx, np.nan)}


def load_weather(spec: SiteSpec, first_day: str | None = None, last_day: str | None = None) -> dict[str, np.ndarray]:
    """Historical forecasts (the first hours of each run) for the site's plane, raw Open-Meteo stamps."""
    first = (date.fromisoformat(first_day or spec.first_day) - timedelta(days=1)).isoformat()
    last = (date.fromisoformat(last_day or spec.last_day) + timedelta(days=1)).isoformat()
    frames = []
    for a, b in W.date_chunks(first, last, 92):
        url, p = W.historical_forecast_request(spec.lat, spec.lon, a, b, spec.tilt, spec.azimuth)
        frames.append(W.parse(om_get(url, p)))
    return W.concat(frames)


def t_air_from(wx: dict[str, np.ndarray]):
    """Air temperature (C) at any time from the forecast archive (20 C where unknown)."""
    t = np.asarray(wx["temperature_2m"], float)
    ok = ~np.isnan(t)
    ts, tv = np.asarray(wx["ts"], float)[ok], t[ok]
    if not ok.any():
        return 20.0
    return lambda x: np.interp(np.asarray(x, float), ts, tv)


def hour_table(spec: SiteSpec, kwp: float | None = None, first_day: str | None = None,
               last_day: str | None = None) -> dict[str, np.ndarray]:
    """Hourly grid over the data window with the sun/clear-sky and forecast features (no PV yet)."""
    first = pd.Timestamp(first_day or spec.first_day, tz=spec.tz).tz_convert("UTC").floor("h").timestamp()
    last = (pd.Timestamp(last_day or spec.last_day, tz=spec.tz) + pd.Timedelta(days=1)).tz_convert("UTC").timestamp()
    hours = np.arange(first, last, HOUR, dtype=float)
    wx = load_weather(spec, first_day, last_day)
    site, planes = site_and_planes(spec, kwp)
    tab = F.sky_features(hours, HOUR, 12, site, planes, t_air=t_air_from(wx))
    fc = F.wx_for_periods(wx, hours, HOUR)
    fc["fc_kgti"] = F.forecast_kgti(fc, tab["poa_cs"])
    tab.update({k: v for k, v in fc.items() if k != "ts"})
    return tab


def write_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=float) + "\n")
    return path
