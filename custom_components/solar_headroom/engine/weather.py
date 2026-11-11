"""Open-Meteo: request builders and response parsers (no I/O; the caller fetches).

Inside Home Assistant the integration fetches with HA's aiohttp session; research scripts use
urllib. Everything here is pure: (url, params) in, numpy arrays out.

Privacy: only coordinates rounded to 0.1 deg (~10 km) and the panel tilt/direction are sent.
No key, account or identifier.

Time convention (checked against inverter data): radiation values stamped T are the mean over
the PRECEDING interval (T - step, T]. Parsers for radiation re-index to the interval START, like
Home Assistant statistics rows. Instantaneous variables (temperature, cloud cover, wind, humidity,
snow depth) keep their own timestamp; features.wx_for_periods handles both kinds.

Frames are dicts: {"ts": float epoch seconds (sorted, unique), "<var>": float arrays (NaN = missing)}.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

SATELLITE_URL = "https://satellite-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"

RAD_VARS = ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "global_tilted_irradiance")
SHORT = {"shortwave_radiation": "ghi", "direct_normal_irradiance": "dni", "diffuse_radiation": "dhi",
         "global_tilted_irradiance": "gti"}
# sums/means over the preceding hour; everything else is instantaneous
INTERVAL_VARS = frozenset({"precipitation", "sunshine_duration", "snowfall", *RAD_VARS})
WX_VARS = ("temperature_2m", "relative_humidity_2m", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
           "cloud_cover_high", "wind_speed_10m", "precipitation", "sunshine_duration", "snowfall", "snow_depth",
           *RAD_VARS)
# forecasts as issued N days before each hour ("previous runs"); best_match has day-ahead radiation from
# ~2024-01; the single models give the spread (their disagreement = uncertainty)
PREVIOUS_RUNS_START = "2024-01-15"
DAILY_VARS = ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "global_tilted_irradiance",
              "cloud_cover", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
              "sunshine_duration", "cape", "snowfall", "snow_depth")
ENSEMBLE = {"gfs_seamless": "gfs", "icon_seamless": "icon", "meteofrance_seamless": "mf"}
PREVIOUS_RUNS_MODELS = {"best_match": DAILY_VARS,
                        **{m: ("global_tilted_irradiance", "cloud_cover") for m in ENSEMBLE}}


@dataclass(frozen=True)
class SatelliteSource:
    model: str
    step_s: int            # native interval
    delay_s: int           # typical publication delay after the interval ends
    since: str             # first day of the archive
    lon_min: float         # rough footprint (deg east); availability is confirmed by a probe at setup
    lon_max: float
    lat_max: float = 65.0
    live: bool = True      # updated often enough for nowcasting


# Open-Meteo satellite products (model ids as in the API). Footprints are generous on purpose:
# the integration probes candidates once and keeps the ones that return data.
SATELLITES = {
    "eumetsat_sarah3": SatelliteSource("eumetsat_sarah3", 1800, 2 * 86400, "1983-01-01", -65, 65, live=False),
    "dwd_sis_europe_africa_v4": SatelliteSource("dwd_sis_europe_africa_v4", 600, 1200, "2026-02-14", -65, 65),
    "jma_jaxa_mtg_fci": SatelliteSource("jma_jaxa_mtg_fci", 600, 1800, "2025-01-01", -65, 100),
    "noaa_goes19": SatelliteSource("noaa_goes19", 600, 1200, "2025-04-07", -145, -5),
    "jma_jaxa_himawari": SatelliteSource("jma_jaxa_himawari", 600, 1800, "2023-01-01", 70, 215),
}


def satellite_candidates(lat: float, lon: float) -> dict[str, list[str]]:
    """Ordered candidates for the point: 'history' (longest archive first) and 'live' (fastest first)."""
    lon_e = lon if lon >= -180 else lon + 360

    def covers(s: SatelliteSource) -> bool:
        lons = (lon_e, lon_e + 360, lon_e - 360)
        return abs(lat) <= s.lat_max and any(s.lon_min <= x <= s.lon_max for x in lons)
    found = [s for s in SATELLITES.values() if covers(s)]
    history = sorted(found, key=lambda s: s.since)
    live = sorted((s for s in found if s.live), key=lambda s: (s.delay_s, s.since))
    return {"history": [s.model for s in history], "live": [s.model for s in live]}


# ---------------------------------------------------------------- requests
def round_coord(value: float, step: float = 0.1) -> float:
    """Coordinates leave the house only on a 0.1 deg grid."""
    return round(round(float(value) / step) * step, 4)


def om_azimuth(azimuth_from_north: float) -> float:
    """Engine azimuth (clockwise from north, 180 = south) -> Open-Meteo (0 = south, -90 = east, 90 = west)."""
    a = float(azimuth_from_north) - 180.0
    return a + 360.0 if a < -180.0 else (a - 360.0 if a >= 180.0 else a)


def _base(lat: float, lon: float, tilt: float | None, azimuth: float | None) -> dict:
    p = {"latitude": round_coord(lat), "longitude": round_coord(lon), "timeformat": "unixtime", "timezone": "GMT"}
    if tilt is not None and azimuth is not None:
        p["tilt"], p["azimuth"] = float(tilt), om_azimuth(azimuth)
    return p


def satellite_request(model: str, lat: float, lon: float, start_date: str, end_date: str,
                      tilt: float | None = None, azimuth: float | None = None, native: bool = True) -> tuple[str, dict]:
    p = _base(lat, lon, tilt, azimuth) | {"start_date": start_date, "end_date": end_date,
                                          "hourly": ",".join(RAD_VARS), "models": model}
    if native:
        p["temporal_resolution"] = "native"
    return SATELLITE_URL, p


def historical_forecast_request(lat: float, lon: float, start_date: str, end_date: str, tilt: float, azimuth: float,
                                variables: Sequence[str] = WX_VARS) -> tuple[str, dict]:
    """Archived forecasts stitched from the first hours of each run (input of the nowcast sky model)."""
    return HISTORICAL_FORECAST_URL, _base(lat, lon, tilt, azimuth) | {
        "start_date": start_date, "end_date": end_date, "hourly": ",".join(variables)}


def lead_names(variables: Iterable[str], leads: Iterable[int]) -> dict[str, str]:
    """Open-Meteo previous-runs names -> '<var>_d<lead>' (lead 0 = newest run)."""
    return {(v if n == 0 else f"{v}_previous_day{n}"): f"{v}_d{n}" for n in leads for v in variables}


def previous_runs_request(lat: float, lon: float, start_date: str, end_date: str, tilt: float, azimuth: float,
                          model: str = "best_match", variables: Sequence[str] | None = None,
                          leads: Sequence[int] = (0, 1)) -> tuple[str, dict]:
    variables = PREVIOUS_RUNS_MODELS[model] if variables is None else variables
    p = _base(lat, lon, tilt, azimuth) | {"start_date": start_date, "end_date": end_date,
                                          "hourly": ",".join(lead_names(variables, leads))}
    if model != "best_match":
        p["models"] = model
    return PREVIOUS_RUNS_URL, p


def forecast_request(lat: float, lon: float, tilt: float, azimuth: float, variables: Sequence[str] = WX_VARS,
                     model: str = "best_match", past_days: int = 1, forecast_days: int = 3) -> tuple[str, dict]:
    p = _base(lat, lon, tilt, azimuth) | {"hourly": ",".join(variables), "past_days": past_days,
                                          "forecast_days": forecast_days}
    if model != "best_match":
        p["models"] = model
    return FORECAST_URL, p


def date_chunks(start_date: str, end_date: str, days: int = 92) -> list[tuple[str, str]]:
    """Inclusive [first, last] date windows for archive requests."""
    s, e = date.fromisoformat(start_date), date.fromisoformat(end_date)
    out = []
    while s <= e:
        c = min(s + timedelta(days=days - 1), e)
        out.append((s.isoformat(), c.isoformat()))
        s = c + timedelta(days=1)
    return out


def is_final(end_date: str, today: str, settle_days: int = 4) -> bool:
    """Archive responses ending this long ago no longer change and can be stored for good."""
    return date.fromisoformat(end_date) < date.fromisoformat(today) - timedelta(days=settle_days)


# ---------------------------------------------------------------- parsing
def parse(body: Mapping, block: str = "hourly", rename: Mapping[str, str] | None = None) -> dict[str, np.ndarray]:
    """One response block -> frame at the raw Open-Meteo timestamps."""
    h = body[block]
    out = {"ts": np.asarray(h["time"], dtype=float)}
    for k, v in h.items():
        if k == "time":
            continue
        out[(rename or {}).get(k, k)] = np.array([np.nan if x is None else x for x in v], dtype=float)
    return out


def shift(frame: Mapping[str, np.ndarray], seconds: float) -> dict[str, np.ndarray]:
    out = dict(frame)
    out["ts"] = np.asarray(frame["ts"], dtype=float) + seconds
    return out


def drop_empty(frame: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rows where every variable is missing are removed."""
    cols = [k for k in frame if k != "ts"]
    if not cols:
        return dict(frame)
    keep = ~np.all(np.isnan(np.vstack([frame[k] for k in cols])), axis=0)
    return {k: np.asarray(v)[keep] for k, v in frame.items()}


def parse_satellite(body: Mapping, step_s: int) -> dict[str, np.ndarray]:
    """Native satellite response -> W/m2 (ghi, dni, dhi, gti) indexed by interval START."""
    return drop_empty(shift(parse(body, rename=SHORT), -step_s))


def concat(frames: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack frames (e.g. date chunks); for duplicate timestamps the earlier frame wins; sorted by time."""
    frames = [f for f in frames if len(f["ts"])]
    if not frames:
        return {"ts": np.zeros(0)}
    cols = list(dict.fromkeys(k for f in frames for k in f if k != "ts"))
    ts = np.concatenate([np.asarray(f["ts"], float) for f in frames])
    data = {c: np.concatenate([np.asarray(f[c], float) if c in f else np.full(len(f["ts"]), np.nan)
                               for f in frames]) for c in cols}
    _, first = np.unique(ts, return_index=True)          # sorted unique times, first occurrence
    return {"ts": ts[first], **{c: v[first] for c, v in data.items()}}


def split_leads(frame: Mapping[str, np.ndarray], leads: Iterable[int]) -> dict[int, dict[str, np.ndarray]]:
    """'<var>_d<lead>' columns -> one frame per lead with the base variable names."""
    out = {}
    for n in leads:
        sfx = f"_d{n}"
        out[n] = {"ts": frame["ts"], **{k[: -len(sfx)]: v for k, v in frame.items() if k.endswith(sfx)}}
    return out


def combine_planes(frames: Sequence[Mapping[str, np.ndarray]], kwp: Sequence[float]) -> dict[str, np.ndarray]:
    """Per-plane responses (same times; GTI differs per tilt/direction) -> one frame whose GTI is the
    kWp-weighted mean over the planes. Other variables come from the first plane."""
    if len(frames) == 1:
        return dict(frames[0])
    base = dict(frames[0])
    w = np.asarray(kwp, dtype=float) / float(np.sum(kwp))
    ts = np.asarray(base["ts"], float)
    acc = np.zeros(len(ts))
    for f, wi in zip(frames, w, strict=True):
        i = np.searchsorted(f["ts"], ts)
        ok = (i < len(f["ts"])) & (np.asarray(f["ts"])[np.minimum(i, len(f["ts"]) - 1)] == ts)
        g = np.full(len(ts), np.nan)
        g[ok] = np.asarray(f["gti"])[i[ok]]
        acc = acc + wi * g
    base["gti"] = acc
    return base
