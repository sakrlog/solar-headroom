"""Home Assistant tests (need .venv-ha: pytest-homeassistant-custom-component). Skipped elsewhere.

A fake Open-Meteo answers every endpoint the integration uses with one consistent synthetic sky:
each UTC day has a clearness k in 0.25..1.0, the forecasts know it, and the fake PV history follows
it (0.85 x k x clear-sky power, 40 % in the afternoon shade), so there is something to learn
that physics alone cannot know.
"""
from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from urllib.parse import parse_qs

import numpy as np
import pytest

pytest.importorskip("pytest_homeassistant_custom_component")

from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMockResponse  # noqa: E402

from engine import features as F  # noqa: E402
from engine import solar as S  # noqa: E402
from engine import weather as W  # noqa: E402

LAT, LON, ELEV, TZ = 45.0, 7.0, 300.0, "Europe/Rome"
KWP, TILT, AZIMUTH = 5.0, 30.0, 180.0
PV_ID = "sensor.pv_power"
SYSTEM_LOSS = 0.85
URLS = ("https://api.open-meteo.com/v1/forecast", "https://historical-forecast-api.open-meteo.com/v1/forecast",
        "https://previous-runs-api.open-meteo.com/v1/forecast", "https://satellite-api.open-meteo.com/v1/archive")
SAT_PUBLISHED_AFTER_S = 900       # the fake satellite publishes an interval 15 min after it ends
SAT_TRUE_OFFSET = (0, 1)          # parallax: the fake satellite sees the site's sky one 0.1 deg step east


def day_k(ts) -> np.ndarray:
    """Clearness of the UTC day of each time: deterministic, 0.25..1.0."""
    d = np.floor(np.asarray(ts, float) / 86400.0)
    return 0.25 + 0.75 * np.mod(np.abs(np.sin(d * 12.9898) * 43758.5453), 1.0)


def sat_k(ts, lat: float, lon: float) -> np.ndarray:
    """Fake satellite clearness at a grid point: the site's sky at the true pixel, more and more
    unrelated hourly noise further away."""
    ts = np.asarray(ts, float)
    d = math.hypot(lat - (LAT + 0.1 * SAT_TRUE_OFFSET[0]), lon - (LON + 0.1 * SAT_TRUE_OFFSET[1])) / 0.1
    w = min(0.8, 0.3 * d)
    noise = 0.25 + 0.75 * np.mod(np.abs(np.sin(np.floor(ts / 3600) * 78.233 + lat * 12.9898) * 43758.5453), 1.0)
    return (1 - w) * day_k(ts) + w * noise


def clear_sky(starts, tilt: float = TILT, azimuth: float = AZIMUTH, kwp: float = KWP, period: float = 3600) -> dict:
    return F.sky_features(np.asarray(starts, float), period, 4, S.Site(LAT, LON, ELEV), [S.Plane(tilt, azimuth, kwp)])


def pv_rows(first: datetime, last: datetime) -> list[dict]:
    """Hourly statistics rows of the fake PV sensor (W), as the recorder stores them."""
    starts = np.arange(first.timestamp(), last.timestamp(), 3600.0)
    sky = clear_sky(starts)
    shade = np.where(sky["hour"] >= 14.5, 0.4, 1.0)     # a tree in the afternoon: physics cannot know it
    p = SYSTEM_LOSS * day_k(starts) * sky["p_cs"] * shade
    return [{"start": datetime.fromtimestamp(t, UTC), "mean": float(w), "min": float(0.6 * w), "max": float(1.2 * w)}
            for t, w in zip(starts, p, strict=True)]


def _step(q: dict) -> float:
    if q.get("temporal_resolution") == "native" and q.get("models") in W.SATELLITES:
        return float(W.SATELLITES[q["models"]].step_s)
    return 3600.0


def _times(q: dict) -> np.ndarray:
    """Open-Meteo stamps (interval END for radiation) for the request's date window; satellites only up
    to what is published."""
    if "start_date" in q:
        a, b = date.fromisoformat(q["start_date"]), date.fromisoformat(q["end_date"])
    else:
        today = datetime.now(UTC).date()
        a = today - timedelta(days=int(q.get("past_days", 0)))
        b = today + timedelta(days=int(q.get("forecast_days", 7)) - 1)
    t0 = datetime(a.year, a.month, a.day, tzinfo=UTC).timestamp()
    t1 = datetime(b.year, b.month, b.day, tzinfo=UTC).timestamp() + 86400
    step = _step(q)
    t = np.arange(t0 + step, t1 + 1, step) if step < 3600 or "temporal_resolution" in q else np.arange(t0, t1, step)
    if "temporal_resolution" in q:
        t = t[t <= datetime.now(UTC).timestamp() - SAT_PUBLISHED_AFTER_S]
    return t


def _values(base: str, t: np.ndarray, q: dict) -> np.ndarray:
    om_az, step = float(q.get("azimuth", 0.0)), _step(q)
    sky = clear_sky(t - step, float(q.get("tilt", TILT)), om_az + 180.0, 1.0, step)   # radiation: preceding step
    if q.get("models") in W.SATELLITES:
        k = sat_k(t - step, float(q["latitude"]), float(q["longitude"]))
    else:
        k = day_k(t - step)
    vals = {"global_tilted_irradiance": sky["poa_cs"] * k, "shortwave_radiation": sky["ghi_cs"] * k,
            "direct_normal_irradiance": sky["dni_cs"] * k ** 2, "diffuse_radiation": sky["ghi_cs"] * k * (1 - k) + 20,
            "cloud_cover": 100 * (1 - k), "cloud_cover_low": 60 * (1 - k), "cloud_cover_mid": 30 * (1 - k),
            "cloud_cover_high": 10 * (1 - k), "temperature_2m": 15 + 5 * k, "relative_humidity_2m": 80 - 30 * k,
            "sunshine_duration": 3600 * k * (sky["ghi_cs"] > 50), "wind_speed_10m": np.full(len(t), 8.0)}
    return np.round(vals.get(base, np.zeros(len(t))), 2)


def open_meteo_body(query: str) -> dict:
    q = {k: v[0] for k, v in parse_qs(query).items()}
    t = _times(q)
    models = q.get("models", "best_match").split(",")
    hourly: dict = {"time": [int(x) for x in t]}
    for name in q["hourly"].split(","):
        base = name.split("_previous_day")[0]
        for m in models:
            col = name if len(models) == 1 else f"{name}_{m}"
            hourly[col] = [None if math.isnan(v) else float(v) for v in _values(base, t, q)]
    return {"latitude": float(q["latitude"]), "longitude": float(q["longitude"]), "hourly": hourly}


@pytest.fixture
def open_meteo(aioclient_mock):
    """Register the fake Open-Meteo on every endpoint; returns the mocker (mock_calls lists requests)."""
    async def answer(method, url, data):
        return AiohttpClientMockResponse(method, url, json=open_meteo_body(url.query_string))
    for u in URLS:
        aioclient_mock.get(u, side_effect=answer)
    return aioclient_mock


@pytest.fixture
async def configured(hass):
    """Home Assistant at the fake site."""
    await hass.config.async_update(latitude=LAT, longitude=LON, elevation=ELEV, time_zone=TZ)
    return hass


@pytest.fixture(autouse=True)
def _ha(recorder_mock, enable_custom_integrations, monkeypatch):
    """Every test: recorder first (the integration depends on it), then custom integrations enabled; no
    pause between requests to the fake Open-Meteo."""
    monkeypatch.setattr("custom_components.solar_headroom.coordinator.REQUEST_PAUSE_S", 0)
    yield
