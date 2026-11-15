"""engine.weather: Open-Meteo request builders and parsers (no network)."""
from __future__ import annotations

import numpy as np
import pytest

from engine import weather as W


def test_coordinates_and_azimuth_convention():
    assert W.round_coord(52.5162) == 52.5 and W.round_coord(-33.8688) == -33.9 and W.round_coord(13.37) == 13.4
    assert W.om_azimuth(180) == 0 and W.om_azimuth(90) == -90 and W.om_azimuth(270) == 90
    assert W.om_azimuth(168) == -12 and W.om_azimuth(0) == -180 and W.om_azimuth(359) == 179


def test_requests_send_only_rounded_points():
    url, p = W.satellite_request("eumetsat_sarah3", 52.5162, 13.3777, "2025-01-01", "2025-01-31", 30, 200)
    assert url == W.SATELLITE_URL
    assert p["latitude"] == 52.5 and p["longitude"] == 13.4 and p["azimuth"] == 20 and p["tilt"] == 30
    assert p["models"] == "eumetsat_sarah3" and p["temporal_resolution"] == "native"
    assert p["timeformat"] == "unixtime" and p["timezone"] == "GMT"
    _, p = W.previous_runs_request(1.234, 2.345, "2025-01-01", "2025-01-02", 10, 180, model="gfs_seamless")
    assert p["models"] == "gfs_seamless"
    assert p["hourly"].split(",") == ["global_tilted_irradiance", "cloud_cover",
                                      "global_tilted_irradiance_previous_day1", "cloud_cover_previous_day1"]
    _, p = W.forecast_request(1, 2, 10, 180)
    assert "models" not in p and p["past_days"] == 1 and "snow_depth" in p["hourly"]
    _, p = W.historical_forecast_request(1, 2, "2025-01-01", "2025-01-02", 10, 180)
    assert "snowfall" in p["hourly"]


def test_parse_and_satellite_interval_start():
    body = {"hourly": {"time": [3600, 4200, 4800], "shortwave_radiation": [100, None, None],
                       "direct_normal_irradiance": [50, None, None], "diffuse_radiation": [40, None, None],
                       "global_tilted_irradiance": [110, 5, None]}}
    f = W.parse(body)
    assert np.isnan(f["shortwave_radiation"][1]) and f["ts"][0] == 3600
    s = W.parse_satellite(body, 600)
    assert list(s["ts"]) == [3000, 3600]                     # value stamped T = mean of (T-step, T]
    assert s["gti"][0] == 110 and np.isnan(s["ghi"][1])       # last row (all missing) dropped


def test_concat_split_leads_and_chunks():
    a = {"ts": np.array([0.0, 3600.0]), "x_d0": np.array([1.0, 2.0])}
    b = {"ts": np.array([3600.0, 7200.0]), "x_d0": np.array([9.0, 3.0]), "x_d1": np.array([5.0, 6.0])}
    c = W.concat([a, b])
    assert list(c["ts"]) == [0, 3600, 7200] and list(c["x_d0"]) == [1, 2, 3]   # earlier frame wins
    assert np.isnan(c["x_d1"][0]) and c["x_d1"][2] == 6
    leads = W.split_leads(c, (0, 1))
    assert list(leads[0]) == ["ts", "x"] and leads[1]["x"][2] == 6
    names = W.lead_names(["a"], (0, 1, 2))
    assert names == {"a": "a_d0", "a_previous_day1": "a_d1", "a_previous_day2": "a_d2"}
    ch = W.date_chunks("2025-01-01", "2025-03-01", 31)
    assert ch[0] == ("2025-01-01", "2025-01-31") and ch[-1][1] == "2025-03-01"
    assert W.is_final("2025-01-01", "2025-01-10") and not W.is_final("2025-01-08", "2025-01-10")
    assert W.concat([])["ts"].size == 0


@pytest.mark.parametrize("lat,lon,history,live", [
    (52.5, 13.4, "eumetsat_sarah3", "dwd_sis_europe_africa_v4"),     # Berlin
    (-1.3, 36.8, "eumetsat_sarah3", "dwd_sis_europe_africa_v4"),     # Nairobi
    (33.4, -112.1, "noaa_goes19", "noaa_goes19"),                    # Phoenix
    (-33.9, 151.2, "jma_jaxa_himawari", "jma_jaxa_himawari"),        # Sydney
    (28.6, 77.2, "jma_jaxa_himawari", "jma_jaxa_mtg_fci"),           # Delhi (both disks)
])
def test_satellite_candidates(lat, lon, history, live):
    c = W.satellite_candidates(lat, lon)
    assert c["history"][0] == history
    assert live in c["live"]
    assert "eumetsat_sarah3" not in c["live"]


def test_no_satellite_near_the_poles():
    assert W.satellite_candidates(78.2, 15.6) == {"history": [], "live": []}


def test_combine_planes_weights_gti_by_kwp():
    t = np.array([0.0, 600.0])
    east = {"ts": t, "ghi": np.array([100.0, 200.0]), "gti": np.array([300.0, 100.0])}
    west = {"ts": t, "ghi": np.array([100.0, 200.0]), "gti": np.array([100.0, np.nan])}
    out = W.combine_planes([east, west], [3.0, 1.0])
    assert out["gti"][0] == pytest.approx(250.0) and np.isnan(out["gti"][1])
    assert list(out["ghi"]) == [100, 200]
    assert W.combine_planes([east], [1.0])["gti"][0] == 300
