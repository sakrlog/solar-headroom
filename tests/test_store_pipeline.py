"""engine.store and the planning parts of engine.pipeline (no network, no Home Assistant)."""
from __future__ import annotations

from datetime import date

import numpy as np

from engine import history as H
from engine import pipeline as P
from engine import solar as S
from engine.store import Store

T0 = 1_700_000_000 - 1_700_000_000 % 3600


def test_store_rows_and_columns_agree(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    rows = [{"start": T0 + 3600 * i, "mean": float(i), "min": None if i == 2 else i - 0.5, "max": i + 0.5}
            for i in range(5)]
    assert store.put_rows("sensor.pv", rows) == 5
    assert store.put_rows("sensor.pv", rows[:1]) == 1                # replace, no duplicate
    cols = store.row_columns(["sensor.pv", "sensor.none"])
    assert np.isnan(cols["sensor.pv"]["min"][2]) and cols["sensor.pv"]["min"][3] == 2.5
    assert len(cols["sensor.none"]["start"]) == 0
    assert store.row_span("sensor.pv") == (T0, T0 + 4 * 3600)
    roles = {"pv_w": [[H.Source("sensor.pv", scale=1000.0)]]}
    a = H.period_table(roles, store.rows(["sensor.pv"]), T0, T0 + 6 * 3600)
    b = H.period_table(roles, cols, T0, T0 + 6 * 3600)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert np.isnan(a["pv_w"][5]) and a["pv_w"][4] == 4000.0
    part = store.row_columns(["sensor.pv"], start=T0 + 3600, end=T0 + 3 * 3600)["sensor.pv"]
    assert part["start"].tolist() == [T0 + 3600, T0 + 7200]


def test_store_responses_blobs_meta(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    store.put_response("hf/a/2024-01-01", "hf", {"hourly": {"time": [1, 2], "x": [None, 1.5]}}, "2024-01-01",
                       "2024-03-31", final=True)
    store.put_response("hf/a/2024-04-01", "hf", {"hourly": {"time": [3]}}, "2024-04-01", "2024-04-10")
    store.put_response("pr/b/2024-01-01", "pr", {"hourly": {"time": [1]}})
    assert [k for k, _ in store.responses("hf/a/")] == ["hf/a/2024-01-01", "hf/a/2024-04-01"]
    assert store.responses("hf/a/")[0][1]["hourly"]["x"] == [None, 1.5]
    idx = store.response_index()
    assert idx["hf/a/2024-01-01"][0] is True and idx["hf/a/2024-04-01"][0] is False
    assert store.drop_responses(["hf/a/2024-04-01"], "hf") == 1
    assert set(store.response_index()) == {"hf/a/2024-04-01", "pr/b/2024-01-01"}
    store.put_blob("models", {"k": [1.0, float("nan")]})
    assert np.isnan(store.get_blob("models")["k"][1]) and store.get_blob("none") is None
    store.set_meta("system", {"a": 1})
    assert store.get_meta("system") == {"a": 1} and store.get_meta("x", 7) == 7


def test_quarters_clip_and_finality():
    q = P.quarters(date(2024, 2, 10), date(2024, 11, 14))
    assert [(a.isoformat(), b.isoformat()) for a, b, _ in q] == [
        ("2024-02-10", "2024-03-31"), ("2024-04-01", "2024-06-30"), ("2024-07-01", "2024-09-30"),
        ("2024-10-01", "2024-11-14")]
    assert q[-1][2] == date(2024, 12, 31)


def _system(planes=((30.0, 180.0, 5.0),)) -> P.System:
    return P.System(45.0, 7.0, 300.0, "Europe/Rome", tuple(S.Plane(*p) for p in planes),
                    {"pv_w": [[{"statistic_id": "sensor.pv"}]]})


def test_archive_plan_keys_and_refresh():
    system = _system(((30.0, 180.0, 5.0), (15.0, 270.0, 2.0)))
    reqs = P.archive_requests(system, date(2024, 3, 1), date(2024, 11, 15))
    keys = {r.key for r in reqs}
    assert "hf/30_180/all/2024-01-01" not in keys and "hf/30_180/all/2024-03-01" in keys
    assert "pr/best_match/15_270/gti/2024-04-01" in keys
    gti_only = next(r for r in reqs if r.key == "hf/15_270/gti/2024-04-01")
    assert gti_only.params["hourly"] == "global_tilted_irradiance"
    assert gti_only.params["latitude"] == 45.0 and gti_only.params["azimuth"] == 90.0     # west, Open-Meteo style
    open_q = next(r for r in reqs if r.key == "hf/30_180/all/2024-10-01")
    assert not open_q.final and open_q.last_day == "2024-11-14"
    assert next(r for r in reqs if r.key == "hf/30_180/all/2024-07-01").final
    now = 1_000_000.0
    index = {r.key: (r.final, now - 3600) for r in reqs}
    assert P.missing(reqs, index, now) == []
    index[open_q.key] = (False, now - 2 * 86400)
    assert [r.key for r in P.missing(reqs, index, now)] == [open_q.key]
    assert P.System.from_dict(system.to_dict()) == system


def test_combine_planes_weights_gti_by_kwp():
    a = {"ts": np.array([0.0, 3600.0]), "global_tilted_irradiance": np.array([100.0, 200.0]),
         "global_tilted_irradiance_d1": np.array([10.0, 20.0]), "temperature_2m": np.array([5.0, 6.0])}
    b = {"ts": np.array([3600.0]), "global_tilted_irradiance": np.array([400.0]),
         "global_tilted_irradiance_d1": np.array([40.0])}
    c = P.combine_planes([a, b], [3.0, 1.0])
    assert np.isnan(c["global_tilted_irradiance"][0])                 # second group missing at that time
    assert c["global_tilted_irradiance"][1] == 0.75 * 200 + 0.25 * 400
    assert c["global_tilted_irradiance_d1"][1] == 0.75 * 20 + 0.25 * 40
    assert c["temperature_2m"].tolist() == [5.0, 6.0]


def test_satellite_choice_follows_coverage_and_speed():
    eu, us = _system(), P.System(39.7, -105.2, 1800.0, "America/Denver", (S.Plane(30.0, 180.0, 5.0),),
                                 {"pv_w": [[{"statistic_id": "sensor.pv"}]]})
    everything = {m: True for m in ("eumetsat_sarah3", "dwd_sis_europe_africa_v4", "jma_jaxa_mtg_fci", "noaa_goes19")}
    assert P.choose_satellites(eu, everything) == {"history": "eumetsat_sarah3", "live": "dwd_sis_europe_africa_v4"}
    assert P.choose_satellites(us, everything) == {"history": "noaa_goes19", "live": "noaa_goes19"}
    no_sarah = everything | {"eumetsat_sarah3": False}
    assert P.choose_satellites(eu, no_sarah)["history"] == "jma_jaxa_mtg_fci"
    assert P.choose_satellites(eu, {}) == {"history": None, "live": None}
    probes = P.probe_requests(eu, date(2026, 11, 15))
    assert {r.key for r in probes} == {"probe/eumetsat_sarah3", "probe/jma_jaxa_mtg_fci",
                                       "probe/dwd_sis_europe_africa_v4"}
    assert all(r.params["hourly"] == "shortwave_radiation" for r in probes)


def test_probe_ok_needs_daylight_values():
    t = [1_700_000_000 + 1800 * i for i in range(4)]
    body = {"hourly": {"time": t, "shortwave_radiation": [None, 0.0, 120.0, 80.0]}}
    assert P.probe_ok("eumetsat_sarah3", body)
    assert not P.probe_ok("eumetsat_sarah3", {"hourly": {"time": t, "shortwave_radiation": [None] * 4}})
    assert not P.probe_ok("eumetsat_sarah3", {"error": True, "reason": "no data"})


def test_satellite_archive_plan():
    system = _system(((30.0, 180.0, 5.0), (15.0, 270.0, 2.0)))
    sats = {"history": "eumetsat_sarah3", "live": "dwd_sis_europe_africa_v4",
            "offsets": {"dwd_sis_europe_africa_v4": [0, 2]}}
    reqs = P.satellite_requests(system, sats, date(2025, 11, 20), date(2026, 11, 15))
    keys = {r.key for r in reqs}
    assert "sat/eumetsat_sarah3/p+0+0/30_180/all/2025-11-20" in keys
    assert "sat/dwd_sis_europe_africa_v4/p+0+2/30_180/all/2026-02-14" in keys    # its archive starts later
    assert not any(k.startswith("sat/dwd_sis_europe_africa_v4/p+0+2/30_180/all/2025") for k in keys)
    second = next(r for r in reqs if r.key == "sat/eumetsat_sarah3/p+0+0/15_270/gti/2026-01-01")
    assert second.params["hourly"] == "global_tilted_irradiance" and second.params["temporal_resolution"] == "native"
    assert (second.params["latitude"], second.params["longitude"]) == (45.0, 7.0)
    dwd = next(r for r in reqs if r.key.startswith("sat/dwd"))
    assert (dwd.params["latitude"], dwd.params["longitude"]) == (45.0, 7.2)       # its pixel, 2 steps east
    live = P.live_satellite_requests(system, sats, 1_794_900_000.0)
    assert [r.key for r in live] == ["livesat/dwd_sis_europe_africa_v4/30_180/all",
                                     "livesat/dwd_sis_europe_africa_v4/15_270/gti"]
    assert live[0].params["longitude"] == 7.2


def test_pixel_hill_climb_finds_the_best_pixel():
    true = (1, 2)                               # the pixel that follows the PV, two steps from the first ring

    def corr(o):
        return 0.95 - 0.04 * ((o[0] - true[0]) ** 2 + (o[1] - true[1]) ** 2) ** 0.5

    scores = {}
    while todo := P.pixels_to_try(scores):
        assert all(max(abs(a), abs(b)) <= P.PIXEL_RADIUS for a, b in todo) and not set(todo) & set(scores)
        scores |= {o: (corr(o), 500) for o in todo}
    assert P.best_pixel(scores) == true and len(scores) <= P.PIXEL_MAX_REQUESTS
    flat = {o: (0.9 + (0.001 if o == (1, 1) else 0.0), 500) for o in P.pixels_to_try({})}
    assert P.best_pixel(flat) == (0, 0)                                        # no clear gain: own point
    assert P.best_pixel({(0, 0): (float("nan"), 3)}) == (0, 0)
    assert P.pixels_to_try({(0, 0): (float("nan"), 3)}) == []                 # no data: stop


def test_pixel_score_and_window():
    system = _system()
    w = P.pixel_window("dwd_sis_europe_africa_v4", date(2024, 1, 1), date(2026, 11, 15))
    assert w is not None and (w[1] - w[0]).days == P.PIXEL_DAYS
    assert P.pixel_window("dwd_sis_europe_africa_v4", date(2026, 11, 1), date(2026, 11, 15)) is None
    req = P.pixel_request(system, "eumetsat_sarah3", (-1, 1), w)
    assert req.params["hourly"] == "global_tilted_irradiance" and "temporal_resolution" not in req.params
    assert (req.params["latitude"], req.params["longitude"]) == (44.9, 7.1)
    ts = np.arange(T0, T0 + 30 * 86400, 3600.0)
    poa = S.poa_clearsky(S.solar(ts + 1800, system.site), system.planes[0])
    k = 0.3 + 0.7 * np.mod(np.arange(len(ts)) * 0.37, 1.0)
    day = poa > 200
    truth = {"ts": ts[day], "k": 0.8 * k[day]}
    body = {"hourly": {"time": (ts + 3600).astype(int).tolist(), "global_tilted_irradiance": (k * poa).tolist()}}
    c, n = P.pixel_score(system, truth, body)
    assert c > 0.999 and n == day.sum()
    noisy = {"hourly": {"time": body["hourly"]["time"], "global_tilted_irradiance": (k[::-1] * poa).tolist()}}
    assert P.pixel_score(system, truth, noisy)[0] < 0.5
    assert np.isnan(P.pixel_score(system, truth, {"error": True})[0])
