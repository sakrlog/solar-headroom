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
