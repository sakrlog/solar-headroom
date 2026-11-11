"""engine.history: statistics rows (recorder format) -> aligned period arrays."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from engine import history as H

T0 = 1_700_000_000 - 1_700_000_000 % 3600      # an hour boundary


def rows(values, start=T0, step=3600, spread=10.0):
    return [{"start": start + i * step, "mean": v, "min": None if v is None else v - spread,
             "max": None if v is None else v + spread} for i, v in enumerate(values)]


def test_rows_on_grid_scale_and_sign():
    starts = H.grid(T0, T0 + 4 * 3600, 3600)
    a = H.rows_to_arrays(rows([1.0, 2.0, None]), starts, scale=1000)          # kW -> W
    assert np.allclose(a["mean"][:2], [1000, 2000]) and np.isnan(a["mean"][2:]).all()
    assert np.allclose(a["min"][:2], [-9000, -8000])
    b = H.rows_to_arrays(rows([500.0]), starts, scale=-1)                     # + = charging -> + = discharging
    assert b["mean"][0] == -500 and b["min"][0] == -510 and b["max"][0] == -490


def test_off_grid_rows_and_datetimes():
    starts = H.grid(T0, T0 + 3 * 3600, 3600)
    off = [{"start": T0 + 1800, "mean": 5.0, "min": 5.0, "max": 5.0}]        # 5-minute row in an hourly query
    assert np.isnan(H.rows_to_arrays(off, starts)["mean"]).all()
    dt = [{"start": datetime.fromtimestamp(T0 + 3600, tz=UTC), "mean": 7.0, "min": 6.0, "max": 8.0}]
    assert H.rows_to_arrays(dt, starts)["mean"][1] == 7.0


def test_renamed_sensor_newest_first_with_windows():
    starts = H.grid(T0, T0 + 4 * 3600, 3600)
    stats = {"sensor.new": rows([None, None, 30.0, 40.0]), "sensor.old": rows([10.0, 20.0, 99.0, None])}
    part = [H.Source("sensor.new"), H.Source("sensor.old")]
    a, applicable = H.part_arrays(part, stats, starts)
    assert list(a["mean"]) == [10.0, 20.0, 30.0, 40.0] and applicable.all()


def test_sum_of_parts_requires_every_applicable_part():
    starts = H.grid(T0, T0 + 4 * 3600, 3600)
    switch = T0 + 2 * 3600                                                   # string 1 removed here
    stats = {"pv1": rows([100.0, None, 5.0, 5.0]), "pv2": rows([200.0, 200.0, 300.0, None])}
    role = [[H.Source("pv2")], [H.Source("pv1", until=switch)]]
    a = H.role_arrays(role, stats, starts)
    assert a["mean"][0] == 300.0
    assert np.isnan(a["mean"][1])               # pv1 should report but has no row: unknown
    assert a["mean"][2] == 300.0                # pv1 after its window is ignored
    assert np.isnan(a["mean"][3])
    assert a["max"][0] == 320.0 and a["min"][0] == 280.0


def test_period_table_and_helpers():
    stats = {"sensor.pv": rows([0.0, 1.5, 2.5], spread=0.5), "sensor.soc": rows([50.0, 60.0, 70.0])}
    roles = {"pv_w": [[H.Source("sensor.pv", scale=1000)]], "soc": [[H.Source("sensor.soc")]]}
    tab = H.period_table(roles, stats, T0 + 10, T0 + 3 * 3600, 3600)       # start floored to the hour
    assert tab["ts"][0] == T0 and len(tab["ts"]) == 3
    assert list(tab["pv_w"]) == [0.0, 1500.0, 2500.0] and tab["pv_w_max"][2] == 3000.0
    assert tab["soc_max"][1] == 70.0
    assert H.statistic_ids(roles) == ["sensor.pv", "sensor.soc"]
    assert H.coverage(tab, "pv_w") == 1.0
    windows = H.chunks(T0, T0 + 70 * 86400, days=31)
    assert len(windows) == 3 and windows[0][0] == T0 and windows[-1][1] == T0 + 70 * 86400
    assert all(b == c for (_, b), (c, _) in zip(windows, windows[1:], strict=False))


def test_mean_missing_uses_midpoint():
    starts = H.grid(T0, T0 + 3600, 3600)
    a = H.rows_to_arrays([{"start": T0, "mean": None, "min": 2.0, "max": 4.0}], starts)
    assert a["mean"][0] == pytest.approx(3.0)
