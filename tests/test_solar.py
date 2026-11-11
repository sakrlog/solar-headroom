"""Parity of engine.solar (numpy only) with pvlib, which the model was developed with.

pvlib, pandas and h5py are dev-only dependencies; these tests are skipped without them.
"""
from __future__ import annotations

import numpy as np
import pytest
from engine import solar as S

pvlib = pytest.importorskip("pvlib")
pd = pytest.importorskip("pandas")
h5py = pytest.importorskip("h5py")

# public reference places (never a user's site): name, lat, lon, altitude
PLACES = [("berlin", 52.52, 13.40, 34.0), ("phoenix", 33.45, -112.07, 331.0), ("sydney", -33.87, 151.21, 58.0),
          ("nairobi", -1.29, 36.82, 1795.0), ("tromso", 69.65, 18.96, 10.0), ("la_paz", -16.50, -68.15, 3640.0)]
TIMES = pd.date_range("2025-01-01", "2025-12-31 23:50", freq="50min", tz="UTC")   # odd step: all minutes/hours
def epoch(idx) -> np.ndarray:
    """UTC epoch seconds of a DatetimeIndex, whatever its resolution (pandas 3 uses microseconds)."""
    return ((idx - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1s")).to_numpy(dtype=float)


TS = epoch(TIMES)


def monthly_tl(lat: float, lon: float) -> tuple[float, ...]:
    path = pvlib.clearsky.__file__.replace("clearsky.py", "data/LinkeTurbidities.h5")
    with h5py.File(path, "r") as f:
        raw = f["LinkeTurbidity"][pvlib.clearsky._degrees_to_index(lat, "latitude"),
                                  pvlib.clearsky._degrees_to_index(lon, "longitude")]
    return tuple(float(v) / 20.0 for v in raw)


@pytest.fixture(scope="module", params=PLACES, ids=[p[0] for p in PLACES])
def place(request):
    name, lat, lon, alt = request.param
    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=alt)
    sp = loc.get_solarposition(TIMES)
    cs = loc.get_clearsky(TIMES, model="ineichen", solar_position=sp)
    site = S.Site(lat, lon, alt, monthly_tl(lat, lon))
    return {"site": site, "loc": loc, "sp": sp, "cs": cs, "mine": S.solar(TS, site)}


def test_calendar_helpers():
    doy, leap = S.day_of_year(TS)
    assert np.array_equal(doy, TIMES.day_of_year.to_numpy())
    assert not leap.any()
    t2024 = pd.date_range("2024-02-28", "2024-03-02", freq="7h", tz="UTC")
    d, lp = S.day_of_year(epoch(t2024))
    assert np.array_equal(d, t2024.day_of_year.to_numpy()) and lp.all()
    assert np.array_equal(S.utc_month(TS), TIMES.month.to_numpy())


def test_extra_radiation():
    assert np.allclose(S.extra_radiation(TS), pvlib.irradiance.get_extra_radiation(TIMES).to_numpy(), rtol=1e-12)


def test_linke_turbidity_interpolation(place):
    ref = pvlib.clearsky.lookup_linke_turbidity(TIMES, place["site"].latitude, place["site"].longitude).to_numpy()
    assert np.allclose(S.linke_turbidity(TS, place["site"].linke_turbidity), ref, atol=1e-9)


def test_sun_position(place):
    sp, mine = place["sp"], place["mine"]
    up = sp["apparent_elevation"].to_numpy() > 1.0
    d_el = np.abs(mine["sun_el"] - sp["apparent_elevation"].to_numpy())[up]
    assert d_el.max() < 0.05, d_el.max()
    # angle between the two sun directions (azimuth alone is ill-defined near the zenith)
    def vec(el, az):
        el, az = np.radians(el), np.radians(az)
        return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    cosang = np.sum(vec(mine["sun_el"], mine["sun_az"]) * vec(sp["apparent_elevation"].to_numpy(),
                                                               sp["azimuth"].to_numpy()), axis=0)
    sep = np.degrees(np.arccos(np.clip(cosang, -1, 1)))[up]
    assert sep.max() < 0.05, sep.max()


def test_clearsky(place):
    cs, mine = place["cs"], place["mine"]
    day = cs["ghi"].to_numpy() > 50
    for ours, theirs in (("ghi_cs", "ghi"), ("dni_cs", "dni"), ("dhi_cs", "dhi")):
        a, b = mine[ours][day], cs[theirs].to_numpy()[day]
        rel = np.abs(a - b) / np.maximum(b, 20)
        assert rel.max() < 0.01, (ours, rel.max())
        assert abs(a.sum() / b.sum() - 1) < 0.001, (ours, a.sum() / b.sum())


@pytest.mark.parametrize("tilt,az", [(10, 168), (30, 180), (35, 90), (35, 270), (60, 200), (0, 180)])
def test_poa_clearsky(place, tilt, az):
    sp, cs, mine = place["sp"], place["cs"], place["mine"]
    irr = pvlib.irradiance.get_total_irradiance(tilt, az, sp["apparent_zenith"], sp["azimuth"], cs["dni"], cs["ghi"],
                                                cs["dhi"], dni_extra=pvlib.irradiance.get_extra_radiation(TIMES),
                                                model="haydavies", albedo=0.2)
    aoi = pvlib.irradiance.aoi(tilt, az, sp["apparent_zenith"], sp["azimuth"])
    ref = ((irr["poa_direct"] * pvlib.iam.physical(aoi)).fillna(0) + irr["poa_diffuse"].fillna(0)).to_numpy()
    ref = np.clip(np.where(sp["apparent_elevation"].to_numpy() <= 0, 0, ref), 0, None)
    ours = S.poa_clearsky(mine, S.Plane(tilt, az, 1.0))
    day = ref > 50
    excess = np.abs(ours[day] - ref[day]) - (0.01 * ref[day] + 1.0)     # 1 % + 1 W/m2
    assert excess.max() <= 0, excess.max()
    assert abs(ours[day].sum() / ref[day].sum() - 1) < 0.001


def test_iam_physical_matches_pvlib():
    aoi = np.linspace(0, 100, 401)
    assert np.allclose(S.iam_physical(aoi), pvlib.iam.physical(aoi), atol=1e-12, equal_nan=True)


def test_period_solar_hourly_power():
    """Hourly means from 12 sub-steps, as the training tables use them (vs a pandas/pvlib version)."""
    name, lat, lon, alt = PLACES[1]
    site, plane = S.Site(lat, lon, alt, monthly_tl(lat, lon)), S.Plane(25, 190, 7.5)
    starts = pd.date_range("2025-03-01", "2025-03-08", freq="1h", tz="UTC", inclusive="left")
    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=alt)
    parts = []
    for i in range(12):
        t = starts + pd.Timedelta("5min") * (i + 0.5)
        sp = loc.get_solarposition(t)
        cs = loc.get_clearsky(t, solar_position=sp)
        irr = pvlib.irradiance.get_total_irradiance(25, 190, sp["apparent_zenith"], sp["azimuth"], cs["dni"],
                                                    cs["ghi"], cs["dhi"],
                                                    dni_extra=pvlib.irradiance.get_extra_radiation(t),
                                                    model="haydavies", albedo=0.2)
        iam = pvlib.iam.physical(pvlib.irradiance.aoi(25, 190, sp["apparent_zenith"], sp["azimuth"]))
        poa = ((irr["poa_direct"] * iam).fillna(0) + irr["poa_diffuse"].fillna(0)).to_numpy()
        poa = np.clip(np.where(sp["apparent_elevation"].to_numpy() <= 0, 0, poa), 0, None)
        parts.append(7500 * poa / 1000 * (1 - 0.0035 * (20 + poa * 25 / 800 - 25)))
    ref = np.mean(parts, axis=0)
    ours = S.period_solar(epoch(starts), 3600, 12, site, (plane,), t_air=20.0)["p_cs"]
    day = ref > 100
    assert np.abs(ours[day] / ref[day] - 1).max() < 0.01
    assert abs(ours.sum() / ref.sum() - 1) < 0.001
