"""Sun position, clear-sky irradiance and plane-of-array physics, numpy only.

These are ports of the pvlib functions the model was developed with, kept numerically
equivalent (tests/test_solar.py checks them against pvlib):

* sun position: NOAA / Meeus low-precision algorithm (pvlib's default is NREL SPA; the two
  agree to ~0.01 deg) + SPA's refraction formula for the apparent elevation
* extraterrestrial irradiance: Spencer (1971), solar constant 1366.1 W/m2
* airmass: Kasten & Young (1989) on the apparent zenith, scaled by site pressure
* clear sky: Ineichen & Perez with monthly Linke turbidity, interpolated to the day of year
* plane of array: Hay & Davies sky diffuse + ground reflection, physical IAM on the beam
* temperature: NOCT-style cell temperature with a linear power coefficient

Times are UTC epoch seconds (float arrays); angles are degrees; azimuth is clockwise from
north (180 = south). No pandas, no timezone handling: local clock time is never needed
(features use solar time).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

RAD = np.pi / 180.0
SOLAR_CONSTANT = 1366.1
REFRACTION_AT_HORIZON = 0.5667      # deg, SPA default
SUN_RADIUS = 0.26667                 # deg
DEFAULT_ALBEDO = 0.2
TEMP_COEFF = -0.0035                 # power change per deg C of cell temperature (crystalline Si)


@dataclass(frozen=True)
class Site:
    latitude: float
    longitude: float
    altitude: float = 0.0
    # 12 monthly Linke turbidity values (Jan..Dec); None -> DEFAULT_TL
    linke_turbidity: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Plane:
    tilt: float                      # deg from horizontal
    azimuth: float                   # deg clockwise from north (180 = south)
    kwp: float                       # effective DC size, kW at 1000 W/m2 and 25 C


DEFAULT_TL = (3.0,) * 12             # rural mid-latitude fallback when no climatology is given


# ---------------------------------------------------------------- calendar helpers
def _as_ts(ts) -> np.ndarray:
    return np.atleast_1d(np.asarray(ts, dtype="float64"))


def day_of_year(ts) -> tuple[np.ndarray, np.ndarray]:
    """UTC day of year (1..366) and leap-year flag for epoch seconds."""
    days = np.floor(_as_ts(ts) / 86400.0).astype("int64").astype("datetime64[D]")
    years = days.astype("datetime64[Y]")
    doy = (days - years.astype("datetime64[D]")).astype("int64") + 1
    y = years.astype("int64") + 1970
    leap = ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)
    return doy, leap


def utc_month(ts) -> np.ndarray:
    """UTC calendar month (1..12)."""
    m = np.floor(_as_ts(ts) / 86400.0).astype("int64").astype("datetime64[D]").astype("datetime64[M]")
    return (m.astype("int64") % 12) + 1


# ---------------------------------------------------------------- atmosphere
def alt2pres(altitude: float) -> float:
    """Standard-atmosphere pressure (Pa) at an altitude (m)."""
    return 100.0 * ((44331.514 - altitude) / 11880.516) ** (1 / 0.1902632)


def relative_airmass(apparent_zenith) -> np.ndarray:
    """Kasten & Young (1989); NaN below the horizon."""
    z = np.where(apparent_zenith > 90, np.nan, apparent_zenith)
    return 1.0 / (np.cos(z * RAD) + 0.50572 * ((6.07995 + (90 - z)) ** -1.6364))


def extra_radiation(ts) -> np.ndarray:
    """Extraterrestrial normal irradiance (W/m2), Spencer's Earth-Sun distance correction."""
    doy, _ = day_of_year(ts)
    b = 2.0 * np.pi / 365.0 * (doy - 1)
    r = (1.00011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b)
         + 0.000719 * np.cos(2 * b) + 7.7e-05 * np.sin(2 * b))
    return SOLAR_CONSTANT * r


# ---------------------------------------------------------------- sun position
def sun_position(ts, latitude: float, longitude: float, altitude: float = 0.0,
                 temperature: float = 12.0) -> dict[str, np.ndarray]:
    """Sun position for UTC epoch seconds.

    Returns dict of arrays: elevation, apparent_elevation, zenith, apparent_zenith, azimuth,
    solar_time (true solar time in hours, 0..24, 12 = solar noon).
    """
    ts = _as_ts(ts)
    jc = (ts / 86400.0 + 2440587.5 - 2451545.0) / 36525.0           # Julian centuries since J2000
    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    m = (357.52911 + jc * (35999.05029 - 0.0001537 * jc)) * RAD
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    c = (np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(2 * m) * (0.019993 - 0.000101 * jc) + np.sin(3 * m) * 0.000289)
    omega = (125.04 - 1934.136 * jc) * RAD
    lam = (l0 + c - 0.00569 - 0.00478 * np.sin(omega)) * RAD           # apparent longitude
    eps0 = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    eps = (eps0 + 0.00256 * np.cos(omega)) * RAD                        # corrected obliquity
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    y = np.tan(eps / 2) ** 2
    l0r = l0 * RAD
    eot = 4.0 / RAD * (y * np.sin(2 * l0r) - 2 * e * np.sin(m) + 4 * e * y * np.sin(m) * np.cos(2 * l0r)
                       - 0.5 * y * y * np.sin(4 * l0r) - 1.25 * e * e * np.sin(2 * m))   # minutes
    tst = np.mod(np.mod(ts, 86400.0) / 60.0 + eot + 4.0 * longitude, 1440.0)            # minutes
    ha = (tst / 4.0 - 180.0) * RAD                                      # hour angle
    lat = latitude * RAD
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    zenith = np.arccos(np.clip(cos_z, -1.0, 1.0)) / RAD
    azimuth = np.mod(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(lat) - np.tan(decl) * np.cos(lat)) / RAD + 180.0,
                     360.0)
    e0 = 90.0 - zenith
    # SPA refraction (pressure in hPa); only where the sun's upper limb can be above the horizon
    pres_hpa = alt2pres(altitude) / 100.0
    up = e0 >= -(SUN_RADIUS + REFRACTION_AT_HORIZON)
    safe = np.where(up, e0, 45.0)
    de = (pres_hpa / 1010.0) * (283.0 / (273.0 + temperature)) * 1.02 / (
        60.0 * np.tan((safe + 10.3 / (safe + 5.11)) * RAD))
    app_el = e0 + np.where(up, de, 0.0)
    return {"elevation": e0, "apparent_elevation": app_el, "zenith": zenith,
            "apparent_zenith": 90.0 - app_el, "azimuth": azimuth, "solar_time": tst / 60.0}


# ---------------------------------------------------------------- clear sky
def _month_middles(leap: bool) -> np.ndarray:
    mdays = np.array([31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    ydays = 366.0 if leap else 365.0
    return np.concatenate([[-31 / 2.0], np.cumsum(mdays) - mdays / 2.0, [ydays + 31 / 2.0]])


def linke_turbidity(ts, monthly: tuple[float, ...] | None) -> np.ndarray:
    """Daily Linke turbidity from 12 monthly values (each taken at mid-month), like pvlib."""
    lts = np.asarray(monthly if monthly is not None else DEFAULT_TL, dtype=float)
    lts = np.concatenate([[lts[-1]], lts, [lts[0]]])
    doy, leap = day_of_year(ts)
    return np.where(leap, np.interp(doy, _month_middles(True), lts), np.interp(doy, _month_middles(False), lts))


def clearsky_ineichen(apparent_zenith, airmass_absolute, tl, altitude: float, dni_extra) -> dict[str, np.ndarray]:
    """Ineichen & Perez clear-sky GHI/DNI/DHI (W/m2); NaN when the sun is below the horizon."""
    cos_z = np.maximum(np.cos(apparent_zenith * RAD), 0.0)
    fh1, fh2 = np.exp(-altitude / 8000.0), np.exp(-altitude / 1250.0)
    cg1, cg2 = 5.09e-05 * altitude + 0.868, 3.92e-05 * altitude + 0.0387
    ghi = np.exp(-cg2 * airmass_absolute * (fh1 + fh2 * (tl - 1)))
    ghi = cg1 * dni_extra * cos_z * np.fmax(ghi, 0)
    b = 0.664 + 0.163 / fh1
    bnci = dni_extra * np.fmax(b * np.exp(-0.09 * airmass_absolute * (tl - 1)), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bnci_2 = (1 - (0.1 - 0.2 * np.exp(-tl)) / (0.1 + 0.882 / fh1)) / cos_z
    bnci_2 = ghi * np.fmin(np.fmax(bnci_2, 0), 1e20)
    dni = np.minimum(bnci, bnci_2)
    return {"ghi": ghi, "dni": dni, "dhi": ghi - dni * cos_z}


def solar(ts, site: Site, temperature: float = 12.0) -> dict[str, np.ndarray]:
    """Sun position + clear-sky irradiance + extraterrestrial irradiance for UTC epoch seconds."""
    sp = sun_position(ts, site.latitude, site.longitude, site.altitude, temperature)
    dni_extra = extra_radiation(ts)
    am = relative_airmass(sp["apparent_zenith"]) * alt2pres(site.altitude) / 101325.0
    cs = clearsky_ineichen(sp["apparent_zenith"], am, linke_turbidity(ts, site.linke_turbidity), site.altitude,
                           dni_extra)
    return {"sun_el": sp["apparent_elevation"], "sun_az": sp["azimuth"], "zenith": sp["apparent_zenith"],
            "solar_time": sp["solar_time"], "ghi_cs": cs["ghi"], "dni_cs": cs["dni"], "dhi_cs": cs["dhi"],
            "dni_extra": dni_extra}


# ---------------------------------------------------------------- plane of array
def aoi_projection(tilt, plane_azimuth, zenith, azimuth) -> np.ndarray:
    p = (np.cos(tilt * RAD) * np.cos(zenith * RAD)
         + np.sin(tilt * RAD) * np.sin(zenith * RAD) * np.cos((azimuth - plane_azimuth) * RAD))
    return np.clip(p, -1.0, 1.0)


def iam_physical(aoi, n: float = 1.526, k: float = 4.0, glass_l: float = 0.002) -> np.ndarray:
    """Fresnel + absorption incidence-angle modifier (no anti-reflective coating)."""
    cos1 = np.maximum(0.0, np.cos(aoi * RAD))
    sin1 = np.sqrt(1 - cos1 ** 2)
    sin2 = sin1 / n
    cos2 = np.sqrt(1 - sin2 ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = ((cos1 - n * cos2) / (cos1 + n * cos2)) ** 2
        rp = ((cos2 - n * cos1) / (cos2 + n * cos1)) ** 2
        att = np.exp(-k * glass_l / cos2)
    tau0 = (1 - ((1 - n) / (1 + n)) ** 2) * np.exp(-k * glass_l)
    return ((1 - rs) * att + (1 - rp) * att) / 2 / tau0


def poa_irradiance(tilt: float, plane_azimuth: float, zenith, azimuth, dni, ghi, dhi, dni_extra,
                   albedo: float = DEFAULT_ALBEDO, iam: bool = True) -> dict[str, np.ndarray]:
    """Plane-of-array irradiance (W/m2): Hay & Davies sky diffuse + ground reflection.
    `beam` includes the IAM loss when iam=True. NaN inputs give NaN components."""
    proj = aoi_projection(tilt, plane_azimuth, zenith, azimuth)
    rb = np.maximum(proj, 0) / np.maximum(np.cos(zenith * RAD), 0.01745)
    with np.errstate(divide="ignore", invalid="ignore"):
        ai = dni / dni_extra
    sky = (np.maximum(dhi * (1 - ai) * 0.5 * (1 + np.cos(tilt * RAD)), 0)
           + np.maximum(dhi * ai * rb, 0))
    ground = ghi * albedo * (1 - np.cos(tilt * RAD)) * 0.5
    aoi = np.arccos(proj) / RAD
    beam = np.maximum(dni * np.cos(aoi * RAD), 0)
    if iam:
        beam = beam * iam_physical(aoi)
    return {"beam": beam, "diffuse": sky + ground, "aoi": aoi}


def poa_clearsky(sol: dict[str, np.ndarray], plane: Plane) -> np.ndarray:
    """Clear-sky irradiance on the panel plane (W/m2) incl. IAM; 0 at night."""
    p = poa_irradiance(plane.tilt, plane.azimuth, sol["zenith"], sol["sun_az"], sol["dni_cs"], sol["ghi_cs"],
                       sol["dhi_cs"], sol["dni_extra"])
    out = np.nan_to_num(p["beam"], nan=0.0) + np.nan_to_num(p["diffuse"], nan=0.0)
    return np.clip(np.where(sol["sun_el"] <= 0, 0.0, out), 0, None)


def temp_derate(poa, t_air, coeff: float = TEMP_COEFF) -> np.ndarray:
    """Power factor from cell temperature (cell = air + 25 C per 800 W/m2)."""
    return 1.0 + coeff * (np.asarray(t_air, dtype=float) + np.asarray(poa) * 25.0 / 800.0 - 25.0)


def clearsky_power(sol: dict[str, np.ndarray], planes: list[Plane] | tuple[Plane, ...], t_air) -> np.ndarray:
    """Expected DC power (W) under a clear sky, summed over panel groups, temperature-corrected."""
    total = np.zeros_like(sol["sun_el"])
    for pl in planes:
        poa = poa_clearsky(sol, pl)
        total = total + pl.kwp * poa * temp_derate(poa, t_air)
    return total


def period_solar(starts, period_s: float, n_sub: int, site: Site, planes=(), t_air=None,
                 temperature: float = 12.0) -> dict[str, np.ndarray]:
    """Solar features averaged over periods [start, start + period) from n_sub midpoint samples.

    t_air: None, a scalar, or a callable(ts) -> array of air temperature (C) at the sample times.
    Means skip NaN (clear-sky irradiance is NaN with the sun below the horizon), like pandas.
    Adds sun_el_max and, with planes, poa_cs / p_cs / p_cs_max.
    """
    starts = _as_ts(starts)
    step = period_s / n_sub
    acc: dict[str, list[np.ndarray]] = {}
    for i in range(n_sub):
        t = starts + step * (i + 0.5)
        s = solar(t, site, temperature)
        if planes:
            ta = t_air(t) if callable(t_air) else (np.full(len(t), 20.0 if t_air is None else float(t_air)))
            s["poa_cs"] = sum(poa_clearsky(s, pl) * pl.kwp for pl in planes) / max(sum(pl.kwp for pl in planes), 1e-9)
            s["p_cs"] = clearsky_power(s, planes, ta)
        for k, v in s.items():
            acc.setdefault(k, []).append(v)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # "mean of empty slice" on all-night periods
        for k, vs in acc.items():
            out[k] = np.nanmean(np.vstack(vs), axis=0)
    out["sun_el_max"] = np.max(np.vstack(acc["sun_el"]), axis=0)
    if planes:
        out["p_cs_max"] = np.max(np.vstack(acc["p_cs"]), axis=0)
    return out
