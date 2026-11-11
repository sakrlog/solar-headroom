"""Which periods show the panels' full potential, and which only a floor.

A measured PV power is the panels' *potential* only while the inverter is not holding back
("full output", label MPP). Otherwise (battery full and nothing to send it to, export limit,
deliberate curtailment) the measurement is only a lower bound (label LIMITED), which the learner
treats as a floor. NIGHT periods have no meaningful PV; BAD periods are unusable data.

Rules, per period, from the min/max of each sensor inside the period (Home Assistant statistics
give hourly / 5-minute mean, min and max):

* unlimited (grid-tied, exports freely): every daylight period is full output
* battery rule (off-grid / export-limited systems with a battery):
    - the battery helped the whole period (discharging), or
    - it accepted everything: state of charge below `soc_accepts_all` and charging below its limit
  For off-grid systems with occasional grid/generator, only while no external source is present.
* export rule (export-limited): the grid export stayed clearly below the limit (or the house was
  importing) for the whole period
* voltage rule (optional PV-voltage sensor): the string voltage never left the full-output band
  (throttling pushes the operating point toward open-circuit voltage). The band is learned per site.
* a curtailment entity being active anywhere in the period forces LIMITED.

Sign conventions follow Home Assistant's Energy dashboard: battery power + = discharging,
grid power + = importing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MPP, LIMITED, NIGHT, BAD = "mpp", "limited", "night", "bad"
KINDS = ("unlimited", "export_limited", "off_grid")


@dataclass(frozen=True)
class RegimeConfig:
    kind: str = "unlimited"
    export_limit_w: float = 0.0          # export_limited: 0 = zero export
    soc_accepts_all: float = 90.0        # below this state of charge the battery takes all the PV offers
    charge_limit_w: float | None = None  # battery's max charge power (learned when None)
    v_full_max: float | None = None      # upper edge of the full-output PV voltage band (learned when None)
    daylight_w: float = 20.0             # PV above this = daylight
    discharge_w: float = 30.0            # battery discharging at least this the whole period = helping
    import_w: float = 50.0               # importing at least this the whole period = PV could not cover the load
    voltage_min_w: float = 100.0         # voltage rule needs real current to mean anything
    grid_present_v: float = 80.0         # grid voltage above this = grid present (absent often reads 20-40 V)
    gen_present_w: float = 50.0          # generator input carrying power = generator running
    # charging below this share of the learned limit = clearly below it. Conservative on purpose: near
    # full charge the battery tapers (real limit lower), and a throttled hour wrongly labeled "full
    # output" biases the model low, while a lost label only costs a little data.
    charge_margin: float = 0.6

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")


def _get(stats: dict, key: str, n: int) -> np.ndarray:
    v = stats.get(key)
    return np.full(n, np.nan) if v is None else np.asarray(v, dtype=float)


def external_source(stats: dict, cfg: RegimeConfig, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(present anywhere in the period, known absent for the whole period). Without grid/generator
    sensors nothing is known: present=False, absent=True for off-grid systems (no source to detect)."""
    grid_v_max = _get(stats, "grid_v_max", n)
    gen_w_max = _get(stats, "gen_w_max", n)
    has_v, has_gen = ~np.isnan(grid_v_max), ~np.isnan(gen_w_max)
    present = (has_v & (grid_v_max > cfg.grid_present_v)) | (has_gen & (gen_w_max > cfg.gen_present_w))
    absent = ~present & (has_v | has_gen | (cfg.kind == "off_grid"))
    return present, absent


def label(stats: dict, cfg: RegimeConfig, sun_el=None) -> dict[str, np.ndarray]:
    """Label periods. stats: dict of equal-length arrays, any of pv_w (mean, required), pv_w_min, pv_w_max,
    pv_v_max, soc_max, bat_w_min, bat_w_max, grid_w_min, grid_w_max, grid_v_max, gen_w_max, curtail_frac.
    sun_el: optional sun elevation at the period midpoint (flags PV at night as BAD).
    Returns {"regime", "reason", "ext_any"}."""
    pv = np.asarray(stats["pv_w"], dtype=float)
    n = len(pv)
    daylight = pv > cfg.daylight_w
    reason = np.full(n, "", dtype=object)
    full = np.zeros(n, bool)

    if cfg.kind == "unlimited":
        full |= daylight
        reason[daylight] = "unlimited"

    bat_min, soc_max = _get(stats, "bat_w_min", n), _get(stats, "soc_max", n)
    present, absent = external_source(stats, cfg, n)
    if cfg.kind != "unlimited" and not np.isnan(bat_min).all():
        limit = cfg.charge_limit_w if cfg.charge_limit_w is not None else learn_charge_limit(bat_min)
        discharging = bat_min > cfg.discharge_w
        hungry = soc_max < cfg.soc_accepts_all              # NaN (no SOC sensor) -> False
        if limit is not None:
            hungry &= bat_min > -cfg.charge_margin * limit   # charging clearly below its limit
        ok_source = absent if cfg.kind == "off_grid" else np.ones(n, bool)
        rule = daylight & ok_source & (discharging | hungry)
        reason[rule & discharging & ~full] = "discharging"
        reason[rule & ~discharging & ~full] = "battery_hungry"
        full |= rule

    if cfg.kind == "export_limited":
        grid_min = _get(stats, "grid_w_min", n)
        if cfg.export_limit_w > 0:
            margin = max(100.0, 0.1 * cfg.export_limit_w)
            rule = daylight & (grid_min > -(cfg.export_limit_w - margin))
        else:
            rule = daylight & (grid_min > cfg.import_w)
        reason[rule & ~full] = "export_below_limit"
        full |= rule

    v_max = _get(stats, "pv_v_max", n)
    if cfg.v_full_max is not None and not np.isnan(v_max).all():
        rule = daylight & (pv > cfg.voltage_min_w) & (v_max < cfg.v_full_max)
        reason[rule & ~full] = "voltage"
        full |= rule

    curtailed = _get(stats, "curtail_frac", n) > 0
    full &= ~curtailed
    reason[curtailed] = "curtailed"

    regime = np.where(~daylight, NIGHT, np.where(full, MPP, LIMITED)).astype(object)
    bad = frozen(stats, n)
    if sun_el is not None:
        bad |= (np.asarray(sun_el) < -3) & (pv > 50)
    regime[bad] = BAD
    reason[bad | ~daylight] = ""
    return {"regime": regime, "reason": reason, "ext_any": present}


def frozen(stats: dict, n: int) -> np.ndarray:
    """Periods in which every available sensor repeated one value (integration stuck)."""
    spreads = []
    for k in ("pv_w", "pv_v", "bat_w"):
        lo, hi = stats.get(f"{k}_min"), stats.get(f"{k}_max")
        if lo is not None and hi is not None:
            spreads.append(np.nan_to_num(np.asarray(hi, float) - np.asarray(lo, float), nan=0.0))
    if not spreads:
        return np.zeros(n, bool)
    pv = np.asarray(stats["pv_w"], dtype=float)
    return (np.sum(spreads, axis=0) == 0) & (pv > 0)


# ---------------------------------------------------------------- learned thresholds
def learn_charge_limit(bat_w_min, min_periods: int = 50) -> float | None:
    """The battery's practical max charge power: a high quantile (99 %) of the hourly peak charging seen."""
    b = np.asarray(bat_w_min, dtype=float)
    charging = -b[np.isfinite(b) & (b < -50)]
    if len(charging) < min_periods:
        return None
    return float(np.quantile(charging, 0.99))


def learn_voltage_band(pv_v_max, regime, reason, pv_w, soc_max=None, bat_w_min=None, bat_w_max=None,
                       max_leak: float = 0.01, min_coverage: float = 0.25, min_periods: int = 100) -> dict:
    """Upper edge of the full-output PV voltage band.

    Throttling pushes the string toward open-circuit voltage, so periods that were surely throttled
    (battery practically full and idle: it neither charged much nor discharged) sit at high voltage.
    The edge is the voltage that at most `max_leak` of them fell below. It is accepted only if it is
    useful: at least `min_coverage` of the battery-rule full-output periods stayed below it.
    Returns {"v_full_max": float|None, "n_full", "n_throttled", "coverage"}."""
    v, pv = np.asarray(pv_v_max, float), np.asarray(pv_w, float)
    battery = np.isin(np.asarray(reason, dtype=object), ["discharging", "battery_hungry"])
    full = battery & (np.asarray(regime, dtype=object) == MPP) & (pv > 100) & np.isfinite(v)
    out = {"v_full_max": None, "n_full": int(full.sum()), "n_throttled": 0, "coverage": None}
    if soc_max is None or bat_w_min is None or bat_w_max is None or full.sum() < min_periods:
        return out
    soc, bmin, bmax = (np.asarray(a, float) for a in (soc_max, bat_w_min, bat_w_max))
    throttled = (soc >= 98) & (bmin > -150) & (bmax < 30) & (pv > 100) & np.isfinite(v)
    out["n_throttled"] = int(throttled.sum())
    if throttled.sum() < min_periods:
        return out
    edge = float(np.quantile(v[throttled], max_leak))
    coverage = float(np.mean(v[full] < edge))
    out["coverage"] = round(coverage, 3)
    if coverage >= min_coverage:
        out["v_full_max"] = round(edge, 1)
    return out
