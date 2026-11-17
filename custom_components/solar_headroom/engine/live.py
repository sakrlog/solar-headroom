"""The panels' potential right now, from sensor updates (seconds apart), without training.

Which state the system is in comes from the newest readings, with the rules that label the history
(engine.regimes) made instantaneous, and the thresholds training learned for this site:

* unlimited systems: always full output
* battery: helping (discharging) for DISCHARGE_HOLD_S, or taking everything (state of charge below
  `soc_accepts_all`, charging clearly below its learned limit); off-grid only while no grid or
  generator is present
* export-limited: export clearly below the limit, or importing
* PV voltage inside the full-output band for V_WINDOW_S. The band's edge is learned from these live
  windows themselves (VoltageBand), not from the hourly statistics: a 60-second maximum is lower than
  an hourly one, so an hourly edge lets throttled moments through (on a real off-grid site 6-9 % at
  the edges hourly data gave, against ~1 % by construction here)

At full output the potential is the measured PV. Otherwise it is the sky model's clearness (k = power
/ clear-sky power), raised when the sky model was too low at the last full-output moment, the
correction fading with time constant TAU_S, and never below the measured PV:

    k = k_sky(now) * max(1, k_last / k_sky(last)) ** exp(-minutes since / TAU)

Only upward: on an off-grid site the panels are often at full output *because* a cloud dimmed them
(the battery has to help), so a downward correction from such a moment would carry the cloud into the
sunnier stretch that follows. Checked on a real off-grid site, 5-second replay: daily potential -3.9 %
against the training's hindsight reconstruction (two-way correction -11.3 %, sky model alone -5.0 %);
at pretended throttling gaps of 5-75 min the same error as the two-way correction (17 % below the
sky model alone) with a bias of -27 W instead of -169 W.

DayLog adds up, per 5 minutes, the seconds covered and the potential and used energy, which the
solar bank (engine.daily.bank_today) counts instead of the forecast for the past part of today.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from . import regimes as R

FULL, THROTTLED, NIGHT, UNKNOWN = "full_output", "throttled", "night", "unknown"
DISCHARGE_HOLD_S = 20.0          # the battery must keep helping this long (not just a load spike)
V_WINDOW_S = 60.0                # PV voltage inside the band this long
LAST_FULL_MAX_AGE_S = 90 * 60.0  # older full-output moments say nothing about now
TAU_S = 40 * 60.0
RATIO_BOUNDS = (1.0, 4.0)        # only upward (see above)
MIN_K_SKY = 0.05
BUCKET_S = 300
MAX_STEP_S = 120.0               # a longer gap between two updates (restart, stall) is not counted
KEEP_S = 2 * 86400
BAND_SAMPLE_S = 60.0             # one voltage-band sample per minute at most
BAND_KEEP = 3000                 # per class (~ several days of daylight)
BAND_MIN = 200
BAND_MAX_LEAK = 0.02             # share of the full-and-idle windows allowed below the edge (some of
                                 # them are full output: the load took all the panels gave)
BAND_MIN_COVERAGE = 0.25         # the edge must find at least this share of full-output windows


@dataclass
class VoltageBand:
    """Upper edge of the full-output PV voltage band for live windows. Samples: the window's highest
    voltage while the battery proves full output (helping, or taking everything), and while the battery
    is full and idle (surely held back: throttling pushes the string toward open circuit). The edge is
    the voltage that at most BAND_MAX_LEAK of the held-back windows fell below (cf. regimes.learn_voltage_band)."""
    full: deque = field(default_factory=lambda: deque(maxlen=BAND_KEEP))
    held: deque = field(default_factory=lambda: deque(maxlen=BAND_KEEP))
    edge: float | None = None
    _new: int = 0

    def add(self, kind: str, v_max: float) -> None:
        (self.full if kind == "full" else self.held).append(round(v_max, 1))
        self._new += 1
        if self._new >= 10:
            self._new = 0
            self.edge = self.learn()

    def learn(self) -> float | None:
        if len(self.full) < BAND_MIN or len(self.held) < BAND_MIN:
            return None
        edge = float(np.quantile(np.asarray(self.held), BAND_MAX_LEAK))
        coverage = float(np.mean(np.asarray(self.full) < edge))
        return round(edge, 1) if coverage >= BAND_MIN_COVERAGE else None

    def to_dict(self) -> dict:
        return {"full": list(self.full), "held": list(self.held)}

    @classmethod
    def from_dict(cls, d: Mapping | None) -> VoltageBand:
        b = cls()
        b.full.extend(float(x) for x in (d or {}).get("full") or ())
        b.held.extend(float(x) for x in (d or {}).get("held") or ())
        b.edge = b.learn()
        return b


@dataclass
class LiveEstimator:
    cfg: R.RegimeConfig
    values: dict[str, float] = field(default_factory=dict)
    disch_since: float | None = None
    v_hist: deque = field(default_factory=deque)            # (ts, pv_v)
    last_full: tuple[float, float, float] | None = None     # (ts, k, k_sky)
    band: VoltageBand = field(default_factory=VoltageBand)
    _band_at: float = -math.inf

    @classmethod
    def for_system(cls, kind: str, export_limit_w: float, learned: Mapping | None) -> LiveEstimator:
        """learned: thresholds from training (the charge limit; the hourly voltage edge is not used live)."""
        learned = learned or {}
        return cls(R.RegimeConfig(kind=kind, export_limit_w=export_limit_w,
                                  charge_limit_w=learned.get("charge_limit_w")))

    # ------------------------------------------------------------ inputs
    def update(self, role: str, value: float | None, ts: float) -> None:
        """A new reading (None = unavailable). Roles as in engine.history: pv_w, bat_w (+ = discharging),
        soc, grid_w (+ = importing), grid_v, gen_w, pv_v."""
        if value is None or not math.isfinite(value):
            self.values.pop(role, None)
            if role == "bat_w":
                self.disch_since = None
            return
        self.values[role] = float(value)
        if role == "bat_w":
            if value > self.cfg.discharge_w:
                self.disch_since = ts if self.disch_since is None else self.disch_since
            else:
                self.disch_since = None
        elif role == "pv_v":
            self.v_hist.append((ts, float(value)))
            while len(self.v_hist) >= 2 and self.v_hist[1][0] <= ts - V_WINDOW_S:
                self.v_hist.popleft()

    def _v_max(self, now: float) -> float | None:
        """Highest PV voltage in effect during the last V_WINDOW_S: the reading in effect at the window's
        start and the ones after it (None if the window is not fully known)."""
        start = now - V_WINDOW_S
        if not self.v_hist or self.v_hist[0][0] > start or "pv_v" not in self.values:
            return None
        at_start = max(i for i, (t, _) in enumerate(self.v_hist) if t <= start)
        return max(v for _, v in list(self.v_hist)[at_start:])

    # ------------------------------------------------------------ state
    def regime(self, now: float) -> tuple[str, str]:
        """(FULL | THROTTLED | UNKNOWN, reason)."""
        c, v = self.cfg, self.values
        pv = v.get("pv_w")
        if pv is None:
            return UNKNOWN, "no_pv_reading"
        if c.kind == "unlimited":
            return FULL, "unlimited"
        external = v.get("grid_v", 0.0) > c.grid_present_v or v.get("gen_w", 0.0) > c.gen_present_w
        bat = v.get("bat_w")
        if bat is not None and not (c.kind == "off_grid" and external):
            if self.disch_since is not None and now - self.disch_since >= DISCHARGE_HOLD_S:
                return FULL, "discharging"
            limit = c.charge_limit_w
            soc = v.get("soc")
            if soc is not None and soc < c.soc_accepts_all and bat <= c.discharge_w \
                    and (limit is None or bat > -c.charge_margin * limit):
                return FULL, "battery_hungry"
        grid = v.get("grid_w")
        if c.kind == "export_limited" and grid is not None:
            if c.export_limit_w > 0:
                if grid > -(c.export_limit_w - max(100.0, 0.1 * c.export_limit_w)):
                    return FULL, "export_below_limit"
            elif grid > c.import_w:
                return FULL, "importing"
        edge = self.band.edge if c.v_full_max is None else c.v_full_max
        if edge is not None and pv > c.voltage_min_w:
            v_max = self._v_max(now)
            if v_max is not None and v_max < edge:
                return FULL, "voltage"
        return THROTTLED, "external_source" if external else "held_back"

    def _sample_band(self, now: float, regime: str, reason: str) -> None:
        v = self.values
        if now - self._band_at < BAND_SAMPLE_S or v.get("pv_w", 0.0) <= self.cfg.voltage_min_w:
            return
        v_max = self._v_max(now)
        if v_max is None:
            return
        bat, soc = v.get("bat_w"), v.get("soc")
        if reason in ("discharging", "battery_hungry"):
            kind = "full"
        elif reason in ("held_back", "voltage", "external_source") and bat is not None and soc is not None \
                and soc >= 99.0 and abs(bat) < 100.0:      # "voltage" too: else the edge would only creep up
            kind = "held"
        else:
            return
        self._band_at = now
        self.band.add(kind, v_max)

    def estimate(self, now: float, p_cs: float, k_sky: float | None) -> dict:
        """Potential now (W). p_cs: clear-sky power of the panels now; k_sky: the sky model's clearness for
        now (None without a model). basis: night | measured | estimated | sky_model | unknown."""
        regime, reason = self.regime(now)
        if "pv_v" in self.values:
            self._sample_band(now, regime, reason)
        pv = max(0.0, self.values.get("pv_w", 0.0))
        out = {"regime": regime, "reason": reason, "pv_w": pv, "clear_sky_w": p_cs}
        if p_cs < 1.0:
            return out | {"regime": NIGHT, "w": pv, "untapped_w": 0.0, "basis": "night"}
        if regime == FULL:
            self.last_full = (now, pv / p_cs, k_sky if k_sky is not None else math.nan)
            return out | {"w": pv, "untapped_w": 0.0, "basis": "measured"}
        k, basis, minutes = None, "sky_model", None
        lf = self.last_full
        if lf is not None and 0 <= now - lf[0] <= LAST_FULL_MAX_AGE_S:
            minutes = (now - lf[0]) / 60.0
            if k_sky is not None and math.isfinite(lf[2]) and lf[2] > MIN_K_SKY:
                ratio = min(max(lf[1] / lf[2], RATIO_BOUNDS[0]), RATIO_BOUNDS[1])
                k = k_sky * ratio ** math.exp(-(now - lf[0]) / TAU_S)
            elif k_sky is None:
                k = lf[1]
            basis = "estimated"
        if k is None:
            k = k_sky
        if k is None:
            return out | {"w": pv if regime != UNKNOWN else None, "untapped_w": None, "basis": "unknown"}
        w = max(pv, k * p_cs) if regime != UNKNOWN else k * p_cs
        return out | {"w": w, "untapped_w": max(0.0, w - pv) if regime != UNKNOWN else None, "basis": basis,
                      "minutes_since_full": None if minutes is None else round(minutes, 1)}

    def to_dict(self) -> dict:
        return {"last_full": list(self.last_full) if self.last_full else None, "band": self.band.to_dict()}

    def restore(self, d: Mapping | None) -> None:
        lf = (d or {}).get("last_full")
        self.last_full = tuple(float(x) for x in lf) if lf else None
        self.band = VoltageBand.from_dict((d or {}).get("band"))


@dataclass
class DayLog:
    """Per 5-minute bucket: [covered seconds, potential Wh, used Wh]."""
    buckets: dict[int, list[float]] = field(default_factory=dict)
    last: tuple[float, float, float] | None = None           # (ts, potential W, pv W) of the previous step

    def step(self, now: float, potential_w: float | None, pv_w: float | None) -> None:
        """Count the time since the previous step with the previous step's values (they held until now)."""
        prev, self.last = self.last, (None if potential_w is None or pv_w is None else (now, potential_w, pv_w))
        if prev is None or not 0 < now - prev[0] <= MAX_STEP_S:
            return
        t = prev[0]
        while t < now:
            b = int(t // BUCKET_S * BUCKET_S)
            t1 = min(now, b + BUCKET_S)
            acc = self.buckets.setdefault(b, [0.0, 0.0, 0.0])
            dt = t1 - t
            acc[0] += dt
            acc[1] += prev[1] * dt / 3600.0
            acc[2] += min(prev[2], prev[1]) * dt / 3600.0
            t = t1
        for old in [b for b in self.buckets if b < now - KEEP_S]:
            del self.buckets[old]

    def known(self, hour_starts) -> dict[str, np.ndarray]:
        """Per hour of hour_starts (sorted): covered_s, potential_wh, used_wh (engine.daily.bank_today)."""
        hs = np.asarray(hour_starts, float)
        out = {k: np.zeros(len(hs)) for k in ("covered_s", "potential_wh", "used_wh")}
        if not len(hs) or not self.buckets:
            return out
        b = np.array(sorted(self.buckets), float)
        vals = np.array([self.buckets[int(x)] for x in b])
        i = np.searchsorted(hs, b, side="right") - 1
        ok = (i >= 0) & (b < hs[np.maximum(i, 0)] + 3600)
        for j, k in enumerate(("covered_s", "potential_wh", "used_wh")):
            out[k] = np.bincount(i[ok], weights=vals[ok, j], minlength=len(hs))[: len(hs)]
        return out

    def wasted_kwh(self, t0: float, t1: float) -> float:
        return sum(max(0.0, v[1] - v[2]) for b, v in self.buckets.items() if t0 <= b < t1) / 1000.0

    def to_dict(self) -> dict:
        return {"buckets": {str(b): v for b, v in self.buckets.items()}}

    @classmethod
    def from_dict(cls, d: Mapping | None) -> DayLog:
        return cls({int(b): [float(x) for x in v] for b, v in ((d or {}).get("buckets") or {}).items()})


def merge_known(kind: str, live: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray] | None) -> dict:
    """Per hour, what bank_today counts as known: the live log; for systems that never hold back also the
    5-minute statistics where they cover more (before setup, during restarts). For the others the
    statistics only give a floor, so uncovered time counts with the forecast instead."""
    if not stats or kind != "unlimited":
        return dict(live)
    use_stats = np.asarray(stats["covered_s"]) > np.asarray(live["covered_s"])
    return {k: np.where(use_stats, stats[k], live[k]) for k in ("covered_s", "potential_wh", "used_wh")}


__all__ = ["FULL", "NIGHT", "THROTTLED", "UNKNOWN", "DayLog", "LiveEstimator", "VoltageBand", "merge_known"]
