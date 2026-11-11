"""Training targets, fitting and scoring of the potential models.

Every model learns k = PV power / clear-sky power of the site's panels (a "cloudiness factor"),
never watts: physics carries the sun, season and system size, the learner only the correction.
Rows at full output give k exactly; throttled rows give a floor (censored loss, see engine.gbm).

Feature sets
  SKY_NOWCAST  hourly, real time: sun/season + satellite of the last published hour + forecast
  SKY_BASIC    sun/season only (fallback while Open-Meteo is unreachable)
  HINDSIGHT    satellite of the period itself + forecast: fills throttled hours afterwards (daily target)

Leak guard (throttled systems)
  Full-output hours are selected on low output: an hour only runs at full output if the house and
  battery can take everything, so at high sun mostly the darker hours do. Time-of-day and sun-geometry
  features (CLOCK) let the learner turn "midday" into "darker than the weather says" and under-estimate
  the throttled hours; the floors only partly correct that. A censored fit with satellite or forecast
  inputs and enough throttled high-sun rows therefore also trains a weather-only model. If the full
  model is clearly lower on those training rows, its predictions blend toward the weather-only model
  (research/README.md).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .features import FC, SAT_LAG, SAT_NOW, SKY, matrix
from .gbm import GBM, GBMParams
from .regimes import LIMITED, MPP

K_MAX = 1.3
SKY_NOWCAST = SKY + SAT_LAG + FC
SKY_BASIC = list(SKY)
HINDSIGHT = SKY + SAT_NOW + FC

CLOCK = ("sun_el", "sun_el_max", "sun_az", "hour", "p_cs", "poa_cs")   # time of day / sun geometry
GUARD_SPAN = 0.10        # full model this much lower than the weather-only one -> weather-only
GUARD_MIN_ROWS = 100     # throttled high-sun training rows needed to measure the leak


@dataclass(frozen=True)
class ModelConfig:
    min_p_cs_w: float            # rows with less clear-sky power are too noisy to learn from (dawn/dusk)
    p_ref_w: float               # row weight (p_cs / p_ref)^2: squared error in watts, scaled to the system
    params: GBMParams = field(default_factory=GBMParams)
    high_sun_w: float = 0.0      # leak guard: compares the models on throttled rows with this much clear sky

    @classmethod
    def for_kwp(cls, kwp: float, params: GBMParams | None = None) -> ModelConfig:
        """Defaults scaled to the system size (an 8 kWp system: 320 W / 1000 W / 4000 W)."""
        return cls(min_p_cs_w=40.0 * kwp, p_ref_w=125.0 * kwp, params=params or GBMParams(),
                   high_sun_w=500.0 * kwp)


@dataclass
class Guarded:
    """A censored model under the leak guard: (1 - weight) x full model + weight x weather-only model."""
    features: list[str]
    weight: float                # share of the weather-only model, 0..1
    ratio: float                 # full / weather-only estimate on the throttled high-sun training rows
    full: GBM | None = None      # kept while weight < 1
    weather: GBM | None = None   # kept while weight > 0

    def predict(self, X) -> np.ndarray:
        """k for X in the order of self.features (each part clipped to 0..K_MAX before blending)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros(len(X))
        if self.full is not None:
            out += (1.0 - self.weight) * np.clip(self.full.predict(X), 0.0, K_MAX)
        if self.weather is not None:
            cols = [self.features.index(f) for f in self.weather.features]
            out += self.weight * np.clip(self.weather.predict(X[:, cols]), 0.0, K_MAX)
        return out

    def to_dict(self) -> dict:
        return {"kind": "guarded", "features": self.features, "weight": self.weight, "ratio": self.ratio,
                "full": None if self.full is None else self.full.to_dict(),
                "weather": None if self.weather is None else self.weather.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> Guarded:
        return cls(features=list(d["features"]), weight=float(d["weight"]), ratio=float(d["ratio"]),
                   full=None if d["full"] is None else GBM.from_dict(d["full"]),
                   weather=None if d["weather"] is None else GBM.from_dict(d["weather"]))


def load_model(d: dict) -> GBM | Guarded:
    """A model saved with to_dict(), plain or guarded."""
    return Guarded.from_dict(d) if d.get("kind") == "guarded" else GBM.from_dict(d)


def guard_weight(ratio: float) -> float:
    """Share of the weather-only model: 0 while the full model is not lower, 1 from GUARD_SPAN lower."""
    return float(np.clip((1.0 - ratio) / GUARD_SPAN, 0.0, 1.0))


def targets(table: Mapping[str, np.ndarray], cfg: ModelConfig) -> dict[str, np.ndarray]:
    """k_obs, censoring flag, weight and which rows can be trained on, from pv_w / p_cs / regime."""
    pv, p_cs = np.asarray(table["pv_w"], float), np.asarray(table["p_cs"], float)
    regime = np.asarray(table["regime"], dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_obs = np.where(p_cs > 1, pv / p_cs, np.nan)
    trainable = np.isin(regime, [MPP, LIMITED]) & (p_cs >= cfg.min_p_cs_w) & ~np.isnan(pv)
    return {"k_obs": k_obs, "cens": regime == LIMITED, "w": (p_cs / cfg.p_ref_w) ** 2, "trainable": trainable}


def _fit_gbm(table: Mapping[str, np.ndarray], rows: np.ndarray, features: Sequence[str], cfg: ModelConfig,
             target: str, censored: bool) -> GBM:
    y = np.asarray(table[target], float)[rows]
    cens = np.asarray(table["cens"], bool)[rows] if censored else np.zeros(len(y), bool)
    w = np.asarray(table["w"], float)[rows]
    X = matrix({k: np.asarray(v)[rows] for k, v in table.items() if k in features or k == "ts"}, features)
    return GBM(cfg.params).fit(X, y, cens=cens, weight=w, features=list(features))


def fit(table: Mapping[str, np.ndarray], rows, features: Sequence[str], cfg: ModelConfig,
        target: str = "k_obs", censored: bool = True, guard: bool = True) -> GBM | Guarded:
    """Fit on table rows (bool mask or indices). censored=False treats every target as exact.
    guard: apply the leak guard where it can matter (censored fit, CLOCK features next to satellite or
    forecast inputs, at least GUARD_MIN_ROWS throttled high-sun rows); otherwise the plain model is
    returned. A sun/season-only model has no weather to fall back on: its time-of-day features carry
    real signal (morning haze, afternoon clouds, shading), so it is never guarded."""
    rows = np.asarray(rows)
    full = _fit_gbm(table, rows, features, cfg, target, censored)
    weather = [f for f in features if f not in CLOCK]
    has_weather = any(f not in SKY for f in features)
    if not (guard and censored and has_weather) or len(weather) == len(features):
        return full
    in_rows = np.zeros(len(table["ts"]), bool)
    in_rows[rows] = True
    p_cs = np.asarray(table["p_cs"], float)
    sunny = in_rows & np.asarray(table["cens"], bool) & (p_cs >= cfg.high_sun_w)
    if sunny.sum() < GUARD_MIN_ROWS:
        return full
    weather_model = _fit_gbm(table, rows, weather, cfg, target, censored)
    est_full = float(np.sum(predict_k(full, table, sunny) * p_cs[sunny]))
    est_weather = float(np.sum(predict_k(weather_model, table, sunny) * p_cs[sunny]))
    ratio = est_full / est_weather if est_weather > 0 else 1.0
    weight = guard_weight(ratio)
    return Guarded(list(features), weight, ratio, full if weight < 1 else None,
                   weather_model if weight > 0 else None)


def predict_k(model: GBM | Guarded, table: Mapping[str, np.ndarray], rows=None) -> np.ndarray:
    t = table if rows is None else {k: np.asarray(v)[rows] for k, v in table.items()}
    return np.clip(model.predict(matrix(t, model.features)), 0.0, K_MAX)


# ---------------------------------------------------------------- evaluation helpers
def month_ids(ts) -> np.ndarray:
    """UTC calendar month of epoch seconds as months since 1970-01."""
    d = np.floor(np.asarray(ts, float) / 86400.0).astype("int64").astype("datetime64[D]")
    return d.astype("datetime64[M]").astype("int64")


def group_folds(groups, n_splits: int = 6) -> list[np.ndarray]:
    """Test masks of group-blocked folds: largest groups first, each to the lightest fold
    (the GroupKFold rule; months as groups keep weather spells inside one fold)."""
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    sizes = np.bincount(inv)
    order = np.argsort(sizes, kind="stable")[::-1]
    load = np.zeros(n_splits)
    to_fold = np.zeros(len(sizes), dtype=np.int64)
    for g in order:
        f = int(np.argmin(load))
        load[f] += sizes[g]
        to_fold[g] = f
    fold = to_fold[inv]
    return [fold == f for f in range(n_splits)]


def metrics(pred_w, pv_w, cens, reason=None, floor_slack_w: float = 150.0) -> dict:
    """Errors on full-output rows (overall and by labelling rule) + share of throttled rows predicted
    clearly below their known floor."""
    pred_w, pv = np.asarray(pred_w, float), np.asarray(pv_w, float)
    cens = np.asarray(cens, bool)
    ex = ~cens
    groups = [("all_full_output", ex)]
    if reason is not None:
        reason = np.asarray(reason, dtype=object)
        groups += [("battery_rule", ex & np.isin(reason, ["discharging", "battery_hungry"])),
                   ("voltage_rule", ex & (reason == "voltage"))]
    out = {}
    for name, m in groups:
        if m.any():
            e = pred_w[m] - pv[m]
            out[f"{name}_n"] = int(m.sum())
            out[f"{name}_mae_w"] = round(float(np.mean(np.abs(e))), 1)
            out[f"{name}_bias_w"] = round(float(np.mean(e)), 1)
    out["throttled_n"] = int(cens.sum())
    if cens.any():
        out["pct_below_known_floor"] = round(float(100 * np.mean(pred_w[cens] < pv[cens] - floor_slack_w)), 1)
    return out
