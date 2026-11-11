"""Training targets, fitting and scoring of the potential models.

Every model learns k = PV power / clear-sky power of the site's panels (a "cloudiness factor"),
never watts: physics carries the sun, season and system size, the learner only the correction.
Rows at full output give k exactly; throttled rows give a floor (censored loss, see engine.gbm).

Feature sets
  SKY_NOWCAST  hourly, real time: sun/season + satellite of the last published hour + forecast
  SKY_BASIC    sun/season only (fallback while Open-Meteo is unreachable)
  HINDSIGHT    satellite of the period itself + forecast: fills throttled hours afterwards (daily target)
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


@dataclass(frozen=True)
class ModelConfig:
    min_p_cs_w: float            # rows with less clear-sky power are too noisy to learn from (dawn/dusk)
    p_ref_w: float               # row weight (p_cs / p_ref)^2: squared error in watts, scaled to the system
    params: GBMParams = field(default_factory=GBMParams)

    @classmethod
    def for_kwp(cls, kwp: float, params: GBMParams | None = None) -> ModelConfig:
        """Defaults scaled to the system size (an 8 kWp system: 320 W / 1000 W)."""
        return cls(min_p_cs_w=40.0 * kwp, p_ref_w=125.0 * kwp, params=params or GBMParams())


def targets(table: Mapping[str, np.ndarray], cfg: ModelConfig) -> dict[str, np.ndarray]:
    """k_obs, censoring flag, weight and which rows can be trained on, from pv_w / p_cs / regime."""
    pv, p_cs = np.asarray(table["pv_w"], float), np.asarray(table["p_cs"], float)
    regime = np.asarray(table["regime"], dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_obs = np.where(p_cs > 1, pv / p_cs, np.nan)
    trainable = np.isin(regime, [MPP, LIMITED]) & (p_cs >= cfg.min_p_cs_w) & ~np.isnan(pv)
    return {"k_obs": k_obs, "cens": regime == LIMITED, "w": (p_cs / cfg.p_ref_w) ** 2, "trainable": trainable}


def fit(table: Mapping[str, np.ndarray], rows, features: Sequence[str], cfg: ModelConfig,
        target: str = "k_obs", censored: bool = True) -> GBM:
    """Fit on table rows (bool mask or indices). censored=False treats every target as exact."""
    rows = np.asarray(rows)
    y = np.asarray(table[target], float)[rows]
    cens = np.asarray(table["cens"], bool)[rows] if censored else np.zeros(len(y), bool)
    w = np.asarray(table["w"], float)[rows]
    X = matrix({k: np.asarray(v)[rows] for k, v in table.items() if k in features or k == "ts"}, features)
    return GBM(cfg.params).fit(X, y, cens=cens, weight=w, features=list(features))


def predict_k(model: GBM, table: Mapping[str, np.ndarray], rows=None) -> np.ndarray:
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
