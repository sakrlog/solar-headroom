"""engine.gbm: learning, censoring, missing values, persistence, parity with LightGBM (dev-only)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from engine.gbm import GBM, GBMParams


def make_data(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2] + rng.normal(0, 0.1, n)
    miss = rng.random(n) < 0.15
    y[miss] += 1.0                                   # missing x3 carries information
    X[miss, 3] = np.nan
    X[:, 3] = np.where(miss, np.nan, rng.normal(size=n))   # x3 itself is noise when present
    return X, y


def split(X, y, frac=0.75):
    k = int(len(y) * frac)
    return X[:k], y[:k], X[k:], y[k:]


FAST = GBMParams(rounds=200, learning_rate=0.08, min_leaf=20)


def test_learns_signal_and_missing_values():
    Xa, ya, Xb, yb = split(*make_data())
    m = GBM(FAST).fit(Xa, ya)
    mae = np.mean(np.abs(m.predict(Xb) - yb))
    baseline = np.mean(np.abs(np.median(ya) - yb))
    assert mae < 0.3 * baseline, (mae, baseline)
    nan_rows = np.isnan(Xb[:, 3])
    assert np.mean(m.predict(Xb[nan_rows]) - yb[nan_rows]) == pytest.approx(0, abs=0.1)


def test_censored_rows_are_floors_not_targets():
    X, truth = make_data(seed=2)
    rng = np.random.default_rng(3)
    cens = rng.random(len(truth)) < 0.6
    observed = np.where(cens, truth - rng.uniform(0.2, 1.5, len(truth)), truth)   # floors sit below the truth
    Xa, oa, Xb, tb = split(X, observed)[0], split(X, observed)[1], split(X, truth)[2], split(X, truth)[3]
    ca = cens[: len(oa)]
    censored = GBM(FAST).fit(Xa, oa, cens=ca)
    naive = GBM(FAST).fit(Xa, oa)                     # treats floors as exact values
    err_c = np.mean(censored.predict(Xb) - tb)
    err_n = np.mean(naive.predict(Xb) - tb)
    assert abs(err_c) < 0.15, err_c                   # unbiased against the hidden truth
    assert err_n < -0.3, err_n                        # naive learning is biased low


def test_per_row_init_learns_residual():
    Xa, ya, Xb, yb = split(*make_data(seed=4))
    prior_a, prior_b = np.sin(Xa[:, 0]), np.sin(Xb[:, 0])
    m = GBM(FAST).fit(Xa, ya, init=prior_a)
    with pytest.raises(ValueError):
        m.predict(Xb)
    assert np.mean(np.abs(m.predict(Xb, init=prior_b) - yb)) < 0.25


def test_json_roundtrip_and_determinism():
    Xa, ya, Xb, _ = split(*make_data(seed=5))
    m = GBM(FAST).fit(Xa, ya, features=["a", "b", "c", "d"])
    m2 = GBM.from_dict(json.loads(json.dumps(m.to_dict())))
    assert m2.features == ["a", "b", "c", "d"]
    assert np.array_equal(m.predict(Xb), m2.predict(Xb))
    again = GBM(FAST).fit(Xa, ya)
    assert np.array_equal(m.predict(Xb), again.predict(Xb))
    other = GBM(GBMParams(**{**FAST.__dict__, "seed": 99})).fit(Xa, ya)
    assert not np.array_equal(m.predict(Xb), other.predict(Xb))


def test_single_row_equals_batch_and_edge_columns():
    Xa, ya, Xb, _ = split(*make_data(seed=6))
    Xa = np.column_stack([Xa, np.full(len(Xa), 3.0), np.full(len(Xa), np.nan)])   # constant + all-NaN columns
    Xb = np.column_stack([Xb, np.full(len(Xb), 3.0), np.full(len(Xb), np.nan)])
    m = GBM(FAST).fit(Xa, ya)
    batch = m.predict(Xb)
    singles = np.array([m.predict(Xb[i])[0] for i in range(25)])
    assert np.allclose(batch[:25], singles)
    with pytest.raises(ValueError):
        m.predict(Xb[:, :3])


def test_parity_with_lightgbm_censored():
    lgb = pytest.importorskip("lightgbm")
    X, truth = make_data(n=6000, seed=7)
    rng = np.random.default_rng(8)
    cens = rng.random(len(truth)) < 0.5
    obs = np.where(cens, truth - rng.uniform(0, 1, len(truth)), truth)
    Xa, oa, Xb = X[:4500], obs[:4500], X[4500:]
    ca, tb = cens[:4500], truth[4500:]
    init = float(np.median(oa[~ca]))

    def objective(preds, ds):
        d = preds - ds.get_label()
        act = ~ca | (d < 0)
        return np.where(act, d, 0.0), np.where(act, 1.0, 1e-3)
    params = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=1, seed=7, objective=objective)
    ds = lgb.Dataset(Xa, oa, init_score=np.full(len(oa), init))
    ref = lgb.train(params, ds, num_boost_round=500).predict(Xb) + init
    ours = GBM().fit(Xa, oa, cens=ca, init=init).predict(Xb)
    mae_ref, mae_ours = np.mean(np.abs(ref - tb)), np.mean(np.abs(ours - tb))
    assert mae_ours < 1.05 * mae_ref, (mae_ours, mae_ref)
