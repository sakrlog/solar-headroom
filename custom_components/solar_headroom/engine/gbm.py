"""Small histogram gradient-boosted trees in numpy, with a censored squared loss.

Why not LightGBM: it has no musl (Alpine) wheels, so it cannot be installed inside Home
Assistant's core container; numpy ships with Home Assistant. On the reference dataset this
learner matches LightGBM (hourly MAE 302 W vs 301 W, same bias and floor violations).

Loss: squared error on exact rows; on censored rows (the observation is only a floor, e.g. the
inverter was throttling) the prediction is penalised only when it falls below the floor.

Features may contain NaN: every feature has a dedicated NaN bin and each split learns which side
missing values go to. Trees grow depth-wise. Models serialise to plain JSON-compatible dicts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

FORMAT_VERSION = 1


@dataclass
class GBMParams:
    rounds: int = 500
    learning_rate: float = 0.03
    max_depth: int = 5
    min_leaf: int = 40               # rows per leaf
    l2: float = 1.0                  # leaf-value regularisation
    n_bins: int = 63                 # value bins per feature (+1 NaN bin); must stay < 255
    subsample: float = 0.8           # rows per tree
    colsample: float = 0.9           # features per tree
    censored_hessian: float = 1e-3   # curvature for floor rows that are already satisfied
    seed: int = 7


@dataclass
class GBM:
    params: GBMParams = field(default_factory=GBMParams)
    features: list[str] = field(default_factory=list)
    init: float = 0.0
    edges: list[np.ndarray] = field(default_factory=list)
    # packed trees: shape (n_trees, n_nodes); feat < 0 marks a leaf
    feat: np.ndarray | None = None
    thr: np.ndarray | None = None
    nan_left: np.ndarray | None = None
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    value: np.ndarray | None = None

    # ------------------------------------------------------------ training
    def fit(self, X, y, cens=None, weight=None, init=None, features: list[str] | None = None) -> GBM:
        """X: (n, f) float array (NaN allowed). y: targets (floors where cens). init: scalar or per-row
        starting prediction (default: median of the exact targets)."""
        p = self.params
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, nf = X.shape
        cens = np.zeros(n, bool) if cens is None else np.asarray(cens, bool)
        w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
        if not 2 <= p.n_bins < 255:
            raise ValueError("n_bins must be in [2, 254]")
        self.features = list(features) if features is not None else [f"f{i}" for i in range(nf)]
        if init is None:
            exact = y[~cens]
            init = float(np.median(exact)) if len(exact) else float(np.median(y))
        base = np.broadcast_to(np.asarray(init, dtype=float), (n,)).copy()
        self.init = float(init) if np.ndim(init) == 0 else float("nan")    # NaN: predict() needs init
        self.edges = [self._feature_edges(X[:, j]) for j in range(nf)]
        B = self._bin(X)
        rng = np.random.default_rng(p.seed)
        pred = base.copy()
        trees = []
        for _ in range(p.rounds):
            d = pred - y
            active = ~cens | (d < 0)
            g = np.where(active, d, 0.0) * w
            h = np.where(active, 1.0, p.censored_hessian) * w
            rows = np.flatnonzero(rng.random(n) < p.subsample) if p.subsample < 1 else np.arange(n)
            cols = np.flatnonzero(rng.random(nf) < p.colsample) if p.colsample < 1 else np.arange(nf)
            if len(cols) == 0:
                cols = np.arange(nf)
            tree = self._grow(B, g, h, rows, cols)
            trees.append(tree)
            pred += p.learning_rate * self._tree_predict(tree, B)
        self._pack(trees)
        return self

    def _feature_edges(self, col: np.ndarray) -> np.ndarray:
        c = col[~np.isnan(col)]
        if len(c) == 0:
            return np.zeros(0)
        q = np.quantile(c, np.linspace(0, 1, self.params.n_bins + 1)[1:-1])
        return np.unique(q)

    def _bin(self, X: np.ndarray) -> np.ndarray:
        nan_bin = self.params.n_bins
        B = np.empty(X.shape, dtype=np.uint8)
        for j, e in enumerate(self.edges):
            col = X[:, j]
            b = np.searchsorted(e, col, side="right")
            b[np.isnan(col)] = nan_bin
            B[:, j] = b
        return B

    def _grow(self, B, g, h, rows, cols) -> dict[str, list]:
        p = self.params
        nb, lam, nc = p.n_bins + 1, p.l2, len(cols)
        offsets = np.arange(nc) * nb
        t = {"feat": [], "thr": [], "nan_left": [], "left": [], "right": [], "value": []}

        def new_node() -> int:
            for key, v in (("feat", -1), ("thr", 0), ("nan_left", False), ("left", -1), ("right", -1), ("value", 0.0)):
                t[key].append(v)
            return len(t["feat"]) - 1

        frontier = [(new_node(), rows)]
        for depth in range(p.max_depth + 1):
            nxt = []
            for nid, idx in frontier:
                G, H = g[idx].sum(), h[idx].sum()
                t["value"][nid] = -G / (H + lam)
                if depth == p.max_depth or len(idx) < 2 * p.min_leaf:
                    continue
                flat = (B[np.ix_(idx, cols)].astype(np.int64) + offsets).ravel()
                Gh = np.bincount(flat, np.repeat(g[idx], nc), nc * nb).reshape(nc, nb)
                Hh = np.bincount(flat, np.repeat(h[idx], nc), nc * nb).reshape(nc, nb)
                Ch = np.bincount(flat, None, nc * nb).reshape(nc, nb)
                GL, HL, CL = (np.cumsum(a[:, :-1], axis=1) for a in (Gh, Hh, Ch))
                parent = G * G / (H + lam)
                best = (0.0, -1, -1, False)
                for nan_left in (False, True):
                    gl = GL + Gh[:, -1:] if nan_left else GL
                    hl = HL + Hh[:, -1:] if nan_left else HL
                    cl = CL + Ch[:, -1:] if nan_left else CL
                    gain = gl * gl / (hl + lam) + (G - gl) ** 2 / (H - hl + lam) - parent
                    gain = np.where((cl >= p.min_leaf) & (len(idx) - cl >= p.min_leaf), gain, -np.inf)
                    k = int(np.argmax(gain))
                    jj, bb = divmod(k, gain.shape[1])
                    if gain[jj, bb] > best[0]:
                        best = (float(gain[jj, bb]), jj, bb, nan_left)
                _, jj, bb, nan_left = best
                if jj < 0:
                    continue
                f = int(cols[jj])
                col = B[idx, f]
                go_left = np.where(col == p.n_bins, nan_left, col <= bb)
                lnode, rnode = new_node(), new_node()
                t["feat"][nid], t["thr"][nid], t["nan_left"][nid] = f, bb, nan_left
                t["left"][nid], t["right"][nid] = lnode, rnode
                nxt += [(lnode, idx[go_left]), (rnode, idx[~go_left])]
            frontier = nxt
        return t

    def _tree_predict(self, t: dict[str, list], B: np.ndarray) -> np.ndarray:
        feat, thr = np.asarray(t["feat"]), np.asarray(t["thr"])
        nan_left, left, right = np.asarray(t["nan_left"]), np.asarray(t["left"]), np.asarray(t["right"])
        node = np.zeros(len(B), dtype=np.int64)
        rows = np.arange(len(B))
        for _ in range(self.params.max_depth + 1):
            f = feat[node]
            leaf = f < 0
            if leaf.all():
                break
            b = B[rows, np.where(leaf, 0, f)]
            go = np.where(b == self.params.n_bins, nan_left[node], b <= thr[node])
            node = np.where(leaf, node, np.where(go, left[node], right[node]))
        return np.asarray(t["value"])[node]

    def _pack(self, trees: list[dict[str, list]]) -> None:
        nt, nn = len(trees), max((len(t["feat"]) for t in trees), default=1)
        self.feat = np.full((nt, nn), -1, dtype=np.int16)
        self.thr = np.zeros((nt, nn), dtype=np.uint8)
        self.nan_left = np.zeros((nt, nn), dtype=bool)
        self.left = np.zeros((nt, nn), dtype=np.int32)
        self.right = np.zeros((nt, nn), dtype=np.int32)
        self.value = np.zeros((nt, nn), dtype=float)
        for i, t in enumerate(trees):
            k = len(t["feat"])
            self.feat[i, :k], self.thr[i, :k], self.nan_left[i, :k] = t["feat"], t["thr"], t["nan_left"]
            self.left[i, :k], self.right[i, :k], self.value[i, :k] = t["left"], t["right"], t["value"]

    # ------------------------------------------------------------ prediction
    @property
    def n_trees(self) -> int:
        return 0 if self.feat is None else self.feat.shape[0]

    def predict(self, X, init=None, chunk: int = 1024) -> np.ndarray:
        """Predictions for X (same feature order as training). init: scalar or per-row starting
        prediction; defaults to the scalar learned at fit time (pass the same prior you trained with)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.edges):
            raise ValueError(f"expected {len(self.edges)} features, got {X.shape[1]}")
        B = self._bin(X)
        out = np.zeros(len(B))
        if self.n_trees:
            tix = np.arange(self.n_trees)[None, :]
            for s in range(0, len(B), chunk):
                b = B[s:s + chunk]
                rows = np.arange(len(b))[:, None]
                node = np.zeros((len(b), self.n_trees), dtype=np.int64)
                for _ in range(self.params.max_depth + 1):
                    f = self.feat[tix, node]
                    leaf = f < 0
                    if leaf.all():
                        break
                    bins = b[rows, np.where(leaf, 0, f)]
                    go = np.where(bins == self.params.n_bins, self.nan_left[tix, node], bins <= self.thr[tix, node])
                    node = np.where(leaf, node, np.where(go, self.left[tix, node], self.right[tix, node]))
                out[s:s + chunk] = self.value[tix, node].sum(axis=1)
        if init is None and np.isnan(self.init):
            raise ValueError("model was trained with a per-row init: pass the same prior to predict()")
        start = self.init if init is None else np.asarray(init, dtype=float)
        return start + self.params.learning_rate * out

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"format": FORMAT_VERSION, "params": asdict(self.params), "features": self.features,
                "init": self.init, "edges": [e.tolist() for e in self.edges],
                "trees": {k: getattr(self, k).tolist() for k in ("feat", "thr", "nan_left", "left", "right", "value")}}

    @classmethod
    def from_dict(cls, d: dict) -> GBM:
        if d.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported model format {d.get('format')}")
        m = cls(params=GBMParams(**d["params"]), features=list(d["features"]), init=float(d["init"]),
                edges=[np.asarray(e, dtype=float) for e in d["edges"]])
        dtypes = {"feat": np.int16, "thr": np.uint8, "nan_left": bool, "left": np.int32, "right": np.int32,
                  "value": float}
        for k, dt in dtypes.items():
            setattr(m, k, np.asarray(d["trees"][k], dtype=dt))
        return m
