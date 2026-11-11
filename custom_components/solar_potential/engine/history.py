"""Turn Home Assistant statistics into aligned period arrays.

Home Assistant keeps, for every measurement sensor, the mean / min / max per hour forever (long-term
statistics) and per 5 minutes for ~10 days (short-term statistics). The recorder returns them as
``{statistic_id: [{"start": <epoch s>, "mean": .., "min": .., "max": ..}, ...]}``
(``recorder.statistics.statistics_during_period``). This module turns those rows into one array per
role on a regular grid of period starts, which is what the labelling rules and features consume.

A *role* (e.g. ``pv_w``) is the sum of one or more *parts* (PV strings, several inverters). A part
is a list of statistic ids over time: the same physical sensor under a new entity id after an
integration was re-added, with optional validity windows. For every period the first id that has a
row wins (list the newest id first).

Units and signs are applied per id with a scale factor (kW -> W: 1000; flip a battery sensor that
reports + = charging: -1). A negative scale swaps min and max.

Sums of parts: mean and the extremes are summed per period. The max of a sum can only be smaller
than the sum of the maxima (and the min only larger), which is exact when the parts peak together
(strings under the same sky) and conservative otherwise. A period counts as missing when any part
that is valid at that time has no row.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

FIELDS = ("mean", "min", "max")
HOUR = 3600
FIVE_MIN = 300


@dataclass(frozen=True)
class Source:
    """One statistic id (or rows supplied directly) with its unit/sign scale and validity window."""
    statistic_id: str
    scale: float = 1.0
    since: float | None = None       # epoch s; rows before are ignored
    until: float | None = None       # epoch s; rows at/after are ignored


Part = Sequence[Source]              # same sensor over time, newest id first
Role = Sequence[Part]                # parts that are summed


def _start_s(v) -> float:
    if isinstance(v, datetime):
        return v.timestamp()
    return float(v)


def grid(start: float, end: float, period: int) -> np.ndarray:
    """Period starts from start (floored to the period) up to, not including, end."""
    s = np.floor(start / period) * period
    return np.arange(s, end, period, dtype=float)


def rows_to_arrays(rows: Iterable[Mapping], starts: np.ndarray, scale: float = 1.0,
                   since: float | None = None, until: float | None = None) -> dict[str, np.ndarray]:
    """mean/min/max of statistics rows placed on `starts` (NaN where no row). Rows whose start is not
    on the grid are ignored (a wrong period length would otherwise mix silently)."""
    starts = np.asarray(starts, dtype=float)
    n = len(starts)
    out = {f: np.full(n, np.nan) for f in FIELDS}
    rows = list(rows)
    if not rows or n == 0:
        return out
    t = np.array([_start_s(r["start"]) for r in rows])
    vals = {f: np.array([np.nan if r.get(f) is None else float(r[f]) for r in rows]) for f in FIELDS}
    keep = np.ones(len(t), bool)
    if since is not None:
        keep &= t >= since
    if until is not None:
        keep &= t < until
    t = t[keep]
    vals = {f: v[keep] for f, v in vals.items()}
    i = np.clip(np.searchsorted(starts, t), 0, n - 1)
    hit = starts[i] == t
    i, t = i[hit], t[hit]
    for f in FIELDS:
        out[f][i] = vals[f][hit] * scale
    if scale < 0:
        out["min"], out["max"] = out["max"], out["min"]
    if np.isnan(out["mean"]).all() and not np.isnan(out["min"]).all():
        # sensors without a mean (rare): use the midpoint of the extremes
        out["mean"] = (out["min"] + out["max"]) / 2
    return out


def _valid_window(src: Source, starts: np.ndarray) -> np.ndarray:
    ok = np.ones(len(starts), bool)
    if src.since is not None:
        ok &= starts >= src.since
    if src.until is not None:
        ok &= starts < src.until
    return ok


def part_arrays(part: Part, stats: Mapping[str, list], starts: np.ndarray) -> tuple[dict, np.ndarray]:
    """First id with a row wins, per period. Returns (arrays, applicable) where `applicable` marks the
    periods inside at least one id's validity window."""
    n = len(starts)
    out = {f: np.full(n, np.nan) for f in FIELDS}
    applicable = np.zeros(n, bool)
    for src in part:
        a = rows_to_arrays(stats.get(src.statistic_id, ()), starts, src.scale, src.since, src.until)
        applicable |= _valid_window(src, starts)
        free = np.isnan(out["mean"]) & ~np.isnan(a["mean"])
        for f in FIELDS:
            out[f][free] = a[f][free]
    return out, applicable


def role_arrays(role: Role, stats: Mapping[str, list], starts: np.ndarray) -> dict[str, np.ndarray]:
    """Sum of the parts of a role; NaN where an applicable part has no row."""
    n = len(starts)
    total = {f: np.zeros(n) for f in FIELDS}
    any_part = np.zeros(n, bool)
    missing = np.zeros(n, bool)
    for part in role:
        a, applicable = part_arrays(part, stats, starts)
        have = ~np.isnan(a["mean"])
        missing |= applicable & ~have
        any_part |= have
        for f in FIELDS:
            total[f] += np.where(have, np.nan_to_num(a[f], nan=0.0), 0.0)
        # extremes missing while the mean exists: fall back to the mean for that part
        for f in ("min", "max"):
            gap = have & np.isnan(a[f])
            total[f][gap] += a["mean"][gap]
    bad = missing | ~any_part
    for f in FIELDS:
        total[f][bad] = np.nan
    return total


def period_table(roles: Mapping[str, Role], stats: Mapping[str, list], start: float, end: float,
                 period: int = HOUR) -> dict[str, np.ndarray]:
    """{"ts": period starts, "<role>": mean, "<role>_min", "<role>_max"} for every role."""
    starts = grid(start, end, period)
    table = {"ts": starts}
    for name, role in roles.items():
        a = role_arrays(role, stats, starts)
        table[name], table[f"{name}_min"], table[f"{name}_max"] = a["mean"], a["min"], a["max"]
    return table


def statistic_ids(roles: Mapping[str, Role]) -> list[str]:
    """Every statistic id the roles need (for one recorder query)."""
    seen: dict[str, None] = {}
    for role in roles.values():
        for part in role:
            for src in part:
                seen.setdefault(src.statistic_id, None)
    return list(seen)


def chunks(start: float, end: float, days: int = 31) -> list[tuple[float, float]]:
    """Consecutive [a, b) windows for paced history reads (one recorder query per window)."""
    out, a, step = [], float(start), days * 86400.0
    while a < end:
        b = min(a + step, float(end))
        out.append((a, b))
        a = b
    return out


def coverage(table: Mapping[str, np.ndarray], role: str) -> float:
    """Share of periods with a value for `role`."""
    v = np.asarray(table[role], dtype=float)
    return float(np.mean(~np.isnan(v))) if len(v) else 0.0
