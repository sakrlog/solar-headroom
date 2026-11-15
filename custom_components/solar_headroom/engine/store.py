"""The integration's own data: statistics rows, Open-Meteo responses and trained models (SQLite).

One file per config entry. Home Assistant writes to it; the training subprocess reads it and writes the
models back. WAL mode lets both work at the same time. Every method opens its own short connection, so
an instance can be used from any thread.

Size: hourly rows ~0.3 MB per sensor and year; weather responses are stored zlib-compressed per calendar
quarter (~1-3 MB per year); models replace each other.
"""
from __future__ import annotations

import json
import sqlite3
import time
import zlib
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS stat_rows (
    statistic_id TEXT NOT NULL, start REAL NOT NULL, mean REAL, min REAL, max REAL,
    PRIMARY KEY (statistic_id, start)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY, kind TEXT NOT NULL, first_day TEXT, last_day TEXT, fetched REAL NOT NULL,
    final INTEGER NOT NULL, body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS blobs (name TEXT PRIMARY KEY, saved REAL NOT NULL, body BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


_MISSING = -1.0e300          # NULL inside numeric row columns (SQLite has no NaN)


def _pack(obj: Any) -> bytes:
    return zlib.compress(json.dumps(obj, separators=(",", ":"), allow_nan=True).encode(), 6)


def _unpack(blob: bytes) -> Any:
    return json.loads(zlib.decompress(blob))


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    # ------------------------------------------------------------ statistics rows
    def put_rows(self, statistic_id: str, rows: Iterable[Mapping]) -> int:
        """Insert or replace rows {"start": epoch s, "mean", "min", "max"}; returns how many."""
        data = [(statistic_id, float(r["start"]), r.get("mean"), r.get("min"), r.get("max")) for r in rows]
        with closing(self._connect()) as db, db:
            db.executemany("INSERT OR REPLACE INTO stat_rows VALUES (?, ?, ?, ?, ?)", data)
        return len(data)

    def rows(self, ids: Sequence[str], start: float | None = None, end: float | None = None) -> dict[str, list[dict]]:
        """{statistic_id: rows sorted by start} in [start, end)."""
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        q = f"SELECT statistic_id, start, mean, min, max FROM stat_rows WHERE statistic_id IN ({marks})"
        args: list = list(ids)
        if start is not None:
            q += " AND start >= ?"
            args.append(float(start))
        if end is not None:
            q += " AND start < ?"
            args.append(float(end))
        out: dict[str, list[dict]] = {i: [] for i in ids}
        with closing(self._connect()) as db:
            for sid, s, mean, lo, hi in db.execute(q + " ORDER BY start", args):
                out[sid].append({"start": s, "mean": mean, "min": lo, "max": hi})
        return out

    def row_columns(self, ids: Sequence[str], start: float | None = None,
                    end: float | None = None) -> dict[str, dict[str, np.ndarray]]:
        """Like rows(), as float columns {"start", "mean", "min", "max"} per id (NaN = missing value).
        A few MB for years of hourly rows, where row dicts would take ~100 MB."""
        out = {}
        dtype = [("start", "f8"), ("mean", "f8"), ("min", "f8"), ("max", "f8")]
        q = ("SELECT start, IFNULL(mean, ?), IFNULL(min, ?), IFNULL(max, ?) FROM stat_rows "
             "WHERE statistic_id = ? AND start >= ? AND start < ? ORDER BY start")
        lo, hi = -np.inf if start is None else float(start), np.inf if end is None else float(end)
        with closing(self._connect()) as db:
            for sid in ids:
                a = np.fromiter(db.execute(q, (_MISSING, _MISSING, _MISSING, sid, lo, hi)), dtype=dtype)
                cols = {f: a[f].copy() for f in ("start", "mean", "min", "max")}
                for f in ("mean", "min", "max"):
                    cols[f][cols[f] == _MISSING] = np.nan
                out[sid] = cols
        return out

    def row_span(self, statistic_id: str) -> tuple[float, float] | None:
        """(first start, last start) stored for an id, or None."""
        with closing(self._connect()) as db:
            a, b = db.execute("SELECT MIN(start), MAX(start) FROM stat_rows WHERE statistic_id = ?",
                              (statistic_id,)).fetchone()
        return None if a is None else (float(a), float(b))

    # ------------------------------------------------------------ Open-Meteo responses
    def put_response(self, key: str, kind: str, body: Mapping, first_day: str | None = None,
                     last_day: str | None = None, final: bool = False) -> None:
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (key, kind, first_day, last_day, time.time(), int(final), _pack(body)))

    def response_index(self) -> dict[str, tuple[bool, float]]:
        """{key: (final, fetched epoch s)} of every stored response."""
        with closing(self._connect()) as db:
            return {k: (bool(f), t) for k, f, t in db.execute("SELECT key, final, fetched FROM responses")}

    def responses(self, prefix: str) -> list[tuple[str, dict]]:
        """(key, body) of responses whose key starts with prefix, ordered by first day."""
        with closing(self._connect()) as db:
            got = db.execute("SELECT key, body FROM responses WHERE substr(key, 1, ?) = ? ORDER BY first_day, key",
                             (len(prefix), prefix)).fetchall()
        return [(k, _unpack(b)) for k, b in got]

    def drop_responses(self, keep: Iterable[str], kind: str) -> int:
        """Delete responses of a kind that are no longer requested (e.g. after a panel change)."""
        keep = set(keep)
        with closing(self._connect()) as db, db:
            old = [k for (k,) in db.execute("SELECT key FROM responses WHERE kind = ?", (kind,)) if k not in keep]
            db.executemany("DELETE FROM responses WHERE key = ?", [(k,) for k in old])
        return len(old)

    # ------------------------------------------------------------ models and other JSON blobs
    def put_blob(self, name: str, obj: Any) -> None:
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?, ?)", (name, time.time(), _pack(obj)))

    def get_blob(self, name: str) -> Any | None:
        with closing(self._connect()) as db:
            got = db.execute("SELECT body FROM blobs WHERE name = ?", (name,)).fetchone()
        return None if got is None else _unpack(got[0])

    def get_meta(self, key: str, default: Any = None) -> Any:
        with closing(self._connect()) as db:
            got = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if got is None else json.loads(got[0])

    def set_meta(self, key: str, value: Any) -> None:
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*") if p.is_file())


__all__ = ["Store"]
