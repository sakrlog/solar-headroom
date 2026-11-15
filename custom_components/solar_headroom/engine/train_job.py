"""Nightly training as a separate, low-priority process: `python -I train_job.py <store.sqlite>`.

Home Assistant starts it with its own interpreter, so numpy is the one Home Assistant ships. Memory used
by training is returned to the system when the process ends. The system description is read from the
store (meta "system"), the result is written back (blob "models"), and one JSON line with the report is
printed to stdout; progress goes to stderr.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ENGINE = "solar_headroom_engine"


def _engine():
    """Load this package under a private name: the integration's folder is never put on sys.path, so its
    modules (sensor.py, const.py, ...) cannot shadow anything."""
    if ENGINE not in sys.modules:
        here = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(ENGINE, here / "__init__.py",
                                                      submodule_search_locations=[str(here)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[ENGINE] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{ENGINE}.pipeline"), importlib.import_module(f"{ENGINE}.store")


def peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:          # not on Linux / macOS
        return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(kb / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def run(store_path: str, now: float | None = None, log=None) -> dict:
    """Train from the store and save the result; returns the report. Also used in-process as a fallback."""
    P, St = _engine()
    store = St.Store(store_path)
    system = P.System.from_dict(store.get_meta("system"))
    t0 = time.time()
    rows = store.row_columns(system.statistic_ids())
    frames = P.archive_frames(system, store.responses)
    result = P.train(system, rows, frames, time.time() if now is None else now,
                     previous=store.get_blob("models"), log=log or (lambda _: None))
    result["report"]["wall_s"] = round(time.time() - t0, 1)
    result["report"]["peak_rss_mb"] = peak_rss_mb()
    store.put_blob("models", result)
    return result["report"]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: train_job.py <store.sqlite>", file=sys.stderr)
        return 2
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    report = run(argv[1], log=lambda m: print(m, file=sys.stderr, flush=True))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
