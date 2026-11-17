"""Coordinator: history sync, Open-Meteo, nightly training and the outputs every few minutes."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import aiohttp

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfElectricPotential, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    ARCHIVE_WORKERS,
    CONF_AZIMUTH,
    CONF_EXPORT_LIMIT,
    CONF_KIND,
    CONF_KWP,
    CONF_PV_POWER,
    CONF_TILT,
    DOMAIN,
    KIND_UNLIMITED,
    LIVE_MAX_AGE_S,
    LIVE_SAT_MAX_GAP_S,
    LIVE_SAT_MIN_GAP_S,
    LIVE_STALE_S,
    LOGGER,
    REQUEST_PAUSE_S,
    REQUEST_TIMEOUT_S,
    RETRAIN_AFTER_S,
    SENSOR_ROLES,
    STATS_CHUNK_DAYS,
    STATUS_COLLECTING,
    STATUS_DEGRADED,
    STATUS_LEARNING,
    STATUS_READY,
    STATUS_STARTING,
    SUBENTRY_TYPE_PLANE,
    TRAIN_HOUR,
    TRAIN_TIMEOUT_S,
    UPDATE_INTERVAL,
)
from .engine import history as H
from .engine import pipeline as P
from .engine import solar as S
from .engine.store import Store
from .live import LiveTracker

type SolarHeadroomConfigEntry = ConfigEntry[SolarHeadroomCoordinator]

TRAIN_JOB = Path(__file__).parent / "engine" / "train_job.py"
STAT_TYPES: set = {"mean", "min", "max"}
UNITS = {"power": UnitOfPower.WATT, "voltage": UnitOfElectricPotential.VOLT}


class OpenMeteoError(Exception):
    """Open-Meteo answered with an error."""


class TrainingError(Exception):
    """The training process failed."""


def roles_from_conf(conf: dict) -> dict[str, list[list[dict]]]:
    """Configured sensors -> engine roles: PV sensors are added up; battery and grid power carry their
    sign (Home Assistant's convention: battery + = discharging, grid + = importing; inverted flips it)."""
    roles = {"pv_w": [[{"statistic_id": eid}] for eid in conf[CONF_PV_POWER]]}
    if conf.get(CONF_KIND, KIND_UNLIMITED) == KIND_UNLIMITED:
        return roles
    for key, (role, inverted) in SENSOR_ROLES.items():
        if conf.get(key):
            roles[role] = [[{"statistic_id": conf[key], "scale": -1.0 if inverted and conf.get(inverted) else 1.0}]]
    return roles


def entry_conf(entry: ConfigEntry) -> dict:
    """The entry's settings: the options once saved by the options flow (complete), else the setup data."""
    return dict(entry.options) if entry.options.get(CONF_KIND) else {**entry.data, **entry.options}


def system_from_entry(hass: HomeAssistant, entry: ConfigEntry) -> P.System:
    conf = entry_conf(entry)
    planes = tuple(S.Plane(float(s.data[CONF_TILT]), float(s.data[CONF_AZIMUTH]), float(s.data[CONF_KWP]))
                   for s in entry.get_subentries_of_type(SUBENTRY_TYPE_PLANE))
    return P.System(float(hass.config.latitude), float(hass.config.longitude), float(hass.config.elevation or 0.0),
                    str(hass.config.time_zone), planes, roles_from_conf(conf), conf.get(CONF_KIND, KIND_UNLIMITED),
                    export_limit_w=float(conf.get(CONF_EXPORT_LIMIT) or 0.0))


def _plain(rows: dict[str, list]) -> dict[str, list[dict]]:
    """Recorder statistics rows -> {"start": epoch s, "mean", "min", "max"}."""
    out = {}
    for sid, rs in rows.items():
        out[sid] = [{"start": float(r["start"].timestamp() if isinstance(r["start"], datetime) else r["start"]),
                     "mean": r.get("mean"), "min": r.get("min"), "max": r.get("max")} for r in rs]
    return out


class SolarHeadroomCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Owns the store, the models and the schedule of one PV system."""

    config_entry: SolarHeadroomConfigEntry

    def __init__(self, hass: HomeAssistant, entry: SolarHeadroomConfigEntry) -> None:
        super().__init__(hass, LOGGER, config_entry=entry, name=DOMAIN, update_interval=UPDATE_INTERVAL)
        self.system = system_from_entry(hass, entry)
        self.store_path = hass.config.path(DOMAIN, f"{entry.entry_id}.sqlite")
        self.store: Store | None = None
        self.result: dict | None = None
        self._models: dict = {}
        self._live: dict = {}
        self.live_at = 0.0
        self.sats: dict = {}                      # chosen satellite products {"history", "live", "offsets"}
        self._live_sat: tuple | None = None
        self.live_sat_at = 0.0
        self._live_sat_misses = 0                 # fetches in a row that did not bring the needed interval
        seed = int(hashlib.sha256(entry.entry_id.encode()).hexdigest()[:8], 16)
        self.jitter_s = seed % 600                # installs spread their forecast fetches over 10 minutes
        self.train_minute = seed % 60
        self.status = STATUS_STARTING
        self.progress = ""
        self.last_error: str | None = None
        self.last_report: dict | None = None
        self._learning: asyncio.Task | None = None
        self.requests: dict = {"day": None, "total": 0, "by_kind": {}}
        self.requests_yesterday: dict | None = None
        self.tracker: LiveTracker | None = None
        self._t_air: Any = 20.0

    # ------------------------------------------------------------ setup
    async def async_setup(self) -> None:
        """Open the store, load the last training result, schedule learning."""
        self.store = await self.hass.async_add_executor_job(Store, self.store_path)
        self.result = await self.hass.async_add_executor_job(self.store.get_blob, "models")
        self._models = await self.hass.async_add_executor_job(P.load_models, self.result)
        self.sats = (await self.hass.async_add_executor_job(self.store.get_meta, "satellite") or {}).get("choice") or {}
        if self.result:
            self.last_report = self.result.get("report")
            ready = (self.last_report or {}).get("status") == "ready"
            self.status = STATUS_READY if ready else STATUS_COLLECTING
        entry = self.config_entry
        self.tracker = LiveTracker(self.hass, entry.entry_id, self.system, entry_conf(entry),
                                   (self.result or {}).get("regime"))
        await self.tracker.async_start(entry)
        entry.async_on_unload(async_at_started(self.hass, self._async_started))
        entry.async_on_unload(async_track_time_change(self.hass, self._async_nightly, hour=TRAIN_HOUR,
                                                      minute=self.train_minute, second=0))

    async def _async_started(self, _hass: HomeAssistant) -> None:
        trained = float((self.result or {}).get("trained_at") or 0)
        changed = (self.result or {}).get("system") != self.system.to_dict()
        if not self.result or changed or time.time() - trained > RETRAIN_AFTER_S:
            self.async_schedule_learning("startup")

    async def _async_nightly(self, _now: datetime) -> None:
        self.async_schedule_learning("nightly")

    @callback
    def async_schedule_learning(self, reason: str) -> None:
        """Sync history and weather archives, then train; one run at a time."""
        if self._learning is not None and not self._learning.done():
            return
        self._learning = self.config_entry.async_create_background_task(
            self.hass, self._async_learn(reason), f"{DOMAIN} learning ({reason})")

    @property
    def learning(self) -> bool:
        return self._learning is not None and not self._learning.done()

    # ------------------------------------------------------------ outputs
    async def _async_update_data(self) -> dict[str, Any]:
        now = time.time()
        if not self._live or now - self.live_at > LIVE_MAX_AGE_S + self.jitter_s:
            try:
                await self._async_fetch_live()
            except (aiohttp.ClientError, TimeoutError, OpenMeteoError, ValueError) as err:
                LOGGER.warning("Open-Meteo forecast not available: %s", err)
                self.last_error = f"forecast: {err}"
        if not self._live:
            raise UpdateFailed("no weather forecast yet (Open-Meteo unreachable)")
        if self._live_sat_wanted(now):
            await self._async_fetch_live_satellite()
        rows5 = await self._async_today_rows()
        log = self.tracker.day_log_copy() if self.tracker else None
        data = await self.hass.async_add_executor_job(P.outputs, self.system, self.result, self._live, now, rows5,
                                                      self._models, self._live_sat, log)
        if self.tracker is not None:
            sky = data.get("now") or {}
            cs = sky.get("clear_sky_w") or 0.0
            self.tracker.set_sky(sky["w"] / cs if cs > 1 and sky.get("w") is not None else None, self._t_air)
        if now - self.live_at > LIVE_STALE_S and not self.learning:
            self.status = STATUS_DEGRADED
        return data

    async def _async_fetch_live(self) -> None:
        session = async_get_clientsession(self.hass)
        bodies = {}
        for r in P.live_requests(self.system):
            try:
                bodies[r.key] = await self._async_get_json(session, r)
            except (aiohttp.ClientError, TimeoutError, OpenMeteoError) as err:
                if "/best_match/" in r.key:
                    raise
                LOGGER.debug("ensemble forecast skipped: %s", err)    # the spread is optional
        self._live = await self.hass.async_add_executor_job(P.live_frames, self.system, bodies)
        self._t_air = await self.hass.async_add_executor_job(P.t_air_from, self._live.get("best_match") or {})
        self.live_at = time.time()
        if self.status == STATUS_DEGRADED and self.last_error and self.last_error.startswith("forecast"):
            self.status = STATUS_READY if self.result else STATUS_COLLECTING
            self.last_error = None

    def _live_sat_wanted(self, now: float) -> bool:
        """Fetch the live satellite when the current quarter hour needs an interval the frame lacks (daylight
        only). At most every 5 min (longer with more panel groups, one request each); while intervals stay
        late, back off to 10, 20, 30 min: ~100 calls a day normally, with the forecast <= ~300 a day."""
        gap = LIVE_SAT_MIN_GAP_S * max(1, math.ceil(len(self.system.planes) / 2))
        if self._live_sat_misses:
            gap = min(LIVE_SAT_MAX_GAP_S, gap * 2 ** self._live_sat_misses)
        return now - self.live_sat_at >= gap and P.live_sat_due(self.system, self.sats, self._live_sat, now)

    async def _async_fetch_live_satellite(self) -> None:
        self.live_sat_at = time.time()
        session = async_get_clientsession(self.hass)
        model = self.sats["live"]
        try:
            bodies = [[await self._async_get_json(session, r)]
                      for r in P.live_satellite_requests(self.system, self.sats, time.time())]
        except (aiohttp.ClientError, TimeoutError, OpenMeteoError) as err:
            LOGGER.debug("live satellite not available: %s", err)    # the estimate falls back to the forecast
            self._live_sat_misses += 1
            return
        self._live_sat = await self.hass.async_add_executor_job(P.satellite_frame, self.system, model, bodies)
        late = P.live_sat_due(self.system, self.sats, self._live_sat, time.time())
        self._live_sat_misses = self._live_sat_misses + 1 if late else 0

    def _count_request(self, kind: str) -> None:
        """Open-Meteo requests per UTC day and purpose (diagnostics; the budget is ~300 a day)."""
        day = datetime.now(UTC).date().isoformat()
        if self.requests["day"] != day:
            if self.requests["day"] is not None:
                self.requests_yesterday = self.requests
            self.requests = {"day": day, "total": 0, "by_kind": {}}
        self.requests["total"] += 1
        self.requests["by_kind"][kind] = self.requests["by_kind"].get(kind, 0) + 1

    async def _async_get_json(self, session: aiohttp.ClientSession, r: P.Request) -> dict:
        self._count_request(r.kind)
        async with session.get(r.url, params={k: str(v) for k, v in r.params.items()},
                               timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise OpenMeteoError(f"HTTP {resp.status}: {text[:200]}")
            return await resp.json(content_type=None)

    async def _async_stats(self, start: float, end: float | None, ids: set[str], period: str,
                           types: set) -> dict[str, list[dict]]:
        rows = await get_instance(self.hass).async_add_executor_job(
            statistics_during_period, self.hass, datetime.fromtimestamp(start, UTC),
            None if end is None else datetime.fromtimestamp(end, UTC), ids, period, UNITS, types)
        return _plain(rows)

    async def _async_today_rows(self) -> dict[str, list[dict]]:
        midnight = dt_util.start_of_local_day().timestamp()
        return await self._async_stats(midnight, None, set(self.system.statistic_ids()), "5minute", STAT_TYPES)

    # ------------------------------------------------------------ learning
    async def _async_learn(self, reason: str) -> None:
        LOGGER.debug("learning started (%s)", reason)
        self.status = STATUS_LEARNING
        self.async_update_listeners()
        try:
            self.progress = "reading history"
            await self._async_sync_statistics()
            self.progress = "checking satellite coverage"
            await self._async_probe_satellites()
            await self._async_pick_pixels()
            self.progress = "downloading weather archives"
            await self._async_sync_archives()
            self.progress = "training"
            self.async_update_listeners()
            self.last_report = await self._async_train()
            self.result = await self.hass.async_add_executor_job(self.store.get_blob, "models")
            self._models = await self.hass.async_add_executor_job(P.load_models, self.result)
            if self.tracker is not None:
                self.tracker.set_learned((self.result or {}).get("regime"))
            self.status = STATUS_READY if self.last_report.get("status") == "ready" else STATUS_COLLECTING
            self.last_error = None
            LOGGER.info("Solar Headroom trained: %s", json.dumps(self.last_report))
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - a background task must report, not vanish
            LOGGER.exception("learning failed")
            self.status = STATUS_DEGRADED
            self.last_error = f"learning: {err}"
        finally:
            self.progress = ""
        await self.async_refresh()

    async def _async_sync_statistics(self) -> None:
        now = time.time()
        for sid in self.system.statistic_ids():
            span = await self.hass.async_add_executor_job(self.store.row_span, sid)
            if span is not None:
                start = span[1] + H.HOUR
            else:
                months = await self._async_stats(0.0, None, {sid}, "month", {"mean"})
                if not months.get(sid):
                    LOGGER.warning("no long-term statistics for %s (needs state_class: measurement)", sid)
                    continue
                start = months[sid][0]["start"]
            for a, b in H.chunks(start, now, STATS_CHUNK_DAYS):
                rows = await self._async_stats(a, b, {sid}, "hour", STAT_TYPES)
                await self.hass.async_add_executor_job(self.store.put_rows, sid, rows.get(sid, []))

    async def _async_probe_satellites(self) -> None:
        """Which satellite products have data here (re-checked monthly)."""
        meta = await self.hass.async_add_executor_job(self.store.get_meta, "satellite")
        if meta and time.time() - float(meta.get("probed_at", 0)) < P.PROBE_MAX_AGE_S:
            self.sats = meta.get("choice") or {}
            return
        session = async_get_clientsession(self.hass)
        ok, failed = {}, 0
        probes = P.probe_requests(self.system, dt_util.now().date())
        for r in probes:
            model = r.key.split("/", 1)[1]
            try:
                body = await self._async_get_json(session, r)
                ok[model] = await self.hass.async_add_executor_job(P.probe_ok, model, body)
            except OpenMeteoError:
                ok[model] = False                   # answered: no such product here
            except (aiohttp.ClientError, TimeoutError):
                failed += 1                         # network: no verdict
            await asyncio.sleep(REQUEST_PAUSE_S)
        if probes and failed == len(probes):
            return                                  # keep the previous choice, try again next time
        choice = P.choose_satellites(self.system, ok)
        old = (meta or {}).get("choice") or {}
        choice["offsets"] = {m: o for m, o in (old.get("offsets") or {}).items() if m in P.chosen_models(choice)}
        await self.hass.async_add_executor_job(
            self.store.set_meta, "satellite",
            {"probed_at": time.time(), "ok": ok, "choice": choice, "pixels": (meta or {}).get("pixels") or {}})
        self.sats = choice
        LOGGER.debug("satellite products: %s -> %s", ok, choice)

    async def _async_first_day(self) -> date | None:
        """The first local day of stored history (None without any)."""
        spans = [await self.hass.async_add_executor_job(self.store.row_span, sid)
                 for sid in self.system.statistic_ids()]
        spans = [s for s in spans if s is not None]
        return P.local_date(min(s[0] for s in spans), self.system.time_zone) if spans else None

    async def _async_pick_pixels(self) -> None:
        """Per chosen product, the satellite pixel whose clearness best follows the PV at full output
        (yearly; weekly while there is too little full-output history to tell)."""
        meta = await self.hass.async_add_executor_job(self.store.get_meta, "satellite")
        first_day = await self._async_first_day()
        if not meta or first_day is None:
            return
        choice, pixels = meta.setdefault("choice", {}), meta.setdefault("pixels", {})
        offsets = choice.setdefault("offsets", {})
        session, now, today = async_get_clientsession(self.hass), time.time(), dt_util.now().date()
        changed = False
        for model in P.chosen_models(choice):
            info = pixels.setdefault(model, {})
            if model in offsets and now - float(info.get("at", 0)) < P.PIXEL_MAX_AGE_S:
                continue
            if now - float(info.get("tried_at", 0)) < P.PIXEL_RETRY_S:
                continue
            window = P.pixel_window(model, first_day, today)
            truth = None
            if window is not None:
                lo = datetime(window[0].year, window[0].month, window[0].day, tzinfo=UTC).timestamp()
                hi = datetime(window[1].year, window[1].month, window[1].day, tzinfo=UTC).timestamp() + 86400
                rows = await self.hass.async_add_executor_job(self.store.row_columns, self.system.statistic_ids(),
                                                              lo, hi)
                truth = await self.hass.async_add_executor_job(P.pixel_truth, self.system, rows, window)
            if truth is None:                       # too little full-output data yet
                info["tried_at"], changed = now, True
                continue
            scores: dict = {}
            try:
                while (todo := P.pixels_to_try(scores)) and len(scores) < P.PIXEL_MAX_REQUESTS:
                    for off in todo[: P.PIXEL_MAX_REQUESTS - len(scores)]:
                        self.progress = f"finding the best satellite pixel ({len(scores) + 1})"
                        r = P.pixel_request(self.system, model, off, window)
                        body = await self._async_get_json(session, r)
                        scores[off] = await self.hass.async_add_executor_job(P.pixel_score, self.system, truth,
                                                                             body)
                        await asyncio.sleep(REQUEST_PAUSE_S)
            except (aiohttp.ClientError, TimeoutError, OpenMeteoError) as err:
                LOGGER.warning("satellite pixel search for %s stopped: %s", model, err)
                continue                            # next learning tries again
            best = P.best_pixel(scores)
            offsets[model] = list(best)
            info.update(at=now, tried_at=now, hours=len(truth["ts"]),
                        scores={f"{o[0]:+d},{o[1]:+d}": round(c, 4) for o, (c, _) in scores.items() if c == c})
            changed = True
            LOGGER.debug("satellite pixel for %s: offset %s of %d tried", model, best, len(scores))
        if changed:
            await self.hass.async_add_executor_job(self.store.set_meta, "satellite", meta)
        self.sats = choice

    async def _async_sync_archives(self) -> None:
        first_day = await self._async_first_day()
        if first_day is None:
            return
        today = dt_util.now().date()
        reqs = P.archive_requests(self.system, first_day, today) + P.satellite_requests(self.system, self.sats,
                                                                                          first_day, today)
        index = await self.hass.async_add_executor_job(self.store.response_index)
        todo = P.missing(reqs, index, time.time())
        session = async_get_clientsession(self.hass)
        queue, done = list(todo), 0

        async def worker() -> None:
            # a few downloads at a time: Open-Meteo needs 5-20 s per archive chunk, so one at a time would
            # keep a fresh install waiting a quarter of an hour; each worker still pauses between requests
            nonlocal done
            while queue:
                r = queue.pop(0)
                try:
                    body = await self._async_get_json(session, r)
                except (aiohttp.ClientError, TimeoutError, OpenMeteoError) as err:
                    LOGGER.warning("weather archive %s not fetched: %s", r.key, err)
                else:
                    await self.hass.async_add_executor_job(self.store.put_response, r.key, r.kind, body,
                                                           r.first_day, r.last_day, r.final)
                done += 1
                self.progress = f"downloading weather archives ({done}/{len(todo)})"
                await asyncio.sleep(REQUEST_PAUSE_S)

        self.progress = f"downloading weather archives (0/{len(todo)})"
        await asyncio.gather(*(worker() for _ in range(min(ARCHIVE_WORKERS, len(todo)))))
        for kind in ("hf", "pr", "sat"):
            await self.hass.async_add_executor_job(self.store.drop_responses, [r.key for r in reqs], kind)

    async def _async_train(self) -> dict:
        await self.hass.async_add_executor_job(self.store.set_meta, "system", self.system.to_dict())
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", str(TRAIN_JOB), self.store_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
        except OSError as err:
            LOGGER.warning("cannot start the training process (%s); training inside Home Assistant", err)
            from .engine import train_job  # noqa: PLC0415
            return await self.hass.async_add_executor_job(train_job.run, self.store_path)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), TRAIN_TIMEOUT_S)
        except (TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0:
            raise TrainingError(err.decode(errors="replace")[-800:])
        return json.loads(out.decode().strip().splitlines()[-1])
