"""Coordinator: history sync, Open-Meteo, nightly training and the outputs every few minutes."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    CONF_AZIMUTH,
    CONF_KIND,
    CONF_KWP,
    CONF_PV_POWER,
    CONF_TILT,
    DOMAIN,
    LIVE_MAX_AGE_S,
    LIVE_STALE_S,
    LOGGER,
    REQUEST_PAUSE_S,
    REQUEST_TIMEOUT_S,
    RETRAIN_AFTER_S,
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

type SolarHeadroomConfigEntry = ConfigEntry[SolarHeadroomCoordinator]

TRAIN_JOB = Path(__file__).parent / "engine" / "train_job.py"
STAT_TYPES: set = {"mean", "min", "max"}
UNITS = {"power": UnitOfPower.WATT}


class OpenMeteoError(Exception):
    """Open-Meteo answered with an error."""


class TrainingError(Exception):
    """The training process failed."""


def system_from_entry(hass: HomeAssistant, entry: ConfigEntry) -> P.System:
    conf = {**entry.data, **entry.options}
    planes = tuple(S.Plane(float(s.data[CONF_TILT]), float(s.data[CONF_AZIMUTH]), float(s.data[CONF_KWP]))
                   for s in entry.get_subentries_of_type(SUBENTRY_TYPE_PLANE))
    roles = {"pv_w": [[{"statistic_id": eid}] for eid in conf[CONF_PV_POWER]]}
    return P.System(float(hass.config.latitude), float(hass.config.longitude), float(hass.config.elevation or 0.0),
                    str(hass.config.time_zone), planes, roles, conf.get(CONF_KIND, "unlimited"))


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
        seed = int(hashlib.sha256(entry.entry_id.encode()).hexdigest()[:8], 16)
        self.jitter_s = seed % 600                # installs spread their forecast fetches over 10 minutes
        self.train_minute = seed % 60
        self.status = STATUS_STARTING
        self.progress = ""
        self.last_error: str | None = None
        self.last_report: dict | None = None
        self._learning: asyncio.Task | None = None

    # ------------------------------------------------------------ setup
    async def async_setup(self) -> None:
        """Open the store, load the last training result, schedule learning."""
        self.store = await self.hass.async_add_executor_job(Store, self.store_path)
        self.result = await self.hass.async_add_executor_job(self.store.get_blob, "models")
        self._models = await self.hass.async_add_executor_job(P.load_models, self.result)
        if self.result:
            self.last_report = self.result.get("report")
            ready = (self.last_report or {}).get("status") == "ready"
            self.status = STATUS_READY if ready else STATUS_COLLECTING
        entry = self.config_entry
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
        rows5 = await self._async_today_rows()
        data = await self.hass.async_add_executor_job(P.outputs, self.system, self.result, self._live, now, rows5,
                                                      self._models)
        if now - self.live_at > LIVE_STALE_S and not self.learning:
            self.status = STATUS_DEGRADED
        return data

    async def _async_fetch_live(self) -> None:
        session = async_get_clientsession(self.hass)
        bodies = {}
        for r in P.live_requests(self.system):
            try:
                bodies[r.key] = await self._async_get_json(session, r.url, r.params)
            except (aiohttp.ClientError, TimeoutError, OpenMeteoError) as err:
                if "/best_match/" in r.key:
                    raise
                LOGGER.debug("ensemble forecast skipped: %s", err)    # the spread is optional
        self._live = await self.hass.async_add_executor_job(P.live_frames, self.system, bodies)
        self.live_at = time.time()
        if self.status == STATUS_DEGRADED and self.last_error and self.last_error.startswith("forecast"):
            self.status = STATUS_READY if self.result else STATUS_COLLECTING
            self.last_error = None

    async def _async_get_json(self, session: aiohttp.ClientSession, url: str, params: dict) -> dict:
        async with session.get(url, params={k: str(v) for k, v in params.items()},
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
            self.progress = "downloading weather archives"
            await self._async_sync_archives()
            self.progress = "training"
            self.async_update_listeners()
            self.last_report = await self._async_train()
            self.result = await self.hass.async_add_executor_job(self.store.get_blob, "models")
            self._models = await self.hass.async_add_executor_job(P.load_models, self.result)
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

    async def _async_sync_archives(self) -> None:
        spans = [await self.hass.async_add_executor_job(self.store.row_span, sid)
                 for sid in self.system.statistic_ids()]
        spans = [s for s in spans if s is not None]
        if not spans:
            return
        first_day = P.local_date(min(s[0] for s in spans), self.system.time_zone)
        reqs = P.archive_requests(self.system, first_day, dt_util.now().date())
        index = await self.hass.async_add_executor_job(self.store.response_index)
        todo = P.missing(reqs, index, time.time())
        session = async_get_clientsession(self.hass)
        for i, r in enumerate(todo, 1):
            self.progress = f"downloading weather archives ({i}/{len(todo)})"
            try:
                body = await self._async_get_json(session, r.url, r.params)
            except (aiohttp.ClientError, TimeoutError, OpenMeteoError) as err:
                LOGGER.warning("weather archive %s not fetched: %s", r.key, err)
                continue
            await self.hass.async_add_executor_job(self.store.put_response, r.key, r.kind, body, r.first_day,
                                                   r.last_day, r.final)
            await asyncio.sleep(REQUEST_PAUSE_S)
        for kind in ("hf", "pr"):
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
