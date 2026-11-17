"""Live layer: follows the PV, battery and grid sensors and keeps the potential estimate current.

Every state change of a configured sensor (debounced) and a 30-second tick step engine.live: the
regime from the newest readings, the potential (measured at full output, estimated while held back)
and today's log of potential and used energy. Live sensors are written at most every WRITE_EVERY_S,
sooner on a meaningful change. The last full-output moment and today's log survive restarts.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfElectricPotential, UnitOfPower
from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, HomeAssistant, State, callback
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.storage import Store as HAStore
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import ElectricPotentialConverter, PowerConverter

from .const import CONF_KIND, CONF_PV_POWER, DOMAIN, KIND_UNLIMITED, LOGGER, SENSOR_ROLES
from .engine import live as L
from .engine import pipeline as P

STORAGE_VERSION = 1
DEBOUNCE_S = 1.0                 # sensors of one inverter poll arrive as a burst
TICK = timedelta(seconds=30)     # holds, windows and the day log advance without state changes too
WRITE_EVERY_S = 30.0
MIN_WRITE_GAP_S = 2.0
SAVE_DELAY_S = 300
CONVERTERS = {"pv_w": PowerConverter, "bat_w": PowerConverter, "grid_w": PowerConverter, "gen_w": PowerConverter,
              "grid_v": ElectricPotentialConverter, "pv_v": ElectricPotentialConverter}
BASE_UNITS = {PowerConverter: UnitOfPower.WATT, ElectricPotentialConverter: UnitOfElectricPotential.VOLT}


def sources_from_conf(conf: dict) -> dict[str, tuple[str, float]]:
    """entity_id -> (role, scale) of the sensors the live estimate follows (PV parts are summed)."""
    out = {eid: ("pv_w", 1.0) for eid in conf[CONF_PV_POWER]}
    if conf.get(CONF_KIND, KIND_UNLIMITED) == KIND_UNLIMITED:
        return out
    for key, (role, inverted) in SENSOR_ROLES.items():
        if conf.get(key):
            out.setdefault(conf[key], (role, -1.0 if inverted and conf.get(inverted) else 1.0))
    return out


def state_value(state: State | None, role: str) -> float | None:
    """A sensor state in the engine's units (W, V, %); None when unavailable or not a number."""
    if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        return None
    try:
        value = float(state.state)
    except ValueError:
        return None
    conv = CONVERTERS.get(role)
    unit = state.attributes.get("unit_of_measurement")
    if conv is not None and unit and unit != BASE_UNITS[conv]:
        if unit not in conv.VALID_UNITS:
            return None
        value = conv.convert(value, unit, BASE_UNITS[conv])
    return value


class LiveTracker:
    """One per PV system; owned by the coordinator."""

    def __init__(self, hass: HomeAssistant, entry_id: str, system: P.System, conf: dict,
                 learned: dict | None) -> None:
        self.hass = hass
        self.system = system
        self.sources = sources_from_conf(conf)
        self.pv_ids = [e for e, (role, _) in self.sources.items() if role == "pv_w"]
        self.estimator = L.LiveEstimator.for_system(system.kind, system.export_limit_w, learned)
        self.day_log = L.DayLog()
        self.now: dict | None = None
        self.k_sky: float | None = None
        self.t_air: Any = 20.0
        self._parts: dict[str, float | None] = {}
        self._unavailable: set[str] = set()
        self.missing_s: dict[str, float] = {eid: 0.0 for eid in self.sources}   # daylight s without a reading
        self._last_step: float | None = None
        self._store: HAStore = HAStore(hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}.live")
        self._listeners: list[Callable[[], None]] = []
        self._pending: CALLBACK_TYPE | None = None
        self._written: tuple[float, float | None, str | None] | None = None
        self._p_cs: tuple[float, float] | None = None              # (minute, clear-sky W)

    # ------------------------------------------------------------ lifecycle
    async def async_start(self, entry: ConfigEntry) -> None:
        saved = await self._store.async_load() or {}
        self.estimator.restore(saved.get("estimator"))
        self.day_log = L.DayLog.from_dict(saved.get("day_log"))
        now = time.time()
        for eid in self.sources:
            state = self.hass.states.get(eid)
            if state is None:
                self._unavailable.add(eid)
            self._read(state, now)
        entry.async_on_unload(async_track_state_change_event(self.hass, list(self.sources), self._async_on_event))
        entry.async_on_unload(async_track_time_interval(self.hass, self._async_tick, TICK))
        entry.async_on_unload(self._cancel_pending)
        self._step(now)

    async def async_stop(self) -> None:
        await self._store.async_save(self._data())

    @callback
    def _cancel_pending(self) -> None:
        if self._pending is not None:
            self._pending()
            self._pending = None

    def _data(self) -> dict:
        return {"estimator": self.estimator.to_dict(), "day_log": self.day_log.to_dict()}

    # ------------------------------------------------------------ inputs from the coordinator
    @callback
    def set_sky(self, k_sky: float | None, t_air: Any) -> None:
        """The sky model's clearness for the current quarter hour and the air temperature source."""
        self.k_sky, self.t_air = k_sky, t_air
        self._p_cs = None

    @callback
    def set_learned(self, learned: dict | None) -> None:
        """Thresholds from a new training (charge limit, PV-voltage band); the readings are kept."""
        learned = learned or {}
        self.estimator.cfg = replace(self.estimator.cfg, charge_limit_w=learned.get("charge_limit_w"),
                                     v_full_max=learned.get("v_full_max"))

    def day_log_copy(self) -> L.DayLog:
        """For a worker thread: the log keeps changing in the event loop."""
        return L.DayLog({b: list(v) for b, v in self.day_log.buckets.items()})

    def wasted_today_kwh(self) -> float:
        midnight = dt_util.start_of_local_day().timestamp()
        return self.day_log.wasted_kwh(midnight, midnight + 86400)

    @callback
    def async_add_listener(self, update: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(update)

        @callback
        def remove() -> None:
            self._listeners.remove(update)
        return remove

    # ------------------------------------------------------------ sensor updates
    def _read(self, state: State | None, now: float) -> None:
        if state is None:
            return
        role, scale = self.sources[state.entity_id]
        value = state_value(state, role)
        if value is None:
            self._unavailable.add(state.entity_id)
        else:
            self._unavailable.discard(state.entity_id)
            self.missing_s[state.entity_id] = 0.0
        ts = min(state.last_updated.timestamp(), now)
        if role == "pv_w":
            self._parts[state.entity_id] = value
            parts = [self._parts.get(e) for e in self.pv_ids]
            self.estimator.update("pv_w", None if any(p is None for p in parts) else sum(parts), ts)
        else:
            self.estimator.update(role, None if value is None else value * scale, ts)

    @callback
    def _async_on_event(self, event: Event[EventStateChangedData]) -> None:
        if event.data["new_state"] is None:                      # removed
            self._unavailable.add(event.data["entity_id"])
        self._read(event.data["new_state"], time.time())
        if self._pending is None:
            self._pending = async_call_later(self.hass, DEBOUNCE_S, self._async_debounced)

    @callback
    def _async_debounced(self, _now: Any) -> None:
        self._pending = None
        self._step(time.time())

    @callback
    def _async_tick(self, _now: Any) -> None:
        self._step(time.time())

    def _clear_sky(self, now: float) -> float:
        minute = now // 60.0
        if self._p_cs is None or self._p_cs[0] != minute:
            self._p_cs = (minute, P.clear_sky_now(self.system, now, self.t_air))
        return self._p_cs[1]

    @callback
    def _step(self, now: float) -> None:
        prev, self._last_step = self._last_step, now
        try:
            p_cs = self._clear_sky(now)
            est = self.estimator.estimate(now, p_cs, self.k_sky)
        except Exception:  # noqa: BLE001 - the live estimate must not take the integration down
            LOGGER.exception("live estimate failed")
            return
        if prev is not None and p_cs > 1.0:
            for eid in self._unavailable:
                self.missing_s[eid] += min(now - prev, L.MAX_STEP_S)
        counted = est["regime"] != L.UNKNOWN and est["w"] is not None
        self.day_log.step(now, est["w"] if counted else None, est["pv_w"] if counted else None)
        self.now = est
        self._store.async_delay_save(self._data, SAVE_DELAY_S)
        prev = self._written
        w = est["w"]
        if prev is not None:
            since = now - prev[0]
            changed = est["basis"] != prev[2] or (w is None) != (prev[1] is None) or (
                w is not None and abs(w - prev[1]) > max(50.0, 0.05 * abs(prev[1])))
            if since < WRITE_EVERY_S and not (changed and since >= MIN_WRITE_GAP_S):
                return
        self._written = (now, w, est["basis"])
        for update in list(self._listeners):
            update()
