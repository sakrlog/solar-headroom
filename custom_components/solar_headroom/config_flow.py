"""Config flow: solar power sensor(s) -> panels -> does the inverter hold back? -> battery / grid sensors ->
history check (with a sign check of battery and grid power). Panel groups are subentries."""
from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import probatio

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import selector

from .const import (
    CONF_AZIMUTH,
    CONF_BATTERY_INVERTED,
    CONF_BATTERY_POWER,
    CONF_BATTERY_SOC,
    CONF_EXPORT_LIMIT,
    CONF_GENERATOR_POWER,
    CONF_GRID_INVERTED,
    CONF_GRID_POWER,
    CONF_GRID_VOLTAGE,
    CONF_KIND,
    CONF_KWP,
    CONF_PV_POWER,
    CONF_PV_VOLTAGE,
    CONF_TILT,
    DEFAULT_AZIMUTH,
    DEFAULT_TILT,
    DOMAIN,
    KIND_EXPORT_LIMITED,
    KIND_OFF_GRID,
    KIND_UNLIMITED,
    KINDS,
    LOGGER,
    MAX_PLANES,
    SENSOR_ROLES,
    SUBENTRY_TYPE_PLANE,
)
from .coordinator import UNITS, entry_conf, roles_from_conf
from .engine import pipeline as P

SIGN_DAYS = 60                               # hourly history the sign check looks at


def _number(lo: float, hi: float, step: float, unit: str) -> probatio.All:
    return probatio.All(selector.NumberSelector(selector.NumberSelectorConfig(
        min=lo, max=hi, step=step, unit_of_measurement=unit, mode=selector.NumberSelectorMode.BOX)),
        probatio.Coerce(float))


def _entity(device_class: str, multiple: bool = False) -> selector.EntitySelector:
    return selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor", device_class=device_class,
                                                                 multiple=multiple))


PLANE_SCHEMA = probatio.Schema({
    probatio.Required(CONF_KWP): _number(0.05, 100000, 0.01, "kWp"),
    probatio.Required(CONF_TILT): _number(0, 90, 1, "°"),
    probatio.Required(CONF_AZIMUTH): _number(0, 360, 1, "°"),
})
PV_SELECTOR = _entity("power", multiple=True)
KIND_SELECTOR = selector.SelectSelector(selector.SelectSelectorConfig(
    options=KINDS, translation_key=CONF_KIND, mode=selector.SelectSelectorMode.LIST))
HOLDBACK_SCHEMA = probatio.Schema({
    probatio.Required(CONF_KIND): KIND_SELECTOR,
    probatio.Optional(CONF_EXPORT_LIMIT): _number(0, 1_000_000, 10, "W"),
})
SENSOR_SELECTORS = {CONF_BATTERY_POWER: _entity("power"), CONF_BATTERY_SOC: _entity("battery"),
                    CONF_GRID_POWER: _entity("power"), CONF_GRID_VOLTAGE: _entity("voltage"),
                    CONF_GENERATOR_POWER: _entity("power"), CONF_PV_VOLTAGE: _entity("voltage")}
# which sensors each kind asks for; the first group is required
KIND_SENSORS = {
    KIND_EXPORT_LIMITED: ([CONF_GRID_POWER], [CONF_BATTERY_POWER, CONF_BATTERY_SOC, CONF_PV_VOLTAGE]),
    KIND_OFF_GRID: ([CONF_BATTERY_POWER], [CONF_BATTERY_SOC, CONF_GRID_VOLTAGE, CONF_GRID_POWER,
                                           CONF_GENERATOR_POWER, CONF_PV_VOLTAGE]),
}
SENSOR_KEYS = list(SENSOR_ROLES)


def sensors_schema(kind: str) -> probatio.Schema:
    required, optional = KIND_SENSORS[kind]
    fields: dict = {}
    for key in (*required, *optional):
        fields[(probatio.Required if key in required else probatio.Optional)(key)] = SENSOR_SELECTORS[key]
        inverted = SENSOR_ROLES[key][1]
        if inverted:
            fields[probatio.Optional(inverted, default=False)] = selector.BooleanSelector()
    return probatio.Schema(fields)


def plane_title(data: dict) -> str:
    return f"{data[CONF_KWP]:g} kWp · tilt {data[CONF_TILT]:g}° · azimuth {data[CONF_AZIMUTH]:g}°"


def _power_source(src: dict) -> tuple[str | None, bool]:
    """(sensor, inverted) of an Energy-dashboard power setting; two-sensor setups use the combined one."""
    pc = src.get("power_config") or {}
    if str(pc.get("stat_rate", "")).startswith("sensor."):
        return pc["stat_rate"], False
    if str(pc.get("stat_rate_inverted", "")).startswith("sensor."):
        return pc["stat_rate_inverted"], True
    rate = src.get("stat_rate")
    return (rate, False) if str(rate or "").startswith("sensor.") else (None, False)


async def energy_prefill(hass: HomeAssistant) -> dict[str, Any]:
    """Solar, battery and grid sensors already chosen in the Energy dashboard (pre-fill only)."""
    try:
        from homeassistant.components.energy.data import async_get_manager  # noqa: PLC0415

        manager = await async_get_manager(hass)
    except Exception:  # noqa: BLE001 - the Energy dashboard is optional
        return {}
    sources = (manager.data or {}).get("energy_sources", [])
    out: dict[str, Any] = {CONF_PV_POWER: [s["stat_rate"] for s in sources if s.get("type") == "solar"
                                           and str(s.get("stat_rate", "")).startswith("sensor.")]}
    for src in sources:
        if src.get("type") == "battery" and CONF_BATTERY_POWER not in out:
            sensor, inverted = _power_source(src)
            if sensor:
                out[CONF_BATTERY_POWER], out[CONF_BATTERY_INVERTED] = sensor, inverted
            if str(src.get("stat_soc", "")).startswith("sensor."):
                out[CONF_BATTERY_SOC] = src["stat_soc"]
        elif src.get("type") == "grid" and CONF_GRID_POWER not in out:
            sensor, inverted = _power_source(src)
            if sensor:
                out[CONF_GRID_POWER], out[CONF_GRID_INVERTED] = sensor, inverted
    return out


async def energy_solar_sensors(hass: HomeAssistant) -> list[str]:
    return (await energy_prefill(hass)).get(CONF_PV_POWER, [])


async def history_months(hass: HomeAssistant, entity_ids: list[str]) -> float:
    """How far back Home Assistant's long-term statistics of the sensors go (months, 0 = none)."""
    rows = await get_instance(hass).async_add_executor_job(
        statistics_during_period, hass, datetime.fromtimestamp(0, UTC), None, set(entity_ids), "month", None,
        {"mean"})
    starts = [r[0]["start"] for r in rows.values() if r]
    if not starts:
        return 0.0
    first = min(s.timestamp() if isinstance(s, datetime) else float(s) for s in starts)
    return max(0.0, (time.time() - first) / (30.44 * 86400))


async def check_signs(hass: HomeAssistant, conf: dict) -> dict[str, dict]:
    """engine.pipeline.sign_checks on the last SIGN_DAYS of hourly statistics."""
    roles = roles_from_conf(conf)
    if "bat_w" not in roles and "grid_w" not in roles:
        return {}
    system = P.System(0.0, 0.0, 0.0, "UTC", (), roles, conf.get(CONF_KIND, KIND_UNLIMITED))
    t1 = time.time() // 3600 * 3600
    t0 = t1 - SIGN_DAYS * 86400
    rows = await get_instance(hass).async_add_executor_job(
        statistics_during_period, hass, datetime.fromtimestamp(t0, UTC), None, set(system.statistic_ids()),
        "hour", UNITS, {"mean"})
    plain = {sid: [{"start": r["start"].timestamp() if isinstance(r["start"], datetime) else float(r["start"]),
                    "mean": r.get("mean"), "min": r.get("mean"), "max": r.get("mean")} for r in rs]
             for sid, rs in rows.items()}
    return await hass.async_add_executor_job(P.sign_checks, system, plain, t0, t1)


def apply_signs(conf: dict, checks: dict[str, dict]) -> list[str]:
    """Flip sensors whose sign looked reversed; returns one line per checked sensor for the user."""
    lines = []
    for key, role, inverted, what in ((CONF_BATTERY_POWER, "bat_w", CONF_BATTERY_INVERTED,
                                       "Battery power (+ = discharging)"),
                                      (CONF_GRID_POWER, "grid_w", CONF_GRID_INVERTED,
                                       "Grid power (+ = from the grid)")):
        c = checks.get(role)
        if not c or not c["hours"] or not conf.get(key):
            continue
        if c["ok"] is False:
            conf[inverted] = not conf.get(inverted, False)
            lines.append(f"- {what}: looked reversed in the last {c['hours']} hours, so it will be read the "
                         "other way round (you can change this in the options).")
        elif c["ok"]:
            lines.append(f"- {what}: sign checked on {c['hours']} hours.")
        else:
            lines.append(f"- {what}: could not be checked yet; set as you chose.")
    return lines


def history_span(months: float) -> str:
    years, rest = divmod(round(months), 12)
    if years:
        return f"{years} year{'s' if years != 1 else ''} {rest} month{'s' if rest != 1 else ''}"
    return f"{max(1, round(months))} month{'s' if round(months) != 1 else ''}"


def _clean(conf: dict) -> dict:
    """Keep only what the kind uses."""
    kind = conf.get(CONF_KIND, KIND_UNLIMITED)
    used = set() if kind == KIND_UNLIMITED else {k for group in KIND_SENSORS[kind] for k in group}
    out = {CONF_PV_POWER: list(conf[CONF_PV_POWER]), CONF_KIND: kind}
    if kind == KIND_EXPORT_LIMITED:
        out[CONF_EXPORT_LIMIT] = float(conf.get(CONF_EXPORT_LIMIT) or 0.0)
    for key in used:
        if conf.get(key):
            out[key] = conf[key]
            inverted = SENSOR_ROLES[key][1]
            if inverted:
                out[inverted] = bool(conf.get(inverted, False))
    return out


class _HoldbackSteps:
    """The 'does the inverter hold back' and sensor steps, shared by the config and options flows."""

    _conf: dict[str, Any]
    _prefill: dict[str, Any]

    async def async_step_holdback(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            self._conf[CONF_KIND] = user_input[CONF_KIND]
            self._conf[CONF_EXPORT_LIMIT] = float(user_input.get(CONF_EXPORT_LIMIT) or 0.0)
            if user_input[CONF_KIND] == KIND_UNLIMITED:
                return await self._async_after_sensors()
            return await self.async_step_sensors()
        guess = KIND_OFF_GRID if self._prefill.get(CONF_BATTERY_POWER) and not self._prefill.get(CONF_GRID_POWER) \
            else KIND_UNLIMITED
        suggested = {CONF_KIND: self._conf.get(CONF_KIND, guess),
                     CONF_EXPORT_LIMIT: self._conf.get(CONF_EXPORT_LIMIT, 0.0)}
        return self.async_show_form(step_id="holdback",
                                    data_schema=self.add_suggested_values_to_schema(HOLDBACK_SCHEMA, suggested))

    async def async_step_sensors(self, user_input: dict[str, Any] | None = None):
        kind = self._conf[CONF_KIND]
        errors: dict[str, str] = {}
        if user_input is not None:
            for key in KIND_SENSORS[kind][0]:
                if not user_input.get(key):
                    errors[key] = "required"
            if not errors:
                for key in SENSOR_KEYS:
                    self._conf.pop(key, None)
                self._conf.update(user_input)
                return await self._async_after_sensors()
        suggested = {k: v for k, v in {**self._prefill, **self._conf}.items() if k in SENSOR_KEYS
                     or k in (CONF_BATTERY_INVERTED, CONF_GRID_INVERTED)}
        return self.async_show_form(step_id="sensors", errors=errors, data_schema=self.add_suggested_values_to_schema(
            sensors_schema(kind), suggested))

    async def _async_after_sensors(self):
        raise NotImplementedError


class SolarHeadroomConfigFlow(_HoldbackSteps, ConfigFlow, domain=DOMAIN):
    VERSION = 1

    def __init__(self) -> None:
        self._conf: dict[str, Any] = {}
        self._prefill: dict[str, Any] = {}
        self._plane: dict[str, float] = {}

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> SolarHeadroomOptionsFlow:
        return SolarHeadroomOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(cls, config_entry: ConfigEntry) -> dict[str, type[ConfigSubentryFlow]]:
        return {SUBENTRY_TYPE_PLANE: PlaneSubentryFlowHandler}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            sensors = list(user_input.get(CONF_PV_POWER) or [])
            if not sensors:
                errors[CONF_PV_POWER] = "no_sensor"
            else:
                await self.async_set_unique_id("|".join(sorted(sensors)))
                self._abort_if_unique_id_configured()
                self._conf[CONF_PV_POWER] = sensors
                return await self.async_step_panels()
        self._prefill = await energy_prefill(self.hass)
        schema = probatio.Schema({probatio.Required(CONF_PV_POWER): PV_SELECTOR})
        suggested = {CONF_PV_POWER: self._prefill.get(CONF_PV_POWER, [])}
        return self.async_show_form(step_id="user", data_schema=self.add_suggested_values_to_schema(schema, suggested),
                                    errors=errors)

    async def async_step_panels(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            self._plane = {k: float(user_input[k]) for k in (CONF_KWP, CONF_TILT, CONF_AZIMUTH)}
            return await self.async_step_holdback()
        return self.async_show_form(step_id="panels", data_schema=self.add_suggested_values_to_schema(
            PLANE_SCHEMA, {CONF_TILT: DEFAULT_TILT, CONF_AZIMUTH: DEFAULT_AZIMUTH}))

    async def _async_after_sensors(self) -> ConfigFlowResult:
        return await self.async_step_history()

    def _create(self) -> ConfigFlowResult:
        return self.async_create_entry(
            title="Solar Headroom", data=_clean(self._conf),
            subentries=[{"subentry_type": SUBENTRY_TYPE_PLANE, "data": self._plane,
                         "title": plane_title(self._plane), "unique_id": None}])

    async def async_step_history(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self._create()
        try:
            months = await history_months(self.hass, self._conf[CONF_PV_POWER])
        except Exception:  # noqa: BLE001 - only informs the user
            LOGGER.exception("could not read the statistics of %s", self._conf[CONF_PV_POWER])
            months = 0.0
        if months < 0.1:
            return await self.async_step_no_history()
        try:
            signs = apply_signs(self._conf, await check_signs(self.hass, self._conf))
        except Exception:  # noqa: BLE001 - only a check
            LOGGER.exception("sign check failed")
            signs = []
        return self.async_show_form(step_id="history", data_schema=probatio.Schema({}),
                                    description_placeholders={"history": history_span(months),
                                                              "signs": "\n".join(signs)})

    async def async_step_no_history(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self._create()
        return self.async_show_form(step_id="no_history", data_schema=probatio.Schema({}))


class SolarHeadroomOptionsFlow(_HoldbackSteps, OptionsFlow):
    """Solar sensor(s), whether and how the inverter holds back, and the battery / grid sensors. Saved
    options are complete (they replace the setup data, see coordinator.entry_conf)."""

    def __init__(self) -> None:
        self._conf: dict[str, Any] = {}
        self._prefill: dict[str, Any] = {}
        self._signs: list[str] = []

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        current = entry_conf(self.config_entry)
        if user_input is not None:
            if not user_input.get(CONF_PV_POWER):
                errors[CONF_PV_POWER] = "no_sensor"
            else:
                self._conf = {**current, CONF_PV_POWER: list(user_input[CONF_PV_POWER])}
                return await self.async_step_holdback()
        self._prefill = await energy_prefill(self.hass)
        schema = probatio.Schema({probatio.Required(CONF_PV_POWER): PV_SELECTOR})
        return self.async_show_form(step_id="init", errors=errors, data_schema=self.add_suggested_values_to_schema(
            schema, {CONF_PV_POWER: current.get(CONF_PV_POWER, [])}))

    async def _async_after_sensors(self) -> ConfigFlowResult:
        try:
            self._signs = apply_signs(self._conf, await check_signs(self.hass, self._conf))
        except Exception:  # noqa: BLE001 - only a check
            LOGGER.exception("sign check failed")
            self._signs = []
        if not self._signs:
            return self.async_create_entry(data=_clean(self._conf))
        return await self.async_step_signs()

    async def async_step_signs(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=_clean(self._conf))
        return self.async_show_form(step_id="signs", data_schema=probatio.Schema({}),
                                    description_placeholders={"signs": "\n".join(self._signs)})


class PlaneSubentryFlowHandler(ConfigSubentryFlow):
    """Add or edit a panel group (panels sharing one direction and tilt)."""

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        entry = self._get_entry()
        if len(entry.get_subentries_of_type(SUBENTRY_TYPE_PLANE)) >= MAX_PLANES:
            return self.async_abort(reason="max_planes")
        if user_input is not None:
            data = {k: float(user_input[k]) for k in (CONF_KWP, CONF_TILT, CONF_AZIMUTH)}
            return self.async_create_entry(title=plane_title(data), data=data)
        return self.async_show_form(step_id="user", data_schema=self.add_suggested_values_to_schema(
            PLANE_SCHEMA, {CONF_TILT: DEFAULT_TILT, CONF_AZIMUTH: DEFAULT_AZIMUTH}))

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        subentry = self._get_reconfigure_subentry()
        if user_input is not None:
            data = {k: float(user_input[k]) for k in (CONF_KWP, CONF_TILT, CONF_AZIMUTH)}
            return self.async_update_and_abort(self._get_entry(), subentry, data=data, title=plane_title(data))
        return self.async_show_form(step_id="reconfigure", data_schema=self.add_suggested_values_to_schema(
            PLANE_SCHEMA, dict(subentry.data)))
