"""Config flow: solar power sensor(s) -> panels -> history check. Panel groups are subentries."""
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
    CONF_KIND,
    CONF_KWP,
    CONF_PV_POWER,
    CONF_TILT,
    DEFAULT_AZIMUTH,
    DEFAULT_TILT,
    DOMAIN,
    LOGGER,
    MAX_PLANES,
    SUBENTRY_TYPE_PLANE,
)


def _number(lo: float, hi: float, step: float, unit: str) -> probatio.All:
    return probatio.All(selector.NumberSelector(selector.NumberSelectorConfig(
        min=lo, max=hi, step=step, unit_of_measurement=unit, mode=selector.NumberSelectorMode.BOX)),
        probatio.Coerce(float))


PLANE_SCHEMA = probatio.Schema({
    probatio.Required(CONF_KWP): _number(0.05, 100000, 0.01, "kWp"),
    probatio.Required(CONF_TILT): _number(0, 90, 1, "°"),
    probatio.Required(CONF_AZIMUTH): _number(0, 360, 1, "°"),
})
PV_SELECTOR = selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor", device_class="power",
                                                                    multiple=True))


def plane_title(data: dict) -> str:
    return f"{data[CONF_KWP]:g} kWp · tilt {data[CONF_TILT]:g}° · azimuth {data[CONF_AZIMUTH]:g}°"


async def energy_solar_sensors(hass: HomeAssistant) -> list[str]:
    """Solar power sensors already chosen in the Energy dashboard (pre-fill only)."""
    try:
        from homeassistant.components.energy.data import async_get_manager  # noqa: PLC0415

        manager = await async_get_manager(hass)
    except Exception:  # noqa: BLE001 - the Energy dashboard is optional
        return []
    sources = (manager.data or {}).get("energy_sources", [])
    return [s["stat_rate"] for s in sources if s.get("type") == "solar" and str(s.get("stat_rate", "")).startswith(
        "sensor.")]


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


class SolarHeadroomConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    def __init__(self) -> None:
        self._sensors: list[str] = []
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
                self._sensors = sensors
                return await self.async_step_panels()
        schema = probatio.Schema({probatio.Required(CONF_PV_POWER): PV_SELECTOR})
        suggested = {CONF_PV_POWER: await energy_solar_sensors(self.hass)}
        return self.async_show_form(step_id="user", data_schema=self.add_suggested_values_to_schema(schema, suggested),
                                    errors=errors)

    async def async_step_panels(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            self._plane = {k: float(user_input[k]) for k in (CONF_KWP, CONF_TILT, CONF_AZIMUTH)}
            return await self.async_step_history()
        return self.async_show_form(step_id="panels", data_schema=self.add_suggested_values_to_schema(
            PLANE_SCHEMA, {CONF_TILT: DEFAULT_TILT, CONF_AZIMUTH: DEFAULT_AZIMUTH}))

    def _create(self) -> ConfigFlowResult:
        return self.async_create_entry(
            title="Solar Headroom", data={CONF_PV_POWER: self._sensors, CONF_KIND: "unlimited"},
            subentries=[{"subentry_type": SUBENTRY_TYPE_PLANE, "data": self._plane,
                         "title": plane_title(self._plane), "unique_id": None}])

    async def async_step_history(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self._create()
        try:
            months = await history_months(self.hass, self._sensors)
        except Exception:  # noqa: BLE001 - only informs the user
            LOGGER.exception("could not read the statistics of %s", self._sensors)
            months = 0.0
        if months < 0.1:
            return await self.async_step_no_history()
        years, rest = divmod(round(months), 12)
        span = f"{years} year{'s' if years != 1 else ''} {rest} month{'s' if rest != 1 else ''}" if years else \
            f"{max(1, round(months))} month{'s' if round(months) != 1 else ''}"
        return self.async_show_form(step_id="history", data_schema=probatio.Schema({}),
                                    description_placeholders={"history": span})

    async def async_step_no_history(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self._create()
        return self.async_show_form(step_id="no_history", data_schema=probatio.Schema({}))


class SolarHeadroomOptionsFlow(OptionsFlow):
    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            if not user_input.get(CONF_PV_POWER):
                errors[CONF_PV_POWER] = "no_sensor"
            else:
                return self.async_create_entry(data={CONF_PV_POWER: list(user_input[CONF_PV_POWER])})
        current = {**self.config_entry.data, **self.config_entry.options}
        schema = probatio.Schema({probatio.Required(CONF_PV_POWER): PV_SELECTOR})
        return self.async_show_form(step_id="init", errors=errors, data_schema=self.add_suggested_values_to_schema(
            schema, {CONF_PV_POWER: current.get(CONF_PV_POWER, [])}))


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
