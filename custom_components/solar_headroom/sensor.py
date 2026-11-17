"""Sensors: potential now, the solar bank (today, left today, tomorrow) and the learning status."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import EntityCategory, UnitOfEnergy, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import STATUSES
from .coordinator import SolarHeadroomConfigEntry, SolarHeadroomCoordinator
from .entity import SolarHeadroomEntity

PARALLEL_UPDATES = 0


def _get(data: dict | None, *path: str) -> Any:
    for p in path:
        if not isinstance(data, dict):
            return None
        data = data.get(p)
    return data


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _range(lo_hi) -> dict[str, float]:
    return {"p10": lo_hi[0], "p90": lo_hi[1]} if lo_hi else {}


def _now_attrs(c: SolarHeadroomCoordinator) -> dict:
    now = _get(c.data, "now") or {}
    return {"basis": now.get("basis"), "clear_sky_power": now.get("clear_sky_w"),
            "satellite_clearness": now.get("satellite_k")}


def _today_attrs(c: SolarHeadroomCoordinator) -> dict:
    t = _get(c.data, "today") or {}
    return {**_range(t.get("day_range_kwh") or t.get("forecast_range_kwh")), "forecast": t.get("forecast_kwh"),
            "so_far": t.get("so_far_potential_kwh"), "used_so_far": t.get("so_far_used_kwh"),
            "clearness": t.get("clearness"), "basis": _get(c.data, "basis", "daily")}


def _left_attrs(c: SolarHeadroomCoordinator) -> dict:
    t = _get(c.data, "today") or {}
    return _range(t.get("left_range_kwh"))


def _tomorrow_attrs(c: SolarHeadroomCoordinator) -> dict:
    t = _get(c.data, "tomorrow") or {}
    return {"p10": t.get("p10_kwh"), "p90": t.get("p90_kwh"), "clearness": t.get("clearness"),
            "basis": _get(c.data, "basis", "daily")}


def _status_attrs(c: SolarHeadroomCoordinator) -> dict:
    rep = c.last_report or {}
    res = c.result or {}
    trained = res.get("trained_at")
    return {"progress": c.progress or None, "days_of_data": rep.get("days_of_data"),
            "last_trained": datetime.fromtimestamp(trained, UTC).isoformat() if trained else None,
            "forecast_model": _get(res, "champion", "daily"), "nowcast_model": _get(res, "champion", "nowcast"),
            "satellite": c.sats.get("live") if _get(res, "satellite", "used") else None,
            "daily_holdout": rep.get("daily_holdout"), "nowcast_holdout": rep.get("nowcast_holdout"),
            "last_error": c.last_error}


@dataclass(frozen=True, kw_only=True)
class HeadroomSensorDescription(SensorEntityDescription):
    value: Callable[[SolarHeadroomCoordinator], Any]
    attrs: Callable[[SolarHeadroomCoordinator], dict] = lambda _c: {}


SENSORS = (
    HeadroomSensorDescription(
        key="potential_now", device_class=SensorDeviceClass.POWER, native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=0,
        value=lambda c: _get(c.data, "now", "w"), attrs=_now_attrs),
    HeadroomSensorDescription(
        key="bank_today", device_class=SensorDeviceClass.ENERGY, native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        suggested_display_precision=1,
        value=lambda c: _first(_get(c.data, "today", "day_estimate_kwh"), _get(c.data, "today", "forecast_kwh")),
        attrs=_today_attrs),
    HeadroomSensorDescription(
        key="bank_left_today", device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR, suggested_display_precision=1,
        value=lambda c: _get(c.data, "today", "left_kwh"), attrs=_left_attrs),
    HeadroomSensorDescription(
        key="forecast_tomorrow", device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR, suggested_display_precision=1,
        value=lambda c: _get(c.data, "tomorrow", "kwh"), attrs=_tomorrow_attrs),
    HeadroomSensorDescription(
        key="status", device_class=SensorDeviceClass.ENUM, options=STATUSES, entity_category=EntityCategory.DIAGNOSTIC,
        value=lambda c: c.status, attrs=_status_attrs),
)


async def async_setup_entry(hass: HomeAssistant, entry: SolarHeadroomConfigEntry,
                            async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    async_add_entities(HeadroomSensor(entry.runtime_data, d) for d in SENSORS)


class HeadroomSensor(SolarHeadroomEntity, SensorEntity):
    entity_description: HeadroomSensorDescription
    # holdout reports and errors change often and say nothing over time: keep them out of the recorder
    _unrecorded_attributes = frozenset({"daily_holdout", "nowcast_holdout", "last_error", "progress"})

    def __init__(self, coordinator: SolarHeadroomCoordinator, description: HeadroomSensorDescription) -> None:
        super().__init__(coordinator, description.key)
        self.entity_description = description

    @property
    def available(self) -> bool:
        if self.entity_description.key == "status":
            return True
        return super().available and self.native_value is not None

    @property
    def native_value(self) -> Any:
        return self.entity_description.value(self.coordinator)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {k: v for k, v in self.entity_description.attrs(self.coordinator).items() if v is not None}
