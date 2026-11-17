"""Sensors: potential now (live), untapped solar and wasted today (systems that hold back), the solar bank
(today, left today, tomorrow) and the learning status."""
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

from .const import KIND_UNLIMITED, STATUSES
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


def _live(c: SolarHeadroomCoordinator) -> dict | None:
    """The live estimate when it has one (a PV reading, or a sky model to estimate from)."""
    est = c.tracker.now if c.tracker is not None else None
    return est if est and est.get("w") is not None and est.get("basis") != "unknown" else None


def _potential(c: SolarHeadroomCoordinator) -> float | None:
    est = _live(c)
    w = est["w"] if est else _get(c.data, "now", "w")
    return None if w is None else round(w)


def _now_attrs(c: SolarHeadroomCoordinator) -> dict:
    sky = _get(c.data, "now") or {}
    est = _live(c) or {}
    basis = est.get("basis")
    if basis in (None, "sky_model"):
        basis = sky.get("basis")            # learned_satellite / learned_forecast / physics / night
    return {"basis": basis, "regime": est.get("regime"), "reason": est.get("reason") or None,
            "sky_model_power": sky.get("w"), "clear_sky_power": _round(est.get("clear_sky_w", sky.get("clear_sky_w"))),
            "minutes_since_full_output": est.get("minutes_since_full"), "satellite_clearness": sky.get("satellite_k")}


def _round(x: float | None, nd: int = 0) -> float | None:
    return None if x is None else round(x, nd) if nd else round(x)


def _untapped(c: SolarHeadroomCoordinator) -> float | None:
    est = _live(c)
    return None if est is None or est.get("untapped_w") is None else round(est["untapped_w"])


def _wasted(c: SolarHeadroomCoordinator) -> float | None:
    return None if c.tracker is None else round(c.tracker.wasted_today_kwh(), 2)


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
    live: bool = False                       # also written by the live tracker (seconds), not only every 5 min
    holds_back_only: bool = False            # only for systems that can hold back


SENSORS = (
    HeadroomSensorDescription(
        key="potential_now", device_class=SensorDeviceClass.POWER, native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=0,
        value=_potential, attrs=_now_attrs, live=True),
    HeadroomSensorDescription(
        key="untapped_now", device_class=SensorDeviceClass.POWER, native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT, suggested_display_precision=0,
        value=_untapped, live=True, holds_back_only=True),
    HeadroomSensorDescription(
        key="wasted_today", device_class=SensorDeviceClass.ENERGY,
        native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR, state_class=SensorStateClass.TOTAL_INCREASING,
        suggested_display_precision=1,
        value=_wasted, live=True, holds_back_only=True),
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
    c = entry.runtime_data
    holds_back = c.system.kind != KIND_UNLIMITED
    async_add_entities(HeadroomSensor(c, d) for d in SENSORS if holds_back or not d.holds_back_only)


class HeadroomSensor(SolarHeadroomEntity, SensorEntity):
    entity_description: HeadroomSensorDescription
    # change often and say nothing over time: keep them out of the recorder
    _unrecorded_attributes = frozenset({"daily_holdout", "nowcast_holdout", "last_error", "progress", "reason",
                                        "sky_model_power", "clear_sky_power", "minutes_since_full_output"})

    def __init__(self, coordinator: SolarHeadroomCoordinator, description: HeadroomSensorDescription) -> None:
        super().__init__(coordinator, description.key)
        self.entity_description = description

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if self.entity_description.live and self.coordinator.tracker is not None:
            self.async_on_remove(self.coordinator.tracker.async_add_listener(self.async_write_ha_state))

    @property
    def available(self) -> bool:
        if self.entity_description.key == "status":
            return True
        if self.entity_description.live and self.native_value is not None:
            return True                      # measured / estimated even while the forecast is down
        return super().available and self.native_value is not None

    @property
    def native_value(self) -> Any:
        return self.entity_description.value(self.coordinator)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {k: v for k, v in self.entity_description.attrs(self.coordinator).items() if v is not None}
