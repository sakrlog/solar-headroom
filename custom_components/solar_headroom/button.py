"""Retrain button."""
from __future__ import annotations

from homeassistant.components.button import ButtonEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .coordinator import SolarHeadroomConfigEntry
from .entity import SolarHeadroomEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(hass: HomeAssistant, entry: SolarHeadroomConfigEntry,
                            async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    async_add_entities([RetrainButton(entry.runtime_data, "retrain")])


class RetrainButton(SolarHeadroomEntity, ButtonEntity):
    _attr_entity_category = EntityCategory.CONFIG

    async def async_press(self) -> None:
        self.coordinator.async_schedule_learning("button")
