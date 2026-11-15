"""Base entity: one service device per PV system."""
from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import SolarHeadroomCoordinator


class SolarHeadroomEntity(CoordinatorEntity[SolarHeadroomCoordinator]):
    _attr_has_entity_name = True

    def __init__(self, coordinator: SolarHeadroomCoordinator, key: str) -> None:
        super().__init__(coordinator)
        entry = coordinator.config_entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)}, name=entry.title,
                                            entry_type=DeviceEntryType.SERVICE, manufacturer="Solar Headroom",
                                            model="Self-learning solar potential")
