"""Solar Headroom: what your solar panels could deliver, learned from Home Assistant's own history."""
from __future__ import annotations

import shutil
from pathlib import Path

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryError

from .const import DOMAIN, SUBENTRY_TYPE_PLANE
from .coordinator import SolarHeadroomConfigEntry, SolarHeadroomCoordinator

PLATFORMS = [Platform.BUTTON, Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: SolarHeadroomConfigEntry) -> bool:
    if not entry.get_subentries_of_type(SUBENTRY_TYPE_PLANE):
        raise ConfigEntryError(translation_domain=DOMAIN, translation_key="no_plane")
    coordinator = SolarHeadroomCoordinator(hass, entry)
    await coordinator.async_setup()
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: SolarHeadroomConfigEntry) -> None:
    """Options or panel groups changed."""
    hass.config_entries.async_schedule_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: SolarHeadroomConfigEntry) -> bool:
    if (tracker := entry.runtime_data.tracker) is not None:
        await tracker.async_stop()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: SolarHeadroomConfigEntry) -> None:
    """Delete the entry's store (history copy, weather archives, models)."""
    base = Path(hass.config.path(DOMAIN))

    def _remove() -> None:
        for p in base.glob(f"{entry.entry_id}.sqlite*"):
            p.unlink(missing_ok=True)
        if base.is_dir() and not any(base.iterdir()):
            shutil.rmtree(base, ignore_errors=True)
    await hass.async_add_executor_job(_remove)
