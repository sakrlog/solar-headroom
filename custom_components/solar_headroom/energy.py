"""Energy dashboard: hourly solar forecast (Wh per hour start)."""
from __future__ import annotations

from datetime import UTC, datetime

from homeassistant.core import HomeAssistant

from .const import DOMAIN


async def async_get_solar_forecast(hass: HomeAssistant, config_entry_id: str) -> dict[str, dict[str, float]] | None:
    entry = hass.config_entries.async_get_entry(config_entry_id)
    if entry is None or entry.domain != DOMAIN or getattr(entry, "runtime_data", None) is None:
        return None
    data = entry.runtime_data.data or {}
    return {"wh_hours": {datetime.fromtimestamp(ts, UTC).isoformat(): round(w, 1)
                         for ts, w in sorted(data.get("hourly", {}).items())}}
