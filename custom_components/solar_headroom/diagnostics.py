"""Diagnostics: learning state and recent accuracy, without the location."""
from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from .coordinator import SolarHeadroomConfigEntry


async def async_get_config_entry_diagnostics(hass: HomeAssistant, entry: SolarHeadroomConfigEntry) -> dict[str, Any]:
    c = entry.runtime_data
    system = c.system.to_dict()
    system["latitude"] = system["longitude"] = "**REDACTED**"
    system["altitude"] = round(system["altitude"], -2)
    res = c.result or {}
    data = c.data or {}
    size = await hass.async_add_executor_job(c.store.size_bytes) if c.store else None
    sat_meta = await hass.async_add_executor_job(c.store.get_meta, "satellite") if c.store else None
    return {
        "system": system,
        "status": c.status, "progress": c.progress, "last_error": c.last_error,
        "store_mb": None if size is None else round(size / 1e6, 2),
        "trained_at": res.get("trained_at"), "champion": res.get("champion"), "factor": res.get("factor"),
        "factor_daily": res.get("factor_daily"), "range": res.get("range"), "report": res.get("report"),
        "satellite": {k: v for k, v in (res.get("satellite") or {}).items() if k != "maps"},
        "satellite_choice": c.sats, "satellite_probes": (sat_meta or {}).get("ok"),
        "satellite_pixels": (sat_meta or {}).get("pixels"),          # offsets in 0.1 deg steps, no coordinates
        "forecast_fetched_at": c.live_at, "satellite_fetched_at": c.live_sat_at,
        "outputs": {k: data.get(k) for k in ("basis", "now", "today", "tomorrow", "day_after")},
    }
