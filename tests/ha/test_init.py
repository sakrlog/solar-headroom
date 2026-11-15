"""Setup, learning (real training subprocess), sensors, Energy forecast, diagnostics, unload and removal."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.diagnostics import get_diagnostics_for_config_entry
from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import entity_registry as er

from custom_components.solar_headroom import energy
from custom_components.solar_headroom.const import DOMAIN

from .conftest import AZIMUTH, KWP, PV_ID, TILT, pv_rows

HISTORY_DAYS = 90


def _entry() -> MockConfigEntry:
    return MockConfigEntry(domain=DOMAIN, unique_id=PV_ID, title="Solar Headroom",
                           data={"pv_power": [PV_ID], "kind": "unlimited"},
                           subentries_data=[{"subentry_type": "plane", "title": "roof", "unique_id": None,
                                             "data": {"kwp": KWP, "tilt": TILT, "azimuth": AZIMUTH}}])


async def _import_history(hass, days: int) -> None:
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    meta = {"source": "recorder", "statistic_id": PV_ID, "unit_of_measurement": "W", "unit_class": "power",
            "mean_type": StatisticMeanType.ARITHMETIC, "has_sum": False, "name": None}
    async_import_statistics(hass, meta, pv_rows(end - timedelta(days=days), end))
    await async_wait_recording_done(hass)


def _state(hass, key: str):
    ent = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{_entry_id(hass)}_{key}")
    return hass.states.get(ent)


def _entry_id(hass) -> str:
    return hass.config_entries.async_entries(DOMAIN)[0].entry_id


async def test_learns_from_history(hass, configured, open_meteo, hass_client):
    await _import_history(hass, HISTORY_DAYS)
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    c = entry.runtime_data

    # Home Assistant is running, nothing trained yet -> learning starts by itself; wait for it
    assert c.learning
    await c._learning
    await hass.async_block_till_done()
    assert c.last_error is None, c.last_error
    assert c.status == "ready"
    rep = c.result["report"]
    assert rep["days_of_data"] >= HISTORY_DAYS - 2
    assert c.result["champion"] == {"nowcast": "ml", "daily": "ml"}      # the fake sky is learnable
    assert rep["daily_holdout"]["ml_mae_kwh"] < rep["daily_holdout"]["physics_mae_kwh"]
    assert c.result["range"] is not None and rep["peak_rss_mb"] > 0
    assert 0.5 < c.result["factor"] < 0.85                               # losses + the shade, on average

    # sensors from the learned models
    tomorrow = _state(hass, "forecast_tomorrow")
    assert float(tomorrow.state) > 0
    assert float(tomorrow.attributes["p10"]) <= float(tomorrow.state) <= float(tomorrow.attributes["p90"])
    assert tomorrow.attributes["basis"] == "learned"
    assert _state(hass, "bank_today").state not in ("unknown", "unavailable")
    status = _state(hass, "status")
    assert status.state == "ready" and status.attributes["forecast_model"] == "ml"

    # Energy dashboard forecast: Wh per hour start, today and tomorrow
    fc = await energy.async_get_solar_forecast(hass, entry.entry_id)
    hours = [datetime.fromisoformat(k) for k in fc["wh_hours"]]
    assert len(hours) > 16 and all(h.minute == 0 for h in hours)
    assert max(fc["wh_hours"].values()) < KWP * 1000

    diag = await get_diagnostics_for_config_entry(hass, hass_client, entry)
    assert diag["system"]["latitude"] == "**REDACTED**" and diag["status"] == "ready"

    # a second run only fetches what changed: no new archive chunks, statistics continue
    archive_calls = sum("historical-forecast" in str(u) or "previous-runs" in str(u)
                        for _, u, _, _ in open_meteo.mock_calls)
    c.async_schedule_learning("test")
    await c._learning
    again = sum("historical-forecast" in str(u) or "previous-runs" in str(u) for _, u, _, _ in open_meteo.mock_calls)
    assert again == archive_calls
    store = Path(c.store_path)
    assert store.exists()

    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED
    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()
    assert not store.exists()


async def test_physics_until_there_is_data(hass, configured, open_meteo):
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    c = entry.runtime_data
    if c.learning:
        await c._learning
    await hass.async_block_till_done()
    assert c.status == "collecting"
    tomorrow = _state(hass, "forecast_tomorrow")
    assert float(tomorrow.state) > 0 and tomorrow.attributes["basis"] == "physics"
    assert _state(hass, "potential_now").state not in ("unknown", "unavailable")
    assert await hass.config_entries.async_unload(entry.entry_id)
