"""Repair issues: raised when something needs the user, cleared when it is fine again."""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed
from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.components.repairs import DOMAIN as REPAIRS
from homeassistant.helpers import issue_registry as ir
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util

from custom_components.solar_headroom.const import DOMAIN

from .conftest import AZIMUTH, BAT_ID, KWP, PV_ID, SOC_ID, TILT, TZ, offgrid_rows, statistics_meta


def _entry(**extra) -> MockConfigEntry:
    data = {"pv_power": [PV_ID], "kind": "off_grid", "battery_power": BAT_ID, "battery_power_inverted": False,
            "battery_soc": SOC_ID, **extra}
    return MockConfigEntry(domain=DOMAIN, unique_id=PV_ID, title="Solar Headroom", data=data,
                           subentries_data=[{"subentry_type": "plane", "title": "roof", "unique_id": None,
                                             "data": {"kwp": KWP, "tilt": TILT, "azimuth": AZIMUTH}}])


async def _setup(hass, entry):
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    c = entry.runtime_data
    if c.learning:
        await c._learning
    await hass.async_block_till_done()
    return c


def _issue(hass, kind: str, entry, key: str = ""):
    iid = f"{kind}_{entry.entry_id}" + (f"_{key}" if key else "")
    return ir.async_get(hass).async_get_issue(DOMAIN, iid)


async def test_sensor_unavailable_in_daylight_only(hass, configured, open_meteo, freezer):
    day = datetime.combine(datetime.now(ZoneInfo(TZ)).date(), datetime.min.time(), ZoneInfo(TZ))
    freezer.move_to(day)                                                 # midnight: inverters may sleep
    hass.states.async_set(PV_ID, "unavailable")
    hass.states.async_set(BAT_ID, "100", {"unit_of_measurement": "W"})
    hass.states.async_set(SOC_ID, "80", {"unit_of_measurement": "%"})
    entry = _entry()
    c = await _setup(hass, entry)
    for _ in range(150):                                                 # 75 minutes of night
        freezer.tick(timedelta(seconds=30))
        async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()
    await c.async_refresh()
    assert _issue(hass, "sensor_unavailable", entry, "sensor_pv_power") is None
    freezer.move_to(day + timedelta(hours=11))
    for _ in range(125):                                                 # an hour of daylight
        freezer.tick(timedelta(seconds=30))
        async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()
    await c.async_refresh()
    issue = _issue(hass, "sensor_unavailable", entry, "sensor_pv_power")
    assert issue is not None and issue.translation_placeholders["entity"] == PV_ID
    hass.states.async_set(PV_ID, "1500", {"unit_of_measurement": "W"})
    await hass.async_block_till_done()
    await c.async_refresh()
    assert _issue(hass, "sensor_unavailable", entry, "sensor_pv_power") is None
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_forecast_unreachable(hass, configured, open_meteo):
    entry = _entry()
    c = await _setup(hass, entry)
    now = time.time()
    c.live_at = now - 7 * 3600
    c._check_forecast(now)
    assert _issue(hass, "forecast_unreachable", entry) is not None
    c.live_at = now
    c._check_forecast(now)
    assert _issue(hass, "forecast_unreachable", entry) is None
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_no_statistics(hass, configured, open_meteo):
    hass.states.async_set(PV_ID, "1500", {"unit_of_measurement": "W"})       # no state_class: no statistics
    entry = _entry()
    await _setup(hass, entry)
    issue = _issue(hass, "no_statistics", entry, "sensor_pv_power")
    assert issue is not None and issue.translation_placeholders["entity"] == PV_ID
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.config_entries.async_remove(entry.entry_id)
    assert _issue(hass, "no_statistics", entry, "sensor_pv_power") is None


async def test_reversed_battery_is_fixed_with_a_click(hass, configured, open_meteo):
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    rows = offgrid_rows(end - timedelta(days=40), end)
    rows[BAT_ID] = [{"start": r["start"], "mean": -r["mean"], "min": -r["max"], "max": -r["min"]} for r in rows[BAT_ID]]
    for sid, rs in rows.items():
        async_import_statistics(hass, statistics_meta(sid), rs)
    await async_wait_recording_done(hass)
    entry = _entry()
    await _setup(hass, entry)
    issue = _issue(hass, "sign_reversed", entry, "bat_w")
    assert issue is not None and issue.is_fixable and issue.translation_placeholders["entity"] == BAT_ID

    assert await async_setup_component(hass, REPAIRS, {})
    flows = hass.data[REPAIRS]["flow_manager"]
    result = await flows.async_init(DOMAIN, data={"issue_id": issue.issue_id})
    assert result["step_id"] == "confirm" and result["description_placeholders"]["entity"] == BAT_ID
    result = await flows.async_configure(result["flow_id"], {})
    assert result["type"] == "create_entry"
    await hass.async_block_till_done()
    assert entry.options["battery_power_inverted"] is True and entry.options["kind"] == "off_grid"
    assert _issue(hass, "sign_reversed", entry, "bat_w") is None
    c = entry.runtime_data                                                   # reloaded with the new sign
    if c.learning:
        await c._learning
    await hass.async_block_till_done()
    assert _issue(hass, "sign_reversed", entry, "bat_w") is None
    assert await hass.config_entries.async_unload(entry.entry_id)
