"""The live layer in Home Assistant: sensors followed by state, units, holds, estimate, day log, restore."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed

from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from custom_components.solar_headroom.const import DOMAIN

from .conftest import AZIMUTH, BAT_ID, KWP, PV_ID, SOC_ID, TILT, TZ


def _entry(kind: str = "off_grid") -> MockConfigEntry:
    data = {"pv_power": [PV_ID], "kind": kind}
    if kind == "off_grid":
        data |= {"battery_power": BAT_ID, "battery_power_inverted": True, "battery_soc": SOC_ID}
    return MockConfigEntry(domain=DOMAIN, unique_id=PV_ID, title="Solar Headroom", data=data,
                           subentries_data=[{"subentry_type": "plane", "title": "roof", "unique_id": None,
                                             "data": {"kwp": KWP, "tilt": TILT, "azimuth": AZIMUTH}}])


def _state(hass, entry, key: str):
    ent = er.async_get(hass).async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{key}")
    return None if ent is None else hass.states.get(ent)


async def _advance(hass, freezer, seconds: float) -> None:
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _setup(hass, entry):
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    c = entry.runtime_data
    if c.learning:
        await c._learning
    await hass.async_block_till_done()
    return c


async def test_live_potential_off_grid(hass, configured, open_meteo, freezer):
    noon = datetime.combine(datetime.now(ZoneInfo(TZ)).date(), datetime.min.time(), ZoneInfo(TZ)) + timedelta(hours=12)
    freezer.move_to(noon)
    # a kW PV sensor; a battery sensor that reports + while charging (configured as inverted)
    hass.states.async_set(PV_ID, "1.2", {"unit_of_measurement": "kW"})
    hass.states.async_set(BAT_ID, "300", {"unit_of_measurement": "W"})        # charging 300 W
    hass.states.async_set(SOC_ID, "100", {"unit_of_measurement": "%"})        # full: held back
    entry = _entry()
    c = await _setup(hass, entry)
    t = c.tracker
    assert t.estimator.values == {"pv_w": 1200.0, "bat_w": -300.0, "soc": 100.0}
    await _advance(hass, freezer, 31)
    est = t.now
    assert est["regime"] == "throttled" and est["reason"] == "held_back"
    assert est["basis"] == "sky_model" and est["w"] > 1200.0
    pot = _state(hass, entry, "potential_now")
    assert float(pot.state) == round(est["w"]) and pot.attributes["regime"] == "throttled"
    assert pot.attributes["basis"] == "physics"                               # the sky model's own basis
    assert float(_state(hass, entry, "untapped_now").state) == round(est["w"] - 1200.0)

    # the house takes more than the panels give: the battery helps; full output once it held 20 s
    hass.states.async_set(BAT_ID, "-400", {"unit_of_measurement": "W"})
    hass.states.async_set(PV_ID, "2.0", {"unit_of_measurement": "kW"})
    await _advance(hass, freezer, 2)
    assert t.now["regime"] == "throttled"                                    # not held long enough yet
    await _advance(hass, freezer, 30)
    assert t.now["regime"] == "full_output" and t.now["w"] == 2000.0 and t.now["basis"] == "measured"
    pot = _state(hass, entry, "potential_now")
    assert pot.state == "2000" and pot.attributes["basis"] == "measured"
    assert _state(hass, entry, "untapped_now").state == "0"
    k_full = 2000.0 / t.now["clear_sky_w"]

    # battery full again: estimated from the last full-output moment (the sky model's error carried over)
    hass.states.async_set(BAT_ID, "0", {"unit_of_measurement": "W"})
    hass.states.async_set(PV_ID, "0.9", {"unit_of_measurement": "kW"})
    await _advance(hass, freezer, 31)
    est = t.now
    assert est["basis"] == "estimated" and est["minutes_since_full"] is not None
    assert abs(est["w"] / est["clear_sky_w"] - k_full) < 0.05                 # minutes later: nearly k_full
    assert _state(hass, entry, "potential_now").attributes["basis"] == "estimated"

    # wasted energy adds up while held back
    for _ in range(10):
        await _advance(hass, freezer, 30)
    wasted = float(_state(hass, entry, "wasted_today").state)
    assert wasted > 0.05
    assert abs(wasted - t.wasted_today_kwh()) < 0.01
    # the solar bank counts the logged potential for the past part of today (not the throttled output)
    await c.async_refresh()
    today = c.data["today"]
    assert today["so_far_potential_kwh"] > today["so_far_used_kwh"] > 0
    assert abs(today["so_far_wasted_kwh"] - wasted) < 0.06 and today["log_coverage_pct"] > 0
    from custom_components.solar_headroom.diagnostics import async_get_config_entry_diagnostics
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["live"]["rules"]["kind"] == "off_grid" and diag["live"]["last_full"] is not None

    # unavailable PV: still estimated from the sky model; untapped cannot be known; not counted
    covered = sum(v[0] for v in t.day_log.buckets.values())
    hass.states.async_set(PV_ID, "unavailable")
    await _advance(hass, freezer, 31)
    await _advance(hass, freezer, 30)
    assert t.now["regime"] == "unknown" and t.now["basis"] == "estimated"
    assert _state(hass, entry, "potential_now").attributes["regime"] == "unknown"
    assert _state(hass, entry, "untapped_now").state == "unavailable"
    assert sum(v[0] for v in t.day_log.buckets.values()) - covered <= 31.0      # only the step before it

    # a reload keeps the last full-output moment and today's log
    last_full = t.estimator.last_full
    wasted = t.wasted_today_kwh()
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    t2 = entry.runtime_data.tracker
    assert t2.estimator.last_full == last_full
    assert abs(t2.wasted_today_kwh() - wasted) < 0.01
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_grid_tied_shows_the_measured_power_and_no_holdback_sensors(hass, configured, open_meteo, freezer):
    noon = datetime.combine(datetime.now(ZoneInfo(TZ)).date(), datetime.min.time(), ZoneInfo(TZ)) + timedelta(hours=12)
    freezer.move_to(noon)
    hass.states.async_set(PV_ID, "2500", {"unit_of_measurement": "W"})
    entry = _entry("unlimited")
    c = await _setup(hass, entry)
    await _advance(hass, freezer, 31)
    assert c.tracker.now["basis"] == "measured"
    assert _state(hass, entry, "potential_now").state == "2500"
    assert _state(hass, entry, "untapped_now") is None and _state(hass, entry, "wasted_today") is None
    assert await hass.config_entries.async_unload(entry.entry_id)
