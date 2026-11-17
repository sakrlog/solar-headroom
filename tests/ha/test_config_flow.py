"""Config flow, options flow and panel-group subentries."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.config_entries import SOURCE_USER
from homeassistant.data_entry_flow import FlowResultType, InvalidData

from custom_components.solar_headroom.const import DOMAIN

from .conftest import BAT_ID, PV_ID, SOC_ID, offgrid_rows, pv_rows, statistics_meta

PANELS = {"kwp": 5.4, "tilt": 25, "azimuth": 160}
SETUP = "custom_components.solar_headroom.async_setup_entry"


async def _through_panels(hass, kind: str | None = "unlimited"):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert result["type"] is FlowResultType.FORM and result["step_id"] == "user"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"pv_power": [PV_ID]})
    assert result["step_id"] == "panels"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], PANELS)
    assert result["step_id"] == "holdback"
    if kind is None:
        return result
    return await hass.config_entries.flow.async_configure(result["flow_id"], {"kind": kind})


async def _energy_prefs(hass, battery_inverted: bool) -> None:
    from homeassistant.components.energy.data import async_get_manager
    power = {"stat_rate_inverted": BAT_ID} if battery_inverted else {"stat_rate": BAT_ID}
    manager = await async_get_manager(hass)
    await manager.async_update({"energy_sources": [
        {"type": "solar", "stat_energy_from": "sensor.pv_energy", "stat_rate": PV_ID},
        {"type": "battery", "stat_energy_from": "sensor.bat_out", "stat_energy_to": "sensor.bat_in",
         "power_config": power, "stat_soc": SOC_ID}]})


async def _import_offgrid(hass, days: int, flip_battery: bool = False) -> None:
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    rows = offgrid_rows(end - timedelta(days=days), end)
    if flip_battery:          # a sensor that reports + while charging
        rows[BAT_ID] = [{"start": r["start"], "mean": -r["mean"], "min": -r["max"], "max": -r["min"]}
                        for r in rows[BAT_ID]]
    for sid, rs in rows.items():
        async_import_statistics(hass, statistics_meta(sid), rs)
    await async_wait_recording_done(hass)


async def test_off_grid_flow_prefills_and_checks_the_battery_sign(hass):
    await _import_offgrid(hass, 40, flip_battery=True)
    await _energy_prefs(hass, battery_inverted=False)          # the Energy dashboard has the sign wrong too
    result = await _through_panels(hass, kind=None)
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"kind": "off_grid"})
    assert result["step_id"] == "sensors"
    suggested = {str(k): k.description.get("suggested_value") for k in result["data_schema"].schema
                 if k.description}
    assert suggested["battery_power"] == BAT_ID and suggested["battery_soc"] == SOC_ID
    with pytest.raises(InvalidData):                            # the form requires the battery sensor
        await hass.config_entries.flow.async_configure(result["flow_id"], {"battery_soc": SOC_ID})
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"battery_power": BAT_ID, "battery_power_inverted": False, "battery_soc": SOC_ID})
    assert result["step_id"] == "history"
    assert "looked reversed" in result["description_placeholders"]["signs"]
    with patch(SETUP, return_value=True):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["data"] == {"pv_power": [PV_ID], "kind": "off_grid", "battery_power": BAT_ID,
                              "battery_power_inverted": True, "battery_soc": SOC_ID}


async def test_export_limited_flow_needs_grid_power(hass):
    result = await _through_panels(hass, kind=None)
    result = await hass.config_entries.flow.async_configure(result["flow_id"],
                                                            {"kind": "export_limited", "export_limit_w": 5000})
    assert result["step_id"] == "sensors"
    with pytest.raises(InvalidData):
        await hass.config_entries.flow.async_configure(result["flow_id"], {})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"grid_power": "sensor.grid"})
    assert result["step_id"] == "no_history"
    with patch(SETUP, return_value=True):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["data"] == {"pv_power": [PV_ID], "kind": "export_limited", "export_limit_w": 5000.0,
                              "grid_power": "sensor.grid", "grid_power_inverted": False}


async def test_flow_without_history(hass):
    result = await _through_panels(hass)
    assert result["step_id"] == "no_history"
    with patch(SETUP, return_value=True):
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {"pv_power": [PV_ID], "kind": "unlimited"}
    entry = result["result"]
    (plane,) = entry.subentries.values()
    assert plane.subentry_type == "plane" and plane.data == {"kwp": 5.4, "tilt": 25.0, "azimuth": 160.0}
    assert entry.unique_id == PV_ID


async def test_flow_reports_history(hass):
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    meta = {"source": "recorder", "statistic_id": PV_ID, "unit_of_measurement": "W", "unit_class": "power",
            "mean_type": StatisticMeanType.ARITHMETIC, "has_sum": False, "name": None}
    async_import_statistics(hass, meta, pv_rows(end - timedelta(days=400), end))
    await async_wait_recording_done(hass)
    result = await _through_panels(hass)
    assert result["step_id"] == "history"
    assert result["description_placeholders"]["history"] in ("1 year 1 month", "1 year 2 months")


async def test_duplicate_sensors_abort(hass):
    MockConfigEntry(domain=DOMAIN, unique_id=PV_ID, data={"pv_power": [PV_ID]}).add_to_hass(hass)
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"pv_power": [PV_ID]})
    assert result["type"] is FlowResultType.ABORT and result["reason"] == "already_configured"


async def test_empty_sensor_list_is_an_error(hass):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"pv_power": []})
    assert result["errors"] == {"pv_power": "no_sensor"}


async def test_options_and_panel_groups(hass):
    entry = MockConfigEntry(domain=DOMAIN, unique_id=PV_ID, data={"pv_power": [PV_ID], "kind": "unlimited"},
                            subentries_data=[{"subentry_type": "plane", "data": PANELS, "title": "p",
                                              "unique_id": None}])
    entry.add_to_hass(hass)
    with patch(SETUP, return_value=True):
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(result["flow_id"],
                                                                   {"pv_power": [PV_ID, "sensor.pv2"]})
        assert result["step_id"] == "holdback"
        result = await hass.config_entries.options.async_configure(result["flow_id"], {"kind": "off_grid"})
        assert result["step_id"] == "sensors"
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"battery_power": BAT_ID, "battery_soc": SOC_ID})
        assert result["type"] is FlowResultType.CREATE_ENTRY            # no history: no sign check to show
        assert entry.options == {"pv_power": [PV_ID, "sensor.pv2"], "kind": "off_grid", "battery_power": BAT_ID,
                                 "battery_power_inverted": False, "battery_soc": SOC_ID}
        # back to a grid-tied system: the battery sensors are dropped, not merged back from the setup data
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(result["flow_id"], {"pv_power": [PV_ID]})
        result = await hass.config_entries.options.async_configure(result["flow_id"], {"kind": "unlimited"})
        assert entry.options == {"pv_power": [PV_ID], "kind": "unlimited"}
        from custom_components.solar_headroom.coordinator import entry_conf
        assert entry_conf(entry) == {"pv_power": [PV_ID], "kind": "unlimited"}

        result = await hass.config_entries.subentries.async_init((entry.entry_id, "plane"),
                                                                 context={"source": SOURCE_USER})
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {"kwp": 2.0, "tilt": 15, "azimuth": 270})
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert len(entry.get_subentries_of_type("plane")) == 2

        first = entry.get_subentries_of_type("plane")[0]
        result = await entry.start_subentry_reconfigure_flow(hass, first.subentry_id)
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {"kwp": 6.0, "tilt": 25, "azimuth": 160})
        assert result["reason"] == "reconfigure_successful"
        assert entry.subentries[first.subentry_id].data["kwp"] == 6.0
