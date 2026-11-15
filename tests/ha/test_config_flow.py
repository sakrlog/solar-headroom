"""Config flow, options flow and panel-group subentries."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.config_entries import SOURCE_USER
from homeassistant.data_entry_flow import FlowResultType

from custom_components.solar_headroom.const import DOMAIN

from .conftest import PV_ID, pv_rows

PANELS = {"kwp": 5.4, "tilt": 25, "azimuth": 160}
SETUP = "custom_components.solar_headroom.async_setup_entry"


async def _through_panels(hass):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert result["type"] is FlowResultType.FORM and result["step_id"] == "user"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {"pv_power": [PV_ID]})
    assert result["step_id"] == "panels"
    return await hass.config_entries.flow.async_configure(result["flow_id"], PANELS)


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
    assert result["description_placeholders"]["history"].startswith("1 year 1 month")


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
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert entry.options == {"pv_power": [PV_ID, "sensor.pv2"]}

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
