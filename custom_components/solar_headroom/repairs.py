"""Repairs: what needs the user's attention, raised and cleared by the coordinator.

* sensor_unavailable -- a configured sensor gave no reading for an hour of daylight (inverters that
  sleep at night are fine)
* forecast_unreachable -- no forecast from Open-Meteo for 6 hours
* no_statistics -- a configured sensor has no long-term statistics (state_class missing): nothing to
  learn from
* sign_reversed -- the nightly check finds battery or grid power the wrong way round; fixable with a
  click (flips the sensor's "inverted" option, which retrains)
"""
from __future__ import annotations

from typing import Any

import probatio

from homeassistant.components.repairs import RepairsFlow, RepairsFlowResult
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir

from .const import DOMAIN, SENSOR_ROLES

SENSOR_UNAVAILABLE = "sensor_unavailable"
FORECAST_UNREACHABLE = "forecast_unreachable"
NO_STATISTICS = "no_statistics"
SIGN_REVERSED = "sign_reversed"


def issue_id(kind: str, entry_id: str, key: str = "") -> str:
    return f"{kind}_{entry_id}" + (f"_{key}" if key else "")


@callback
def raise_issue(hass: HomeAssistant, entry: ConfigEntry, kind: str, placeholders: dict[str, str],
                key: str = "", fixable: bool = False, data: dict | None = None) -> None:
    ir.async_create_issue(
        hass, DOMAIN, issue_id(kind, entry.entry_id, key), is_fixable=fixable, is_persistent=False,
        severity=ir.IssueSeverity.WARNING, translation_key=kind,
        translation_placeholders={"title": entry.title, **placeholders},
        data={"entry_id": entry.entry_id, **(data or {})})


@callback
def clear_issue(hass: HomeAssistant, entry: ConfigEntry, kind: str, key: str = "") -> None:
    ir.async_delete_issue(hass, DOMAIN, issue_id(kind, entry.entry_id, key))


@callback
def issues_of(hass: HomeAssistant, entry: ConfigEntry, kind: str) -> list[str]:
    """Keys of this entry's open issues of one kind."""
    prefix = issue_id(kind, entry.entry_id) + "_"
    return [i[1][len(prefix):] for i in ir.async_get(hass).issues if i[0] == DOMAIN and i[1].startswith(prefix)]


class SignRepairFlow(RepairsFlow):
    """Read battery or grid power the other way round (options updated; the entry reloads and retrains)."""

    def __init__(self, entry_id: str, key: str) -> None:
        self._entry_id, self._key = entry_id, key

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> RepairsFlowResult:
        return await self.async_step_confirm()

    async def async_step_confirm(self, user_input: dict[str, Any] | None = None) -> RepairsFlowResult:
        from .coordinator import entry_conf  # noqa: PLC0415 - coordinator imports this module

        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if entry is None:
            return self.async_abort(reason="entry_removed")
        if user_input is not None:
            conf = entry_conf(entry)
            flag = SENSOR_ROLES[self._key][1]
            conf[flag] = not conf.get(flag, False)
            self.hass.config_entries.async_update_entry(entry, options=conf)
            return self.async_create_entry(data={})
        issue = ir.async_get(self.hass).async_get_issue(DOMAIN, self.issue_id)
        return self.async_show_form(step_id="confirm", data_schema=probatio.Schema({}),
                                    description_placeholders=(issue.translation_placeholders if issue else None))


async def async_create_fix_flow(hass: HomeAssistant, issue_id: str, data: dict[str, Any] | None) -> RepairsFlow:
    if not data or "entry_id" not in data or "key" not in data:
        raise ValueError("missing data for the repair flow")
    return SignRepairFlow(data["entry_id"], data["key"])
