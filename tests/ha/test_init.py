"""Setup, learning (real training subprocess), sensors, Energy forecast, diagnostics, unload and removal."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.diagnostics import get_diagnostics_for_config_entry
from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import entity_registry as er

from custom_components.solar_headroom import energy
from custom_components.solar_headroom.const import DOMAIN
from custom_components.solar_headroom.engine import pipeline as P

from .conftest import AZIMUTH, KWP, PV_ID, SYSTEM_LOSS, TILT, clear_sky, day_k, open_meteo_body, pv_rows

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

    # satellite: SARAH3 trains, DWD (live) is mapped onto it, the real-time model uses both; both are read
    # at the pixel that follows the PV (the fake's parallax), not at the site's own grid point
    assert (c.sats["history"], c.sats["live"]) == ("eumetsat_sarah3", "dwd_sis_europe_africa_v4")
    assert c.sats["offsets"] == {"eumetsat_sarah3": [0, 1], "dwd_sis_europe_africa_v4": [0, 1]}
    native = [u for _, u, _, _ in open_meteo.mock_calls       # archive + live (not the coverage probes)
              if "satellite-api" in str(u) and u.query.get("temporal_resolution") == "native"
              and u.query["hourly"] != "shortwave_radiation"]
    assert native and {(u.query["latitude"], u.query["longitude"]) for u in native} == {("45.0", "7.1")}
    pixels = (await get_diagnostics_for_config_entry(hass, hass_client, entry))["satellite_pixels"]
    assert pixels["eumetsat_sarah3"]["scores"]["+0,+1"] > pixels["eumetsat_sarah3"]["scores"]["+0,+0"]
    sat = c.result["satellite"]
    assert sat["used"] and sat["share"] > 0.9 and sat["maps"]["n_pairs"] >= 200
    assert c.result["nowcast_fc"] is not None and "sat_kgti_lag" in c.result["nowcast"]["features"]
    assert c._live_sat is not None                                         # fetched with the first update
    # yesterday late morning (inside the live forecast): the satellite model with the live feed is close to truth
    noon = (datetime.now(UTC) - timedelta(days=1)).replace(hour=11, minute=7, second=0, microsecond=0).timestamp()
    day = datetime.fromtimestamp(noon, UTC).date().isoformat()
    req = P.live_satellite_requests(c.system, c.sats, noon)[0]
    query = urlencode(req.params | {"start_date": day, "end_date": day})
    live_sat = P.satellite_frame(c.system, "dwd_sis_europe_africa_v4", [[open_meteo_body(query)]])
    out = P.outputs(c.system, c.result, c._live, noon, models=c._models, live_sat=live_sat)
    truth = SYSTEM_LOSS * day_k([noon])[0] * clear_sky([noon - 420], period=900)["p_cs"][0]
    assert out["now"]["basis"] == "learned_satellite"
    assert abs(out["now"]["w"] - truth) < 0.15 * truth, (out["now"], truth)
    # the feed stopped at 11:00: two hours later the forecast-only model takes over
    cut = {k: (v[live_sat[0]["ts"] < noon] if hasattr(v, "__len__") else v) for k, v in live_sat[0].items()}
    stale = P.outputs(c.system, c.result, c._live, noon + 2 * 3600, models=c._models, live_sat=(cut, live_sat[1]))
    assert stale["now"]["basis"] == "learned_forecast" and stale["now"].get("satellite_k") is None

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
    def archive_calls() -> int:      # the satellite archive is planned the same way (tests/test_store_pipeline.py)
        return sum("historical-forecast" in str(u) or "previous-runs" in str(u) for _, u, _, _ in open_meteo.mock_calls)
    before = archive_calls()
    c.async_schedule_learning("test")
    await c._learning
    assert archive_calls() == before
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
