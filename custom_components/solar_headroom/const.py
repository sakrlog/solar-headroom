"""Constants for Solar Headroom."""
from __future__ import annotations

import logging
from datetime import timedelta

DOMAIN = "solar_headroom"
LOGGER = logging.getLogger(__package__)

CONF_PV_POWER = "pv_power"
CONF_KIND = "kind"                           # does the inverter hold back? see KINDS
CONF_EXPORT_LIMIT = "export_limit_w"
CONF_BATTERY_POWER = "battery_power"
CONF_BATTERY_INVERTED = "battery_power_inverted"
CONF_BATTERY_SOC = "battery_soc"
CONF_GRID_POWER = "grid_power"
CONF_GRID_INVERTED = "grid_power_inverted"
CONF_GRID_VOLTAGE = "grid_voltage"
CONF_GENERATOR_POWER = "generator_power"
CONF_PV_VOLTAGE = "pv_voltage"
KIND_UNLIMITED, KIND_EXPORT_LIMITED, KIND_OFF_GRID = "unlimited", "export_limited", "off_grid"
KINDS = [KIND_UNLIMITED, KIND_EXPORT_LIMITED, KIND_OFF_GRID]
# optional sensors -> engine roles (engine.history); inverted flags flip the sign
SENSOR_ROLES = {CONF_BATTERY_POWER: ("bat_w", CONF_BATTERY_INVERTED), CONF_BATTERY_SOC: ("soc", None),
                CONF_GRID_POWER: ("grid_w", CONF_GRID_INVERTED), CONF_GRID_VOLTAGE: ("grid_v", None),
                CONF_GENERATOR_POWER: ("gen_w", None), CONF_PV_VOLTAGE: ("pv_v", None)}
CONF_KWP = "kwp"
CONF_TILT = "tilt"
CONF_AZIMUTH = "azimuth"
SUBENTRY_TYPE_PLANE = "plane"
MAX_PLANES = 6

DEFAULT_TILT = 30
DEFAULT_AZIMUTH = 180

UPDATE_INTERVAL = timedelta(minutes=5)       # outputs from the newest forecast + today's measurements
LIVE_MAX_AGE_S = 3600                        # live forecast fetched again after this (+ per-install jitter)
LIVE_SAT_MIN_GAP_S = 300                     # live satellite: when an interval is needed, at most every 5 min ...
LIVE_SAT_MAX_GAP_S = 1800                    # ... backing off to 30 min while the feed is late or down
LIVE_STALE_S = 6 * 3600                      # older than this: status "degraded"
RETRAIN_AFTER_S = 20 * 3600                  # at startup: train if the last training is older than this
TRAIN_HOUR = 2                               # nightly training, local time (+ per-install minute)
TRAIN_TIMEOUT_S = 3600
STATS_CHUNK_DAYS = 31                        # one recorder query per month of history
ARCHIVE_WORKERS = 3                          # archive downloads at a time (first learning)
REQUEST_PAUSE_S = 1.5                        # between Open-Meteo archive requests
REQUEST_TIMEOUT_S = 120

STATUS_STARTING = "starting"
STATUS_COLLECTING = "collecting"             # not enough data to learn yet: physics only
STATUS_LEARNING = "learning"                 # reading history / weather archives / training
STATUS_READY = "ready"
STATUS_DEGRADED = "degraded"                 # forecast unreachable or training failed
STATUSES = [STATUS_STARTING, STATUS_COLLECTING, STATUS_LEARNING, STATUS_READY, STATUS_DEGRADED]
