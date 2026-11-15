"""Constants for Solar Headroom."""
from __future__ import annotations

import logging
from datetime import timedelta

DOMAIN = "solar_headroom"
LOGGER = logging.getLogger(__package__)

CONF_PV_POWER = "pv_power"
CONF_KIND = "kind"
CONF_KWP = "kwp"
CONF_TILT = "tilt"
CONF_AZIMUTH = "azimuth"
SUBENTRY_TYPE_PLANE = "plane"
MAX_PLANES = 6

DEFAULT_TILT = 30
DEFAULT_AZIMUTH = 180

UPDATE_INTERVAL = timedelta(minutes=5)       # outputs from the newest forecast + today's measurements
LIVE_MAX_AGE_S = 3600                        # live forecast fetched again after this (+ per-install jitter)
LIVE_STALE_S = 6 * 3600                      # older than this: status "degraded"
RETRAIN_AFTER_S = 20 * 3600                  # at startup: train if the last training is older than this
TRAIN_HOUR = 2                               # nightly training, local time (+ per-install minute)
TRAIN_TIMEOUT_S = 3600
STATS_CHUNK_DAYS = 31                        # one recorder query per month of history
REQUEST_PAUSE_S = 1.5                        # between Open-Meteo archive requests
REQUEST_TIMEOUT_S = 120

STATUS_STARTING = "starting"
STATUS_COLLECTING = "collecting"             # not enough data to learn yet: physics only
STATUS_LEARNING = "learning"                 # reading history / weather archives / training
STATUS_READY = "ready"
STATUS_DEGRADED = "degraded"                 # forecast unreachable or training failed
STATUSES = [STATUS_STARTING, STATUS_COLLECTING, STATUS_LEARNING, STATUS_READY, STATUS_DEGRADED]
