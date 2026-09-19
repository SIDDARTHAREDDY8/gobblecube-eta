"""Shared cleaning + feature logic for the Gobblecube ETA challenge.

Used by training scripts and (a dependency-light twin of it) by predict.py.
Conventions (kept identical everywhere):
  - All datetimes are NYC wall time (America/New_York).
  - dow: Monday=0 .. Sunday=6  (pandas dayofweek convention)
  - duration bounds: 60s .. 10800s, zones 1..265, passenger_count >= 1
"""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

NYC = ZoneInfo("America/New_York")

MIN_DURATION_S = 30.0  # matches challenge cleaning: drop trips < 30s or > 3h
MAX_DURATION_S = 10800.0
MIN_ZONE, MAX_ZONE = 1, 265

EARTH_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(a))


def parse_request_time(requested_at: str) -> datetime:
    """Parse ISO-8601 requested_at -> NYC wall time.

    Naive strings are assumed to already be NYC wall time (the challenge's
    eval requests are generated from NYC-local trip records). Aware strings
    are converted to America/New_York.
    """
    s = requested_at.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    ts = datetime.fromisoformat(s)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=NYC)
    return ts.astimezone(NYC)


def is_rush_hour(hour: int, dow: int) -> int:
    return int(dow < 5 and (7 <= hour <= 10 or 16 <= hour <= 19))


# Feature order is fixed and shared with train_model.py / predict.py.
# MUST stay in lock-step with the feature vector predict.py builds — same
# names, same order. Month and its cyclic encodings are deliberately absent:
# dev and eval are winter-holiday slices, so the month effect learned from the
# rest of the year does not transfer (270.4s with month vs 257.4s without, on
# the official dev set). Keeping them here while predict.py omitted them is
# what made `train_full.py` emit a 20-feature model that predict.py could not
# load — i.e. following the README's own reproduce steps broke the submission.
FEATURE_NAMES = [
    "pu", "do", "hour", "dow",                   # hour/dow native categorical
    "passenger_count",
    "pu_lat", "pu_lon", "do_lat", "do_lon",
    "haversine_km",
    "is_weekend",
    "is_rush",
    "log_pair_count",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
# NOTE: pu/do are NOT native categoricals: sklearn's HGBR caps native
# categorical cardinality at 255, but zones run 1..265. pu/do go in as
# numerics (their spatial signal also rides on lat/lon + haversine);
# hour/dow stay native categoricals.
CATEGORICAL_IDX = [2, 3]


def cyclic_time(hour: int, dow: int, month: int) -> tuple[float, ...]:
    """Sin/cos encodings so 23:00 is near 00:00, Dec near Jan, Sun near Mon."""
    return (
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
        math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
    )
