"""Gobblecube AI Builder ETA Challenge - submission.

Exposes:
    def predict(request: dict) -> float

request = {"pickup_zone": int 1-265, "dropoff_zone": int 1-265,
           "requested_at": str ISO-8601, "passenger_count": int}
returns predicted trip duration in seconds (float).

Method: hierarchical backoff lookup tables over (pu,do,hour,dow) built from
11.5 months of 2023 NYC yellow-taxi trips, plus a HistGradientBoosting
regressor trained on the lookup residual with geo + calendar features.
The residual model deliberately excludes month features: the dev/eval
windows are holiday slices whose month effect does not transfer from the
rest of the year. Single-request latency is a few ms on CPU (budget:
200 ms). No network calls.
"""
from __future__ import annotations

import csv
import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
NYC = ZoneInfo("America/New_York")
EARTH_KM = 6371.0

# ---------------------------------------------------------------- artifacts
_zones: dict[int, tuple[float, float]] = {}
with open(ART / "zones.csv") as _f:
    for _row in csv.DictReader(_f):
        _zones[int(_row["LocationID"])] = (float(_row["lat"]), float(_row["lon"]))
_FALLBACK = (40.7580, -73.9855)  # Midtown, used only if a zone id is unknown

_z = np.load(ART / "lookups.npz")
_TABLES = {
    "global_mean": float(_z["global_mean"]),
    "l1": (_z["l1_means"], _z["l1_counts"]),
    "l2": (_z["l2_means"], _z["l2_counts"]),
    "l3": (_z["l3_means"], _z["l3_counts"]),
    "l4": (_z["l4_means"], _z["l4_counts"]),
    "l5": (_z["l5_means"], _z["l5_counts"]),
}
_THRESHOLDS = json.loads((ART / "meta.json").read_text())["thresholds"]

with open(ART / "model.pkl", "rb") as _f:
    _MODEL = pickle.load(_f)

# ---------------------------------------------------------------- helpers

def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _lookup(pu: int, do: int, hour: int, dow: int) -> float:
    m1, c1 = _TABLES["l1"]
    if c1[pu, do, hour, dow] >= _THRESHOLDS["l1_pu_do_hour_dow"]:
        return float(m1[pu, do, hour, dow])
    m2, c2 = _TABLES["l2"]
    if c2[pu, do, hour] >= _THRESHOLDS["l2_pu_do_hour"]:
        return float(m2[pu, do, hour])
    m3, c3 = _TABLES["l3"]
    if c3[pu, do] >= _THRESHOLDS["l3_pu_do"]:
        return float(m3[pu, do])
    m4, c4 = _TABLES["l4"]
    if c4[pu] >= _THRESHOLDS["l4_pu"]:
        return float(m4[pu])
    m5, c5 = _TABLES["l5"]
    if c5[do] >= _THRESHOLDS["l5_do"]:
        return float(m5[do])
    return float(_TABLES["global_mean"])


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------- public API

def predict(request: dict) -> float:
    pu = _clamp(int(request["pickup_zone"]), 1, 265)
    do = _clamp(int(request["dropoff_zone"]), 1, 265)
    pax = max(1, int(request.get("passenger_count", 1)))

    s = str(request["requested_at"]).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    ts = datetime.fromisoformat(s)
    ts = ts.replace(tzinfo=NYC) if ts.tzinfo is None else ts.astimezone(NYC)
    hour, dow = ts.hour, ts.weekday()  # Monday=0

    base = _lookup(pu, do, hour, dow)

    pu_lat, pu_lon = _zones.get(pu, _FALLBACK)
    do_lat, do_lon = _zones.get(do, _FALLBACK)
    pair_count = int(_TABLES["l3"][1][pu, do])

    # 17 features; month and its cyclic encodings are intentionally excluded
    # (holiday-slice distribution shift — see module docstring).
    x = np.array([[
        pu, do, hour, dow,
        pax,
        pu_lat, pu_lon, do_lat, do_lon,
        _haversine_km(pu_lat, pu_lon, do_lat, do_lon),
        int(dow >= 5),
        int(dow < 5 and (7 <= hour <= 10 or 16 <= hour <= 19)),
        math.log1p(pair_count),
        # cyclic time encodings (mirrors src/features.cyclic_time)
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
    ]], dtype=np.float64)

    resid = float(_MODEL.predict(x)[0])
    return float(min(14400.0, max(30.0, base + resid)))


if __name__ == "__main__":
    demo = {"pickup_zone": 132, "dropoff_zone": 236,
            "requested_at": "2024-01-15T08:30:00", "passenger_count": 1}
    import time
    t = time.perf_counter()
    print(f"{predict(demo):.1f}s ({(time.perf_counter()-t)*1000:.1f} ms)")
