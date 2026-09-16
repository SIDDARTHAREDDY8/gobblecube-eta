"""Cleaning + feature engineering shared by training and validation.

TLC yellow-taxi columns used: tpep_pickup_datetime, tpep_dropoff_datetime,
passenger_count, PULocationID, DOLocationID, trip_distance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import (
    FEATURE_NAMES, MAX_DURATION_S, MAX_ZONE, MIN_DURATION_S, MIN_ZONE,
    cyclic_time, haversine_km, is_rush_hour,
)

PICKUP, DROPOFF = "tpep_pickup_datetime", "tpep_dropoff_datetime"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[PICKUP, DROPOFF, "passenger_count", "PULocationID",
             "DOLocationID", "trip_distance"]].copy()
    df[PICKUP] = pd.to_datetime(df[PICKUP], errors="coerce")
    df[DROPOFF] = pd.to_datetime(df[DROPOFF], errors="coerce")
    df = df.dropna(subset=[PICKUP, DROPOFF])
    df["duration_s"] = (df[DROPOFF] - df[PICKUP]).dt.total_seconds()
    df = df[(df["duration_s"] >= MIN_DURATION_S)
            & (df["duration_s"] <= MAX_DURATION_S)]
    df = df[(df["PULocationID"] >= MIN_ZONE) & (df["PULocationID"] <= MAX_ZONE)
            & (df["DOLocationID"] >= MIN_ZONE) & (df["DOLocationID"] <= MAX_ZONE)]
    df = df[(df["passenger_count"] >= 1) & (df["trip_distance"] > 0)]
    # TLC datetimes are NYC wall time; derive calendar fields directly.
    pu = df[PICKUP]
    df["pu"] = df["PULocationID"].astype(np.int64)
    df["do"] = df["DOLocationID"].astype(np.int64)
    df["hour"] = pu.dt.hour.astype(np.int64)
    df["dow"] = pu.dt.dayofweek.astype(np.int64)  # Monday=0
    df["month"] = pu.dt.month.astype(np.int64)
    return df.reset_index(drop=True)


def load_zones(path) -> dict[int, tuple[float, float]]:
    import csv
    zones = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            zones[int(row["LocationID"])] = (float(row["lat"]), float(row["lon"]))
    return zones


def add_geo_features(df: pd.DataFrame, zones: dict) -> pd.DataFrame:
    df = df.copy()
    lats = np.array([zones.get(z, (40.7128, -74.0060))[0] for z in df["pu"]])
    lons = np.array([zones.get(z, (40.7128, -74.0060))[1] for z in df["pu"]])
    # note: zones.get returns (lat, lon); default = Manhattan-ish fallback
    df["pu_lat"] = lats
    df["pu_lon"] = np.array([zones.get(z, (40.7128, -74.0060))[1] for z in df["pu"]])
    df["do_lat"] = np.array([zones.get(z, (40.7128, -74.0060))[0] for z in df["do"]])
    df["do_lon"] = np.array([zones.get(z, (40.7128, -74.0060))[1] for z in df["do"]])
    df["haversine_km"] = [
        haversine_km(a, b, c, d) for a, b, c, d
        in zip(df["pu_lat"], df["pu_lon"], df["do_lat"], df["do_lon"])
    ]
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int64)
    df["is_rush"] = [is_rush_hour(h, d) for h, d in zip(df["hour"], df["dow"])]
    cyc = [cyclic_time(h, d, m) for h, d, m
           in zip(df["hour"], df["dow"], df["month"])]
    (df["hour_sin"], df["hour_cos"], df["dow_sin"],
     df["dow_cos"], df["mon_sin"], df["mon_cos"]) = (np.array(c) for c in zip(*cyc))
    return df


def add_pair_count(df: pd.DataFrame, pair_counts: dict | None) -> pd.DataFrame:
    """log1p count of (pu,do) in train. Counts are target-free (no leakage)."""
    df = df.copy()
    if pair_counts is None:
        df["log_pair_count"] = 0.0
    else:
        df["log_pair_count"] = np.log1p([
            pair_counts.get((int(p), int(d)), 0)
            for p, d in zip(df["pu"], df["do"])
        ])
    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([df[c].to_numpy() for c in FEATURE_NAMES])
    # categorical columns must be int for native HGBR categorical support
    return X


def load_months(months: list[str], raw_dir) -> pd.DataFrame:
    import pyarrow.parquet as pq
    from pathlib import Path
    frames = []
    for m in months:
        p = Path(raw_dir) / f"yellow_tripdata_{m}.parquet"
        print(f"reading {p} ...")
        frames.append(pq.read_table(p).to_pandas())
    return pd.concat(frames, ignore_index=True)
