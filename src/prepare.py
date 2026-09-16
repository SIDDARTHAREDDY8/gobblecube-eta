"""Cleaning + feature engineering shared by training and validation.

TLC yellow-taxi columns used: tpep_pickup_datetime, tpep_dropoff_datetime,
passenger_count, PULocationID, DOLocationID, trip_distance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import (FEATURE_NAMES, MAX_DURATION_S, MAX_ZONE, MIN_DURATION_S,
                      MIN_ZONE)

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
    df = df[(df["passenger_count"].fillna(1) >= 1) & (df["trip_distance"] > 0)]
    df["passenger_count"] = df["passenger_count"].fillna(1).astype(np.int64)
    # TLC datetimes are NYC wall time; derive calendar fields directly.
    pu = df[PICKUP]
    df = df[(pu.dt.year == 2023)]  # matches challenge cleaning
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
    fb = (40.7580, -73.9855)  # Midtown fallback for unknown zone ids
    lat_arr = np.array([zones.get(i, fb)[0] for i in range(266)])
    lon_arr = np.array([zones.get(i, fb)[1] for i in range(266)])
    pu = df["pu"].to_numpy()
    do = df["do"].to_numpy()
    df["pu_lat"] = lat_arr[pu]
    df["pu_lon"] = lon_arr[pu]
    df["do_lat"] = lat_arr[do]
    df["do_lon"] = lon_arr[do]
    # vectorized haversine
    p1 = np.radians(df["pu_lat"].to_numpy())
    p2 = np.radians(df["do_lat"].to_numpy())
    dp = np.radians(df["do_lat"].to_numpy() - df["pu_lat"].to_numpy())
    dl = np.radians(df["do_lon"].to_numpy() - df["pu_lon"].to_numpy())
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    df["haversine_km"] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int64)
    h = df["hour"].to_numpy()
    d = df["dow"].to_numpy()
    m = df["month"].to_numpy()
    df["is_rush"] = ((d < 5) & (((h >= 7) & (h <= 10)) | ((h >= 16) & (h <= 19)))
                     ).astype(np.int64)
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["dow_sin"] = np.sin(2 * np.pi * d / 7)
    df["dow_cos"] = np.cos(2 * np.pi * d / 7)
    df["mon_sin"] = np.sin(2 * np.pi * m / 12)
    df["mon_cos"] = np.cos(2 * np.pi * m / 12)
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
