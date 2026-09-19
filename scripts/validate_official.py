"""Validate against the challenge's OWN dev definition, not ours.

Why this exists
---------------
`src/prepare.clean()` drops rows with `trip_distance == 0` or
`passenger_count == 0`. That is right for *training* — they are garbage — but
wrong for *validation*, because the grader does not apply it. The starter's
`data/download_data.py` keeps those rows, and `trip_distance` is not even a
field we receive at inference time, so they will be in Eval.

Measured on the real dev window, that filter removes 2.7% of rows whose MAE
is 1.09x the rest, flattering the reported number:

    our cleaning   1,197,687 rows   256.8 s
    official       1,230,911 rows   257.4 s   <- what the grader will see

0.6s is small, but a number you report should be the number they compute.

Speed
-----
`grade.py` calls `predict()` row by row, which is the contract but takes ~6.5
minutes per 50k rows — 2.5+ hours for full dev. This script uses a vectorized
mirror of `predict()` and *asserts it is row-exact* against the real function
on a random sample before trusting it, so a divergence fails loudly instead of
silently reporting a number for code we do not ship.

Usage:
    python scripts/validate_official.py                 # full dev
    python scripts/validate_official.py --sample 50000  # eval-sized sample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

CUTOFF = pd.Timestamp("2023-12-18")
RAW = HERE / "data" / "raw" / "yellow_tripdata_2023-12.parquet"
OFFICIAL_DEV = HERE / "data" / "dev_official.parquet"


def build_official_dev() -> pd.DataFrame:
    """Reproduce the starter's download_data.py cleaning exactly.

    Dev is pickups >= 2023-12-18, so only the December file can contribute.
    """
    df = pd.read_parquet(RAW, columns=[
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID", "passenger_count"])
    duration = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    clean = pd.DataFrame({
        "pickup_zone": df["PULocationID"].astype("int32"),
        "dropoff_zone": df["DOLocationID"].astype("int32"),
        "requested_at": df["tpep_pickup_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "passenger_count": df["passenger_count"].fillna(1).astype("int8"),
        "duration_seconds": duration.astype("float64"),
        "_ts": df["tpep_pickup_datetime"],
    })
    mask = (
        (clean["duration_seconds"] >= 30)
        & (clean["duration_seconds"] <= 3 * 3600)
        & (clean["pickup_zone"].between(1, 265))
        & (clean["dropoff_zone"].between(1, 265))
        & (clean["_ts"].dt.year == 2023)
    )
    dev = clean.loc[mask & (clean["_ts"] >= CUTOFF)].drop(columns=["_ts"])
    return dev.reset_index(drop=True)


def vectorized_predict(dev: pd.DataFrame, P) -> np.ndarray:
    """Mirror of predict.predict over a whole frame. Verified row-exact below."""
    ts = pd.to_datetime(dev["requested_at"]).dt.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward")
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    pu = dev["pickup_zone"].to_numpy().clip(1, 265)
    do = dev["dropoff_zone"].to_numpy().clip(1, 265)
    pax = np.maximum(1, dev["passenger_count"].to_numpy().astype(int))

    T, TH = P._TABLES, P._THRESHOLDS
    m1, c1 = T["l1"]; m2, c2 = T["l2"]; m3, c3 = T["l3"]; m4, c4 = T["l4"]; m5, c5 = T["l5"]
    base = np.full(len(dev), T["global_mean"], dtype=np.float64)
    for cond, vals in (
        (c5[do] >= TH["l5_do"], m5[do]),
        (c4[pu] >= TH["l4_pu"], m4[pu]),
        (c3[pu, do] >= TH["l3_pu_do"], m3[pu, do]),
        (c2[pu, do, hour] >= TH["l2_pu_do_hour"], m2[pu, do, hour]),
        (c1[pu, do, hour, dow] >= TH["l1_pu_do_hour_dow"], m1[pu, do, hour, dow]),
    ):
        base[cond] = vals[cond]

    lat = np.array([P._zones.get(z, P._FALLBACK)[0] for z in range(266)])
    lon = np.array([P._zones.get(z, P._FALLBACK)[1] for z in range(266)])
    pla, plo, dla, dlo = lat[pu], lon[pu], lat[do], lon[do]
    r1, r2 = np.radians(pla), np.radians(dla)
    a = (np.sin(np.radians(dla - pla) / 2) ** 2
         + np.cos(r1) * np.cos(r2) * np.sin(np.radians(dlo - plo) / 2) ** 2)
    hav = 2 * P.EARTH_KM * np.arcsin(np.sqrt(a))
    X = np.column_stack([
        pu, do, hour, dow, pax, pla, plo, dla, dlo, hav,
        (dow >= 5).astype(int),
        ((dow < 5) & (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19)))).astype(int),
        np.log1p(c3[pu, do].astype(np.int64)),
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ]).astype(np.float64)
    return np.clip(base + P._MODEL.predict(X), 30.0, 14400.0), base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="evaluate a random sample (Eval is ~50k trips)")
    ap.add_argument("--check-n", type=int, default=400,
                    help="rows used to verify the mirror against predict()")
    args = ap.parse_args()

    if OFFICIAL_DEV.exists():
        dev = pd.read_parquet(OFFICIAL_DEV)
    else:
        if not RAW.exists():
            raise SystemExit(f"missing {RAW}; run scripts/download_data.py first")
        dev = build_official_dev()
        dev.to_parquet(OFFICIAL_DEV, index=False)
    if args.sample and args.sample < len(dev):
        dev = dev.sample(n=args.sample, random_state=42).reset_index(drop=True)

    import predict as P
    pred, base = vectorized_predict(dev, P)

    # The mirror is only trustworthy if it reproduces the shipped function.
    idx = np.random.default_rng(0).choice(len(dev), min(args.check_n, len(dev)), replace=False)
    ref = np.array([P.predict(r) for r in dev.iloc[idx][
        ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count"]].to_dict("records")])
    drift = float(np.abs(ref - pred[idx]).max())
    if drift > 1e-9:
        raise SystemExit(f"mirror diverged from predict() by {drift}; refusing to report a number")
    print(f"mirror verified row-exact vs predict() on {len(idx)} rows (max diff {drift:g})")

    y = dev["duration_seconds"].to_numpy()
    m = lambda p: float(np.mean(np.abs(p - y)))
    results = {
        "n_dev_official": int(len(dev)),
        "cleaning": "starter download_data.py (no trip_distance / passenger_count filter)",
        "global_mean_mae": m(np.full(len(y), P._TABLES["global_mean"])),
        "lookup_only_mae": m(np.clip(base, 30, 14400)),
        "shipped_mae": m(pred),
        "model_n_features": int(P._MODEL.n_features_in_),
    }
    print(f"\nofficial dev rows: {len(dev):,}")
    print(f"  global mean                 {results['global_mean_mae']:7.1f} s")
    print(f"  lookup only                 {results['lookup_only_mae']:7.1f} s")
    print(f"  shipped (lookup + residual) {results['shipped_mae']:7.1f} s")
    if not args.sample:
        out = HERE / "artifacts" / "validation_official.json"
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
