"""Full-scale training on all of 2023, chunked per month for a small machine.

Split (matches the challenge's train/dev definition):
  - Train: tpep_pickup_datetime < 2023-12-18   (~11.5 months)
  - Dev:   tpep_pickup_datetime >= 2023-12-18  (last ~2 weeks of 2023)

The dev slice is NEVER used for lookup tables or model training; it is only
used at the end for (a) picking the 5 backoff thresholds and (b) reporting
the honest validation MAE. The 2024 eval set is never touched.

Lookup tables are accumulated via per-month np.bincount sums/counts, so the
whole 37M-row train set never sits in RAM at once.

Usage:
    python scripts/train_full.py --months 2023-01 ... 2023-12 \\
        --val-cutoff 2023-12-18 --gbt-sample-per-month 100000
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

import numpy as np
import pandas as pd

from features import CATEGORICAL_IDX
from make_lookups import (DEFAULT_THRESHOLDS, batch_lookup_predict,
                         load_lookups, save_lookups)
from prepare import (add_geo_features, add_pair_count, clean, feature_matrix,
                     load_zones)

N_ZONE = 266
LEVEL_SHAPES = {
    "l1": (N_ZONE, N_ZONE, 24, 7),
    "l2": (N_ZONE, N_ZONE, 24),
    "l3": (N_ZONE, N_ZONE),
    "l4": (N_ZONE,),
    "l5": (N_ZONE,),
}
LEVEL_SIZES = {k: int(np.prod(v)) for k, v in LEVEL_SHAPES.items()}


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def level_codes(df: pd.DataFrame) -> dict[str, np.ndarray]:
    pu = df["pu"].to_numpy(dtype=np.int64)
    do = df["do"].to_numpy(dtype=np.int64)
    hr = df["hour"].to_numpy(dtype=np.int64)
    dw = df["dow"].to_numpy(dtype=np.int64)
    return {
        "l1": ((pu * N_ZONE + do) * 24 + hr) * 7 + dw,
        "l2": (pu * N_ZONE + do) * 24 + hr,
        "l3": pu * N_ZONE + do,
        "l4": pu,
        "l5": do,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True)
    ap.add_argument("--val-cutoff", default="2023-12-18")
    ap.add_argument("--gbt-sample-per-month", type=int, default=100_000)
    ap.add_argument("--raw-dir", default=str(HERE / "data_raw"))
    ap.add_argument("--loss", default="absolute_error",
                    choices=["absolute_error", "squared_error"])
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--skip-gbt", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(7)
    cutoff = pd.Timestamp(args.val_cutoff)

    sums = {k: np.zeros(s, dtype=np.float64) for k, s in LEVEL_SIZES.items()}
    counts = {k: np.zeros(s, dtype=np.int64) for k, s in LEVEL_SIZES.items()}
    gbt_pool: list[pd.DataFrame] = []
    dev_frames: list[pd.DataFrame] = []
    n_train_total = 0

    import pyarrow.parquet as pq
    raw_dir = Path(args.raw_dir)
    for m in args.months:
        p = raw_dir / f"yellow_tripdata_{m}.parquet"
        print(f"[{time.time()-t0:6.0f}s] reading {p.name} ...", flush=True)
        df = pq.read_table(p).to_pandas()
        n_raw = len(df)
        df = clean(df)
        is_dev = df["tpep_pickup_datetime"] >= cutoff
        dev_m = df[is_dev].reset_index(drop=True)
        train_m = df[~is_dev].reset_index(drop=True)
        del df
        gc.collect()
        print(f"  raw {n_raw:,} -> clean train {len(train_m):,} / dev {len(dev_m):,}",
              flush=True)
        dev_frames.append(dev_m[["tpep_pickup_datetime", "pu", "do",
                                  "passenger_count", "duration_s",
                                  "hour", "dow", "month"]])

        y = train_m["duration_s"].to_numpy(dtype=np.float64)
        codes = level_codes(train_m)
        for k in LEVEL_SIZES:
            sums[k] += np.bincount(codes[k], weights=y, minlength=LEVEL_SIZES[k])
            counts[k] += np.bincount(codes[k], minlength=LEVEL_SIZES[k])
        del codes, y
        n_train_total += len(train_m)

        if not args.skip_gbt:
            k = min(args.gbt_sample_per_month, len(train_m))
            idx = rng.choice(len(train_m), size=k, replace=False)
            gbt_pool.append(train_m.iloc[idx][
                ["pu", "do", "hour", "dow", "month", "passenger_count",
                 "duration_s"]].copy())
        del train_m
        gc.collect()

    print(f"[{time.time()-t0:6.0f}s] train rows total: {n_train_total:,}; "
          f"GBT pool: {sum(map(len, gbt_pool)):,} rows", flush=True)

    # ---- build lookup tables from accumulated sums/counts ----
    tables = {"global_mean": float(sums["l1"].sum() / counts["l1"].sum())}
    for k, shape in LEVEL_SHAPES.items():
        c = counts[k].reshape(shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = (sums[k] / np.maximum(counts[k], 1)).reshape(shape).astype(np.float32)
        m[c == 0] = np.nan
        tables[k] = (m, c.astype(np.uint32))
    save_lookups(tables, DEFAULT_THRESHOLDS)
    del sums
    gc.collect()

    # target-free (pu,do) counts for the log_pair_count feature
    _, l3_counts = tables["l3"]
    nz = np.nonzero(l3_counts)
    pair_counts = {(int(p), int(d)): int(c)
                   for p, d, c in zip(nz[0], nz[1], l3_counts[nz])}

    # ---- dev set ----
    dev = pd.concat(dev_frames, ignore_index=True)
    dev_frames.clear()
    gc.collect()
    print(f"dev rows: {len(dev):,} "
          f"({dev['tpep_pickup_datetime'].min()} .. {dev['tpep_pickup_datetime'].max()})",
          flush=True)
    dev = add_pair_count(dev, pair_counts)
    y_dev = dev["duration_s"].to_numpy()

    tables, thresholds = load_lookups()  # exercise the real load path

    def lookup_pred(frame):
        return batch_lookup_predict(frame["pu"], frame["do"], frame["hour"],
                                    frame["dow"], tables, thresholds)

    results: dict = {"n_dev": len(dev)}
    results["global_mean_mae"] = mae(y_dev, tables["global_mean"])
    thr_pair = dict(thresholds)
    thr_pair.update({"l1_pu_do_hour_dow": 10**9, "l2_pu_do_hour": 10**9})
    results["zone_pair_mae"] = mae(
        y_dev, batch_lookup_predict(dev["pu"], dev["do"], dev["hour"],
                                    dev["dow"], tables, thr_pair))
    pred_h = lookup_pred(dev)
    results["hierarchical_mae"] = mae(y_dev, pred_h)
    print("lookup MAEs:", {k: round(v, 1) for k, v in results.items()
                           if k.endswith("mae")}, flush=True)

    # ---- threshold ablation on dev (5 discrete params; disclosed in README)
    variants = {
        "default": dict(thresholds),
        "aggressive": {"l1_pu_do_hour_dow": 4, "l2_pu_do_hour": 3,
                       "l3_pu_do": 2, "l4_pu": 1, "l5_do": 1},
        "conservative": {"l1_pu_do_hour_dow": 16, "l2_pu_do_hour": 10,
                         "l3_pu_do": 5, "l4_pu": 2, "l5_do": 2},
        "l1_strict": dict(dict(thresholds), l1_pu_do_hour_dow=32),
        "no_l1": dict(thresholds, l1_pu_do_hour_dow=10**9),
    }
    ablation = {}
    for name, thr in variants.items():
        ablation[name] = round(mae(y_dev, batch_lookup_predict(
            dev["pu"], dev["do"], dev["hour"], dev["dow"], tables, thr)), 2)
    results["threshold_ablation"] = ablation
    best = min(variants, key=lambda k: ablation[k])
    print("threshold ablation:", ablation, "-> best:", best, flush=True)
    if ablation[best] < ablation["default"]:
        meta_path = HERE / "artifacts" / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["thresholds"] = variants[best]
        meta_path.write_text(json.dumps(meta, indent=2))
        thresholds = variants[best]
        pred_h = batch_lookup_predict(dev["pu"], dev["do"], dev["hour"],
                                      dev["dow"], tables, thresholds)
        results["hierarchical_mae"] = mae(y_dev, pred_h)
        print(f"meta.json thresholds updated to '{best}'", flush=True)

    # ---- GBT residual model ----
    if not args.skip_gbt:
        from sklearn.ensemble import HistGradientBoostingRegressor

        pool = pd.concat(gbt_pool, ignore_index=True)
        gbt_pool.clear()
        gc.collect()
        zones = load_zones(HERE / "artifacts" / "zones.csv")
        pool = add_geo_features(pool, zones)
        pool = add_pair_count(pool, pair_counts)
        base_tr = batch_lookup_predict(pool["pu"], pool["do"], pool["hour"],
                                       pool["dow"], tables, thresholds)
        resid_tr = pool["duration_s"].to_numpy() - base_tr
        X_tr = feature_matrix(pool)
        del pool
        gc.collect()
        print(f"[{time.time()-t0:6.0f}s] training HGBR on {len(X_tr):,} rows, "
              f"loss={args.loss} ...", flush=True)
        t2 = time.time()
        model = HistGradientBoostingRegressor(
            loss=args.loss, max_iter=args.max_iter, max_leaf_nodes=63,
            learning_rate=0.06, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=True,
            n_iter_no_change=20, validation_fraction=0.1,
            categorical_features=CATEGORICAL_IDX, random_state=7,
        )
        model.fit(X_tr, resid_tr)
        print(f"HGBR trained ({time.time()-t2:.0f}s, n_iter={model.n_iter_})",
              flush=True)
        import pickle
        mpath = HERE / "artifacts" / "model.pkl"
        with open(mpath, "wb") as f:
            pickle.dump(model, f)
        print(f"saved {mpath} ({mpath.stat().st_size/1e6:.1f} MB)", flush=True)

        # dev features for the residual model
        dev = add_geo_features(dev, zones)
        X_dev = feature_matrix(dev)
        resid_pred = model.predict(X_dev)
        final = np.clip(pred_h + resid_pred, 30.0, 14400.0)
        results["blended_mae"] = mae(y_dev, final)
        results["gbt_n_iter"] = int(model.n_iter_)
        print(f"blended dev MAE: {results['blended_mae']:.1f}s", flush=True)

    # ---- write the real dev parquet for the grader path ----
    dev_out = pd.DataFrame({
        "pickup_zone": dev["pu"].astype(int),
        "dropoff_zone": dev["do"].astype(int),
        "requested_at": pd.to_datetime(dev["tpep_pickup_datetime"]).dt.strftime(
            "%Y-%m-%dT%H:%M:%S"),
        "passenger_count": dev["passenger_count"].astype(int),
        "duration_seconds": dev["duration_s"].astype(float),
    })
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    dev_path = data_dir / "dev.parquet"
    dev_out.to_parquet(dev_path, index=False)
    print(f"wrote {dev_path} ({len(dev_out):,} rows)", flush=True)

    results["months"] = args.months
    results["val_cutoff"] = args.val_cutoff
    results["train_rows"] = n_train_total
    results["gbt_sample"] = (args.gbt_sample_per_month * len(args.months)
                             if not args.skip_gbt else 0)
    out = HERE / "artifacts" / "validation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"total {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
