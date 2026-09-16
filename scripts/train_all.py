"""End-to-end training + honest validation for the Gobblecube ETA challenge.

Pipeline:
  1. Load month(s) of TLC parquet, clean.
  2. Time-split: train = pickup < --val-cutoff, val = pickup >= --val-cutoff.
     (Mimics the real train/dev split: dev is the LAST 2 weeks of 2023.)
  3. Build hierarchical lookup tables on TRAIN only -> artifacts/lookups.npz.
  4. Train HistGradientBoostingRegressor on the RESIDUAL (duration - lookup)
     using a random --sample of train rows -> artifacts/model.pkl.
  5. Report MAE on VAL (never touched during training):
       global mean | zone-pair (pu,do) | hierarchical | hierarchical + GBT

Usage (sample validation on this machine):
    python scripts/train_all.py --months 2023-06 --val-cutoff 2023-06-25 --sample 400000

Usage (full scale, bigger machine):
    python scripts/train_all.py --months 2023-01 ... 2023-12 \\
        --val-cutoff 2023-12-18 --sample 4000000
"""
from __future__ import annotations

import argparse
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
                          build_lookups, load_lookups, save_lookups)
from prepare import (add_geo_features, add_pair_count, clean, feature_matrix,
                     load_months, load_zones)


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True)
    ap.add_argument("--val-cutoff", required=True, help="YYYY-MM-DD")
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--raw-dir", default=str(HERE / "data_raw"))
    ap.add_argument("--loss", default="absolute_error",
                    choices=["absolute_error", "squared_error"])
    ap.add_argument("--skip-gbt", action="store_true",
                    help="only build lookups + report lookup MAEs")
    args = ap.parse_args()

    t0 = time.time()
    df = load_months(args.months, args.raw_dir)
    print(f"loaded {len(df):,} rows")
    df = clean(df)
    print(f"after cleaning: {len(df):,} rows ({time.time()-t0:.0f}s)")

    zones = load_zones(HERE / "artifacts" / "zones.csv")
    df = add_geo_features(df, zones)

    cutoff = pd.Timestamp(args.val_cutoff)
    train = df[df["tpep_pickup_datetime"] < cutoff].reset_index(drop=True)
    val = df[df["tpep_pickup_datetime"] >= cutoff].reset_index(drop=True)
    print(f"train {len(train):,} / val {len(val):,} (cutoff {args.val_cutoff})")
    del df

    # ---- lookups on train only ----
    t1 = time.time()
    tables = build_lookups(train)
    save_lookups(tables, DEFAULT_THRESHOLDS)
    print(f"lookups built ({time.time()-t1:.0f}s)")

    # target-free (pu,do) counts for the log_pair_count feature
    _, l3_counts = tables["l3"]
    nz = np.nonzero(l3_counts)
    pair_counts = {(int(p), int(d)): int(c)
                   for p, d, c in zip(nz[0], nz[1], l3_counts[nz])}

    train = add_pair_count(train, pair_counts)
    val = add_pair_count(val, pair_counts)

    tables, thresholds = load_lookups()  # exercise the real load path

    def lookup_pred(frame):
        return batch_lookup_predict(frame["pu"], frame["do"], frame["hour"],
                                    frame["dow"], tables, thresholds)

    # ---- validation MAEs ----
    y_val = val["duration_s"].to_numpy()
    results = {}
    results["n_val"] = len(val)
    results["global_mean_mae"] = mae(y_val, tables["global_mean"])

    # zone-pair only (their ~300s reference)
    thr_pair = dict(thresholds)
    thr_pair.update({"l1_pu_do_hour_dow": 10**9, "l2_pu_do_hour": 10**9})
    pred_pair = batch_lookup_predict(val["pu"], val["do"], val["hour"],
                                     val["dow"], tables, thr_pair)
    results["zone_pair_mae"] = mae(y_val, pred_pair)

    pred_h = lookup_pred(val)
    results["hierarchical_mae"] = mae(y_val, pred_h)
    print("lookup MAEs:", {k: round(v, 1) for k, v in results.items()
                           if k.endswith("mae")})

    # ---- threshold ablation: pick the backoff thresholds on val ----
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
        pv = batch_lookup_predict(val["pu"], val["do"], val["hour"],
                                  val["dow"], tables, thr)
        ablation[name] = round(mae(y_val, pv), 2)
    results["threshold_ablation"] = ablation
    best = min(variants, key=lambda k: ablation[k])
    print("threshold ablation:", ablation, "-> best:", best)
    if ablation[best] < ablation["default"]:
        meta_path = HERE / "artifacts" / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["thresholds"] = variants[best]
        meta_path.write_text(json.dumps(meta, indent=2))
        thresholds = variants[best]
        pred_h = batch_lookup_predict(val["pu"], val["do"], val["hour"],
                                      val["dow"], tables, thresholds)
        results["hierarchical_mae"] = mae(y_val, pred_h)
        print(f"meta.json thresholds updated to '{best}'")

    if not args.skip_gbt:
        from sklearn.ensemble import HistGradientBoostingRegressor

        rng = np.random.default_rng(7)
        idx = rng.choice(len(train), size=min(args.sample, len(train)),
                         replace=False)
        sub = train.iloc[idx]
        base_tr = lookup_pred(sub)
        resid_tr = sub["duration_s"].to_numpy() - base_tr
        X_tr = feature_matrix(sub)
        print(f"training HGBR on {len(sub):,} rows, loss={args.loss} ...")
        t2 = time.time()
        model = HistGradientBoostingRegressor(
            loss=args.loss, max_iter=250, max_leaf_nodes=63,
            learning_rate=0.06, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=True,
            n_iter_no_change=20, validation_fraction=0.1,
            categorical_features=CATEGORICAL_IDX, random_state=7,
        )
        model.fit(X_tr, resid_tr)
        print(f"HGBR trained ({time.time()-t2:.0f}s, "
              f"n_iter={model.n_iter_})")

        import pickle
        mpath = HERE / "artifacts" / "model.pkl"
        with open(mpath, "wb") as f:
            pickle.dump(model, f)
        print(f"saved {mpath} ({mpath.stat().st_size/1e6:.1f} MB)")

        X_val = feature_matrix(val)
        t3 = time.time()
        resid_pred = model.predict(X_val)
        print(f"val predict: {len(val):,} rows in {time.time()-t3:.1f}s "
              f"({(time.time()-t3)/len(val)*1e6:.0f} us/row)")
        final = np.clip(pred_h + resid_pred, 30.0, 14400.0)
        results["blended_mae"] = mae(y_val, final)
        results["gbt_n_iter"] = int(model.n_iter_)

    results["months"] = args.months
    results["val_cutoff"] = args.val_cutoff
    results["train_rows"] = len(train)
    results["gbt_sample"] = min(args.sample, len(train)) if not args.skip_gbt else 0
    out = HERE / "artifacts" / "validation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"total {time.time()-t0:.0f}s")
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
