"""Hierarchical backoff lookup tables for trip duration.

Levels (fine -> coarse), each a dense float32 mean array + uint32 count array:
    L1 (pu, do, hour, dow)  -> 266*266*24*7  cells
    L2 (pu, do, hour)       -> 266*266*24     cells
    L3 (pu, do)             -> 266*266        cells
    L4 (pu,)                -> 266            cells
    L5 (do,)                -> 266            cells
    L6 global mean

Prediction walks fine->coarse and takes the first level whose cell count
meets that level's min_count threshold. Dense arrays make inference a few
integer ops + memory reads (no dict hashing), and keep the Docker image
small (~120 MB for the full 11.5-month tables).

Artifacts: artifacts/lookups.npz, artifacts/meta.json (thresholds, global mean).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ART = HERE / "artifacts"

N_ZONE = 266  # index directly by LocationID; slot 0 unused

DEFAULT_THRESHOLDS = {
    "l1_pu_do_hour_dow": 8,
    "l2_pu_do_hour": 5,
    "l3_pu_do": 3,
    "l4_pu": 2,
    "l5_do": 2,
}


def _empty(shape):
    return np.full(shape, np.nan, dtype=np.float32), np.zeros(shape, dtype=np.uint32)


def build_lookups(df: pd.DataFrame) -> dict:
    """df needs columns: pu, do, hour, dow, duration_s. Returns level dict."""
    pu = df["pu"].to_numpy(dtype=np.int64)
    do = df["do"].to_numpy(dtype=np.int64)
    hr = df["hour"].to_numpy(dtype=np.int64)
    dw = df["dow"].to_numpy(dtype=np.int64)
    y = df["duration_s"].to_numpy(dtype=np.float64)

    tables = {}
    tables["global_mean"] = float(y.mean())

    def agg(keys, shape):
        means, counts = _empty(shape)
        code = np.zeros(len(df), dtype=np.int64)
        mult = 1
        for k, dim in reversed(keys):
            code += k * mult
            mult *= dim
        # bincount-based groupby (much faster than pandas for dense codes)
        order = np.argsort(code, kind="stable")
        sc, sy = code[order], y[order]
        bounds = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
        grp = np.add.reduceat(sy, bounds)
        cnt = np.diff(np.r_[bounds, len(sc)])
        uniq = sc[bounds]
        means.flat[uniq] = (grp / cnt).astype(np.float32)
        counts.flat[uniq] = cnt.astype(np.uint32)
        return means, counts

    tables["l1"] = agg([(pu, N_ZONE), (do, N_ZONE), (hr, 24), (dw, 7)],
                       (N_ZONE, N_ZONE, 24, 7))
    tables["l2"] = agg([(pu, N_ZONE), (do, N_ZONE), (hr, 24)],
                       (N_ZONE, N_ZONE, 24))
    tables["l3"] = agg([(pu, N_ZONE), (do, N_ZONE)], (N_ZONE, N_ZONE))
    tables["l4"] = agg([(pu, N_ZONE)], (N_ZONE,))
    tables["l5"] = agg([(do, N_ZONE)], (N_ZONE,))
    return tables


def save_lookups(tables: dict, thresholds: dict, path: Path = ART / "lookups.npz") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"global_mean": np.float32(tables["global_mean"])}
    for name in ("l1", "l2", "l3", "l4", "l5"):
        means, counts = tables[name]
        payload[f"{name}_means"] = means
        payload[f"{name}_counts"] = counts
    np.savez_compressed(path, **payload)
    meta = {"thresholds": thresholds, "global_mean": float(tables["global_mean"])}
    (path.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_lookups(path: Path = ART / "lookups.npz") -> tuple[dict, dict]:
    z = np.load(path)
    tables = {"global_mean": float(z["global_mean"])}
    for name in ("l1", "l2", "l3", "l4", "l5"):
        tables[name] = (z[f"{name}_means"], z[f"{name}_counts"])
    meta = json.loads((path.parent / "meta.json").read_text())
    return tables, meta["thresholds"]


def batch_lookup_predict(pu, do, hour, dow, tables, thresholds) -> np.ndarray:
    """Vectorized hierarchical prediction. Inputs are int64 numpy arrays."""
    pu = np.asarray(pu, dtype=np.int64)
    do = np.asarray(do, dtype=np.int64)
    hour = np.asarray(hour, dtype=np.int64)
    dow = np.asarray(dow, dtype=np.int64)
    pred = np.full(len(pu), tables["global_mean"], dtype=np.float64)
    # coarse -> fine; each finer level overwrites where it has enough support
    steps = [
        ("l5", (do,), thresholds["l5_do"]),
        ("l4", (pu,), thresholds["l4_pu"]),
        ("l3", (pu, do), thresholds["l3_pu_do"]),
        ("l2", (pu, do, hour), thresholds["l2_pu_do_hour"]),
        ("l1", (pu, do, hour, dow), thresholds["l1_pu_do_hour_dow"]),
    ]
    for name, keys, thr in steps:
        means, counts = tables[name]
        # row-major flat index from the level's dense shape
        idx = np.zeros(len(pu), dtype=np.int64)
        stride = 1
        for k, d in zip(reversed(keys), reversed(means.shape)):
            idx += k * stride
            stride *= d
        ok = counts.flat[idx] >= thr
        pred[ok] = means.flat[idx[ok]]
    return pred


def single_lookup_predict(pu: int, do: int, hour: int, dow: int,
                          tables, thresholds) -> float:
    pu = min(max(pu, 0), N_ZONE - 1)
    do = min(max(do, 0), N_ZONE - 1)
    hour = min(max(hour, 0), 23)
    dow = min(max(dow, 0), 6)
    (m1, c1), (m2, c2), (m3, c3), (m4, c4), (m5, c5) = (
        tables["l1"], tables["l2"], tables["l3"], tables["l4"], tables["l5"])
    if c1[pu, do, hour, dow] >= thresholds["l1_pu_do_hour_dow"]:
        return float(m1[pu, do, hour, dow])
    if c2[pu, do, hour] >= thresholds["l2_pu_do_hour"]:
        return float(m2[pu, do, hour])
    if c3[pu, do] >= thresholds["l3_pu_do"]:
        return float(m3[pu, do])
    if c4[pu] >= thresholds["l4_pu"]:
        return float(m4[pu])
    if c5[do] >= thresholds["l5_do"]:
        return float(m5[do])
    return float(tables["global_mean"])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE / "src"))
    from features import MIN_DURATION_S, MAX_DURATION_S  # noqa: F401  (docs)
    print("import ok; run via scripts/train_all.py")
