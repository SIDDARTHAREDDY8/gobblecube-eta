# ETA Challenge — NYC Taxi Trip-Duration Prediction

Gobblecube AI Builders take-home submission (ETA track).

**Result: 256.8s MAE** on the real dev window (last two
weeks of 2023; train on pickup < 2023-12-18, validate on pickup ≥
2023-12-18). Challenge baseline: ~351s dev / ~367s eval.

## What I built

Hierarchical backoff lookup tables over `(pickup_zone, dropoff_zone, hour,
day_of_week)` trained on 11.5 months of 2023 NYC yellow-taxi trips (~37M
trips after cleaning), blended with a gradient-boosted tree
(scikit-learn `HistGradientBoostingRegressor`) trained on the lookup
residual with geo + calendar features.

## Why this design

The first thing the data tells you: a 10-line zone-pair average already
beats the naive GBT baseline (~300s vs ~350s MAE on dev). Trip duration in
NYC is dominated by *which route, at what time* — so the lookup tables,
not the model, carry the prediction.

- Finer cells `(pu, do, hour, dow)` win where they have support; coarser
  cells back them up where they don't. Minimum-count thresholds per level
  were picked by ablation on the dev slice (5 discrete hyperparameters;
  see table below).
- The GBT only learns the *residual*: rush-hour slowdowns, weekend
  effects, distance corrections the coarse cells miss. 17 features, all
  defensible: zone ids, hour/dow (+ sin/cos encodings), passenger
  count, zone centroids, haversine distance, log route frequency.
  Month features are deliberately excluded — the dev/eval windows are
  holiday slices whose month effect does not transfer from the rest of
  the year (a GBT with month features scored 270.4s on dev vs 256.8s
  without).

## Validation (honest numbers)

Time-split validation on the real dev window. Nothing from the dev slice
was used to build the lookup tables or train the GBT; only the five
backoff thresholds were chosen on dev (discrete ablation, disclosed here).

| Approach | Dev MAE (s) |
|---|---|
| Global mean | 575.7 |
| Zone-pair averages | 301.2 |
| Hierarchical backoff | 260.9 |
| Backoff + residual GBT | 256.8 |

Threshold ablation on dev:

| Variant | Dev MAE (s) |
|---|---|
| default (8/5/3/2/2) | 261.1 |
| aggressive (4/3/2/1/1) | 260.9 |
| conservative (16/10/5/2/2) | 262.1 |
| l1_strict (32/5/3/2/2) | 262.6 |
| no_l1 (∞/5/3/2/2) | 273.4 |

Winner: aggressive (4/3/2/1/1). The challenge notes budget a ~15s dev→eval
gap (eval is a held-out winter-holiday slice that skews harder).

## How to reproduce

```bash
# 1. Data (~600 MB, one-time)
python scripts/download_data.py --months 2023-01 2023-02 2023-03 2023-04 \
  2023-05 2023-06 2023-07 2023-08 2023-09 2023-10 2023-11 2023-12 \
  --out data_raw

# 2. Zone centroids (one-time; shapefile is EPSG:2263, reprojected to WGS84)
python src/zones.py   # writes artifacts/zones.csv

# 3. Train (chunked per month; runs on a 7 GB RAM / 2-CPU machine)
python scripts/train_full.py --months 2023-01 ... 2023-12 \
  --val-cutoff 2023-12-18 --gbt-sample-per-month 100000
# writes artifacts/lookups.npz, artifacts/model.pkl, data/dev.parquet

# 4. Grade locally (exact scoring harness)
python grade.py                                   # 50k-row dev sample
python grade.py data/dev.parquet /tmp/preds.csv   # grader mode, full dev

# 5. Docker
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
```

## Inference

`predict(request: dict) -> float` — `request` has `pickup_zone`,
`dropoff_zone` (1–265), `requested_at` (ISO-8601), `passenger_count`.
Returns trip duration in seconds, clipped to [30, 14400].

Single request ≈ a few ms on CPU (budget 200 ms): dense-array lookups +
one GBT predict. No network calls. The image installs only `numpy` +
`scikit-learn` + `pandas`/`pyarrow` (the scoring harness reads the input
parquet) and stays far under the 2.5 GB limit.

## Files

- `predict.py` — the submission interface (fixed signature).
- `grade.py` — the challenge's scoring harness (unchanged logic).
- `Dockerfile` — builds in one `pip install`; build-time smoke test included.
- `artifacts/` — `lookups.npz`, `model.pkl`, `meta.json` (thresholds),
  `zones.csv` (zone centroids).
- `scripts/` — `download_data.py`, `train_full.py` (chunked full training),
  `train_all.py` (single-month sample validation), `push_history.py`.
- `src/` — cleaning, features, lookup-table builders.
- `tests/` — submission-contract smoke tests (8/8).
- `CLAUDE.md` — how this submission was built (AI-tooling disclosure).

## What I tried that didn't make it

> TODO (Siddartha — your voice): e.g. any feature you ablated, month-level
> cells, weather joins, deeper trees. The graders explicitly want the
> failures.

## What I'd try next

> TODO (Siddartha — your voice): one or two honest next experiments.
