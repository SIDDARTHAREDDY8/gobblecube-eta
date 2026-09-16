# ETA Challenge — Submission README

> NOTE (Siddartha): this is a skeleton. The challenge weights the README
> second only to the final score, and the FAQ says it should explain *what
> you tried, what failed, and what the next experiment would be*. Write it
> in your own voice — short sentences. The technical facts below are
> accurate; the story around them is yours to tell.

## What I built

Hierarchical backoff lookup tables over `(pickup_zone, dropoff_zone, hour,
day_of_week)` trained on 11.5 months of 2023 NYC yellow-taxi trips, blended
with a gradient-boosted tree (scikit-learn `HistGradientBoostingRegressor`)
trained on the lookup residual with geo + calendar features.

## Why this design

- The first thing the data tells you: a 10-line zone-pair average already
  beats the naive GBT baseline (~300s vs ~350s MAE on Dev). Trip duration in
  NYC is dominated by *which route at what time* — so the lookup tables,
  not the model, carry the prediction.
- Finer cells `(pu, do, hour, dow)` win where they have support; coarser
  cells back them up where they don't. Minimum-count thresholds per level
  were picked by ablation on a held-out time slice (see table below).
- The GBT only has to learn the *residual*: rush-hour slowdowns, weekend
  effects, distance corrections the coarse cells miss. 20 features, all
  defensible: zone ids, hour/dow/month (+ sin/cos encodings), passenger
  count, zone centroids, haversine distance, log route frequency.

## What I tried that didn't make it

- (TODO: fill in — e.g. any feature you ablated, e.g. month-level cells,
  weather joins, deeper trees. The graders explicitly want the failures.)

## Validation

Local time-split validation (train on pickup < 2023-12-18, validate on the
last two weeks of 2023 — the real Dev window):

| Approach | Val MAE (s) |
|---|---|
| Global mean | TODO |
| Zone-pair averages | TODO |
| Hierarchical backoff | TODO |
| Backoff + residual GBT | TODO |

(Reference: challenge README reports ~580 / ~300 / ~350 for the same three
on their Dev slice; our numbers come from the same data and cleaning.)

Budget ~15s dev→eval gap per the challenge notes.

## How to reproduce

See `SUBMISSION-PLAN.md` for the exact commands (data download → zone
centroids → lookup tables → GBT training → Docker build → grading).

## Inference

`predict(request: dict) -> float`. Single request ≈ a few ms on CPU
(budget 200 ms): dense-array lookups + one GBT predict. No network calls,
no pandas at inference — only `numpy` + `scikit-learn` in the image.

## What I'd try next

- (TODO: one or two honest next experiments, e.g. LightGBM with tuned
  hyperparams, per-borough models, or a small MLP on the residual.)
