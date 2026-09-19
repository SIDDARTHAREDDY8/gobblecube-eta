# CLAUDE.md — how this submission was built

This file exists because the challenge asks for it: the graders read the
trajectory, not just the final state.

## How it was built

This submission was built with an AI pair-programmer (Muse) doing the
implementation — code, experiments, and lab notebook — under the
candidate's direction. The trajectory below matches the git log.

## The trajectory (matches the git log)

1. **Pipeline skeleton** — features, hierarchical lookups, residual GBT,
   `predict.py`, Dockerfile, smoke tests. First commit, deliberately rough.
2. **Cleaning + features** — matched the challenge's cleaning rules
   (30s–3h duration, 2023 pickups), vectorized geo features, cyclic time
   encodings.
3. **HGBR fix** — scikit-learn's native categorical support caps at 255
   categories; NYC has 265 zones. Zones went in as numerics (their spatial
   signal also rides on lat/lon + haversine); hour/dow stayed native
   categoricals.
4. **CRS fix** — the taxi-zone shapefile is EPSG:2263 (state-plane feet),
   not WGS84. Centroids were silently wrong until reprojected with pyproj.
   This was the biggest correctness bug caught before submission.
5. **Sample validation** — June-2023 slice: 240.3s blended MAE
   (591.9 global-mean / 290.2 zone-pair / 255.8 hierarchical ladder).
6. **Full-scale training** — all 12 months of 2023, validated on the real
   dev window (pickup >= 2023-12-18). See README for the final number.
7. **Holiday-slice fix** — the residual GBT hurt on the December dev until
   month features were removed (270.4s → 256.8s). Final model uses 17
   features, no month.

## What the AI contributed

- The vectorized groupby/lookup code and the Docker plumbing.
- Catching the CRS bug by checking units, not just shapes.
- Keeping every experiment reproducible (one command per stage).
- The inference-venv grader-path validation, which caught a real bug:
  `grade.py` needed pandas, which was missing from the image.

## Candidate's own sections

- [x] Why this approach (motivation, design judgment)
- [x] What was tried that didn't make it
- [x] What would be tried next with more time

### Why this approach

The data made the call. A 10-line zone-pair average beat the naive GBT baseline, so I stopped trying to make the model smart and let lookup tables carry the prediction instead. The hierarchy is the whole bet: (pickup_zone, dropoff_zone, hour, day_of_week) cells where they have support, coarser cells backing them up where they don't, five minimum-count thresholds picked by ablation on the dev slice. The GBT only learns the residual: rush-hour slowdowns, weekend effects, distance corrections the coarse cells miss. Boring design, best number.

### What was tried that didn't make it

Month as a GBT feature hurt: 270.4s on dev vs 256.8s without it, because December is a holiday slice and the month effect from the rest of the year doesn't transfer. A pure GBT carrying the whole prediction landed around 300s on the sample slice vs 255.8s for the hierarchical lookups alone, so the GBT got demoted to residual duty. And the zone centroids were silently wrong until I reprojected the shapefile from EPSG:2263 (state-plane feet) — caught it by checking units.

### What would be tried next with more time

Weather joins first — precipitation and snow hit trip times hard and December has plenty of it. Then finer lookup cells with shrinkage instead of hard minimum-count cutoffs, so sparse cells borrow strength instead of backing off. And a deeper residual model: the 17-feature GBT is small, and with the lookup tables handling the base rate, a bigger model on the residual might squeeze out a few more seconds.

## Traps in this repo (found the hard way)

**Training features must match `predict.py`'s vector, name for name and in
order.** `src/features.FEATURE_NAMES` once carried 20 features (month,
mon_sin, mon_cos) while `predict.py` built 17 without them. Consequence:
running the README's own reproduce command wrote a 20-feature `model.pkl`
that `predict.py` could not load — the documented path broke the
submission, and `artifacts/validation_results.json` (blended 270.4s, worse
than lookups alone) was that broken model's score. Fixed, and
`tests/test_submission.py` now asserts the two agree. Do not add a feature
to one side only.

**Validate with the starter's cleaning, not ours.** `src/prepare.clean()`
drops `trip_distance == 0` and `passenger_count == 0`. Right for training,
wrong for reporting: the grader keeps those rows, and `trip_distance` is
not an inference field. It is 2.7% of dev at 1.09x the MAE. Use
`scripts/validate_official.py` for any number you intend to publish.

**`grade.py` is the contract but is slow for sweeps** — row-by-row
`predict()` is ~6.5 min per 50k rows, 2.5+ h for full dev.
`scripts/validate_official.py` vectorizes it and finishes in ~10 s, but it
asserts row-exactness against the real `predict()` first and refuses to
print a number if they diverge. Keep that check if you touch it.

**Numbers to beat:** official dev 257.2s shipped, 261.7s lookups only,
576.2s global mean. Challenge baseline ~351s dev / ~367s eval.
