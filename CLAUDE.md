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

## Candidate's own sections (TODO)

<!-- The candidate fills these in their own words before submitting. -->

- [ ] Why this approach (motivation, design judgment)
- [ ] What was tried that didn't make it
- [ ] What would be tried next with more time
