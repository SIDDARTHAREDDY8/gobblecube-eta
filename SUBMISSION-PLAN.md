# SUBMISSION-PLAN.md — Gobblecube ETA Challenge

## 1. What the solution does

**Approach: hierarchical backoff lookup + residual gradient-boosted tree.**

1. **Lookup tables (the main signal).** Mean trip duration per cell, backed
   off fine → coarse with per-level minimum-count thresholds:
   `(pu, do, hour, dow) → (pu, do, hour) → (pu, do) → (pu) → (do) → global`.
   Stored as dense `float32`/`uint32` arrays → O(1) inference, ~100 MB on
   disk for the full 11.5-month train set. Thresholds were chosen by
   ablation on a held-out time slice (see §3).
2. **Residual GBT (the correction).** `HistGradientBoostingRegressor`
   (scikit-learn, `absolute_error` loss = MAE objective) trained on
   `duration − lookup_pred` with 20 features: zone ids, hour/dow/month
   (+ sin/cos encodings), passenger count, zone centroids, haversine
   distance, weekend/rush flags, log route frequency. Native categorical
   support for zone/time ids.
3. **predict.py**: lookup + GBT residual, clipped to [30s, 4h]. Single
   request ≈ a few ms on CPU (budget 200 ms; measured single-threaded:
   p50 3.6 ms, p99 11 ms, max 13 ms over 200 requests — the Dockerfile pins
   OMP_NUM_THREADS=1 because multi-threaded BLAS caused 1s+ p99 spikes under
   CPU contention on a 2-CPU VM). Inference deps:
   only `numpy` + `scikit-learn`. No network calls, no pandas at inference.

Why this shape: the challenge's own numbers say a 10-line zone-pair average
(~300s Dev MAE) already beats their naive GBT (~350s). Duration in NYC is
dominated by *which route at what time* — so the lookup carries the
prediction and the model only learns corrections. This also keeps the
Docker image small and inference fast.

## 2. Repo layout

```
predict.py            # submission interface: predict(request: dict) -> float
grade.py              # challenge's exact scoring harness (unchanged)
Dockerfile            # python:3.12-slim + numpy/sklearn, ENTRYPOINT grade.py
requirements.txt      # training deps (pinned, verified)
requirements-infer.txt# inference deps (pinned; keep sklearn in lockstep!)
src/features.py       # shared conventions: cleaning bounds, feature order
src/prepare.py        # cleaning + feature engineering (training side)
src/make_lookups.py   # dense hierarchical lookup tables
src/zones.py          # zone centroids from the official TLC shapefile
scripts/download_data.py  # fetch TLC parquet months
scripts/train_all.py      # end-to-end: clean → split → lookups → GBT → report
tests/test_submission.py  # smoke tests: contract, latency, no-network, MAE sanity
artifacts/            # zones.csv, lookups.npz, model.pkl, meta.json,
                      # validation_results.json (built by the pipeline)
```

## 3. Sample validation (this machine, 2026-09-16)

One real month of TLC data — `yellow_tripdata_2023-06.parquet` (55 MB,
~3.4M trips) — cleaned with the challenge's rules (30s–3h, zones 1–265,
2023 pickups), time-split train < 2023-06-25 / val ≥ 2023-06-25
(~2.7M / ~0.6M rows). GBT trained on a 400k-row sample of train.

| Approach | Val MAE (s) |
|---|---|
| Global mean | 591.9 |
| Zone-pair averages | 290.2 |
| Hierarchical backoff | 255.8 |
| Backoff + residual GBT | **239.8** |

Threshold ablation (val MAE, lower is better): default 258.8, aggressive
258.5, conservative 262.7, l1_strict(≥32) 256.8, **no_l1 255.8** → the
`(pu,do,hour,dow)` cells overfit on one month of train, so the shipped
thresholds disable L1 (`meta.json`). The GBT then carries the time-of-day
signal via its hour/dow features instead. **The full-scale run re-runs
this ablation on the real Dec-2023 dev slice** — with 11.5 months of
support per cell, L1 may earn its place back; let the data decide there.

**How to read these numbers honestly:** the val slice is the last 6 days
of *June* 2023, not the real Dev (last 2 weeks of *Dec* 2023), so the
absolute values aren't directly comparable to the challenge's ~580/300/350
Dev references. What the sample validates: the pipeline mechanics
end-to-end on real data, the ordering (lookup ≫ global mean,
blend ≥ lookup), and that inference meets the latency/contract tests.
Expect a ~15s dev→eval gap on top (challenge's own guidance).

## 4. Full-scale reproduction (bigger machine)

Hardware: any laptop with ~16 GB RAM, or a free Colab/Kaggle CPU notebook.
~2–3 GB free disk for the 12 monthly parquets.

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Data: all of 2023 (~37M trips, ~600 MB parquet)
python scripts/download_data.py --months 2023-01 2023-02 2023-03 2023-04 \
  2023-05 2023-06 2023-07 2023-08 2023-09 2023-10 2023-11 2023-12 \
  --out data_raw

# 2. Zone centroids (one-time, ~1 min)
python src/zones.py

# 3. Train: lookups on all of train, GBT on a 2-4M row sample.
#    val-cutoff 2023-12-18 reproduces the real train/dev split
#    (dev = last 2 weeks of 2023). Expect ~30-60 min on a laptop CPU,
#    peak RAM ~8-10 GB during the L1 groupby.
python scripts/train_all.py \
  --months 2023-01 2023-02 2023-03 2023-04 2023-05 2023-06 \
           2023-07 2023-08 2023-09 2023-10 2023-11 2023-12 \
  --val-cutoff 2023-12-18 --sample 3000000

# 4. Smoke tests (needs artifacts/ from step 3)
python -m pytest tests/ -q   # or: python tests/test_submission.py
# (contract, latency < 200 ms, no-network-at-inference, MAE sanity)

# 5. Grade-path check with the challenge's own harness.
python grade.py                      # local: MAE on the val slice
```
Note: on the sample run the GBT hit `max_iter=250` (early stopping never
fired) — at full scale, try `--max-iter 400` if you have the CPU budget;
it was still improving.

Expected artifacts: `artifacts/lookups.npz` (~60–120 MB),
`artifacts/model.pkl` (~5–15 MB), `artifacts/zones.csv`,
`artifacts/meta.json`, `artifacts/validation_results.json`.

Expected val MAE (Dec 2023 slice): zone-pair ≈ 300s (challenge reference),
hierarchical a few seconds better, blend a few more. If the blend does not
beat the hierarchy on the real dev slice, ship the hierarchy alone —
`predict.py` already supports that (it just adds the residual).

## 5. Docker build + submit

```bash
docker build -t eta-submit .
# grade-path test exactly as the challenge runs it:
docker run --rm -v $(pwd)/data:/work eta-submit /work/dev.parquet /work/preds.csv
```

`data/dev.parquet` comes from the starter repo's `data/download_data.py`
(train/dev splits with the challenge's cleaning). Image should be ~600 MB
(≤ 2.5 GB budget). Then push the repo to GitHub and send the URL +
LinkedIn to agentic-hiring@gobblecube.ai (one submission per candidate).

## 6. What's left for you (Siddartha)

- [ ] **README.md** — skeleton is in the repo; write it in your own voice
      (what you tried, what failed, next experiment). Challenge weighs it
      #2 after score.
- [ ] Run §4 on your laptop or a Colab/Kaggle notebook (this VM has no
      Docker and only 2 CPUs / 7 GB RAM — full training wasn't feasible here).
- [ ] `docker build` + grader-path test (§5) — untested here (no Docker).
- [ ] Re-tune `--sample` and GBT `max_iter` if you have headroom; try
      LightGBM as a drop-in if you want (their requirements list it).
- [ ] Sanity-check `validation_results.json` on the real Dec-2023 dev slice
      before submitting; if blended MAE ≥ hierarchical MAE, consider
      shipping lookup-only.
- [ ] Push to GitHub (staged git history is already in the repo — real
      commits, keep appending honestly) and submit the URL. **You decide
      if/when anything goes out — nothing here submits externally.**

## 7. Open items / risks

- Docker image **not built here** (no Docker daemon on this VM). The
  Dockerfile mirrors the starter's tested pattern (same ENTRYPOINT/grade.py
  pathway), but build it yourself before submitting.
- `model.pkl` is scikit-learn-version-sensitive: `requirements-infer.txt`
  pins must stay in lockstep with `requirements.txt`.
- Artifacts (`lookups.npz`, `model.pkl`) are gitignored; for submission
  either commit them (check GitHub's 100 MB/file limit) or rebuild inside
  `docker build` from a data URL.
- Do **not** train on 2024 data, do not add external API calls at
  inference, do not hardcode predictions (they fuzz requests).
