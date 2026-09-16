FROM python:3.12-slim

WORKDIR /app

# Inference needs only numpy + scikit-learn (no pandas/geopandas/xgboost):
# image stays far under the 2.5 GB budget.
COPY requirements-infer.txt .
RUN pip install --no-cache-dir -r requirements-infer.txt

# Submission files. grade.py is the challenge's exact scoring harness
# (unchanged from the starter); the grader runs:
#   docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
COPY predict.py grade.py ./
COPY artifacts/ ./artifacts/

# Build-time smoke test: the module must import and answer in budget.
RUN python -c "import predict, time; \
    r={'pickup_zone':132,'dropoff_zone':236,'requested_at':'2024-01-15T08:30:00','passenger_count':1}; \
    t=time.perf_counter(); v=predict.predict(r); dt=(time.perf_counter()-t)*1000; \
    assert isinstance(v, float) and 30 <= v <= 14400, v; \
    assert dt < 200, dt; \
    print(f'smoke ok: {v:.1f}s in {dt:.1f} ms')"

ENTRYPOINT ["python", "grade.py"]
