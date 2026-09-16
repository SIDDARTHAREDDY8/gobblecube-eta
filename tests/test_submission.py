"""Smoke tests for the Gobblecube ETA submission.

Run:  python -m pytest tests/ -q   (or plain: python tests/test_submission.py)

Covers the challenge constraints:
  - predict(request: dict) -> float exists and behaves
  - single-request latency < 200 ms on CPU
  - no network access at inference
  - prediction is deterministic and within sane bounds
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import predict  # noqa: E402


def sample_request(**kw):
    r = {"pickup_zone": 132, "dropoff_zone": 236,
         "requested_at": "2024-01-15T08:30:00", "passenger_count": 1}
    r.update(kw)
    return r


def test_returns_float_in_bounds():
    v = predict.predict(sample_request())
    assert isinstance(v, float), type(v)
    assert 30.0 <= v <= 14400.0, v


def test_positive_duration():
    assert predict.predict(sample_request()) > 0


def test_accepts_edge_zones():
    for zone in (1, 132, 265):
        v = predict.predict(sample_request(pickup_zone=zone, dropoff_zone=zone))
        assert isinstance(v, float)


def test_varies_with_time_of_day():
    # mirrors the starter's contract test: rush hour must differ from night
    morning = predict.predict(sample_request(pickup_zone=100, dropoff_zone=200,
                                             requested_at="2024-02-14T08:00:00"))
    night = predict.predict(sample_request(pickup_zone=100, dropoff_zone=200,
                                           requested_at="2024-02-14T23:00:00"))
    assert morning != night, "predictions should vary by time of day"


def test_latency_under_200ms():
    lat = []
    for i in range(200):
        r = sample_request(pickup_zone=1 + (i * 37) % 265,
                           dropoff_zone=1 + (i * 91) % 265,
                           requested_at=f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:15:00")
        t = time.perf_counter()
        predict.predict(r)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    p50, p99, mx = lat[100], lat[198], lat[-1]
    print(f"\nlatency ms: p50={p50:.1f} p99={p99:.1f} max={mx:.1f}")
    assert mx < 200, f"max latency {mx:.1f} ms exceeds 200 ms budget"


def test_no_network_at_inference():
    real_socket = socket.socket

    def blocked(*a, **k):
        raise AssertionError("network call during predict()")

    socket.socket = blocked
    try:
        v = predict.predict(sample_request())
    finally:
        socket.socket = real_socket
    assert isinstance(v, float)


def test_deterministic():
    r = sample_request()
    assert predict.predict(r) == predict.predict(r)


def test_timezone_aware_input():
    a = predict.predict(sample_request(requested_at="2024-01-15T13:30:00+00:00"))
    b = predict.predict(sample_request(requested_at="2024-01-15T08:30:00-05:00"))
    assert a == b, "same instant in different tz representations must match"


def test_validation_numbers_sane():
    p = HERE / "artifacts" / "validation_results.json"
    if not p.exists():
        print("\nvalidation_results.json not present; skipping")
        return
    res = json.loads(p.read_text())
    for k in ("global_mean_mae", "zone_pair_mae", "hierarchical_mae"):
        assert k in res, f"missing {k}"
    # hierarchy must beat the global mean on real data; blend must be sane
    assert res["hierarchical_mae"] < res["global_mean_mae"]
    assert res.get("blended_mae", 1e9) < 450


if __name__ == "__main__":
    for name, fn in sorted([(k, v) for k, v in globals().items()
                            if k.startswith("test_")]):
        fn()
        print(f"PASS {name}")
    print("all smoke tests passed")
