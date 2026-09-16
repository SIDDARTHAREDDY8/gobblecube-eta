"""Download NYC TLC yellow-taxi trip parquet (public data).

Usage:
    python scripts/download_data.py --months 2023-06 [--months 2023-07 ...] --out data_raw

Full-scale training (11.5 months of 2023) needs ~2-3 GB of parquet + RAM for
groupbys; run it on a bigger machine (see SUBMISSION-PLAN.md). Locally we
validate the pipeline on a single month.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True,
                    help="e.g. 2023-06 2023-07")
    ap.add_argument("--out", default="data_raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import urllib.request

    for m in args.months:
        url = f"{BASE}/yellow_tripdata_{m}.parquet"
        dest = out / f"yellow_tripdata_{m}.parquet"
        if dest.exists():
            print(f"exists, skipping: {dest}")
            continue
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
