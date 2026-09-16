"""Compute taxi-zone centroids from the official TLC shapefile.

Downloads taxi_zones.zip (public), reads polygons with pyshp (pure python),
computes area-weighted centroids, writes artifacts/zones.csv:

    LocationID,lat,lon

Run once; predict.py only needs the CSV.
"""
from __future__ import annotations

import csv
import urllib.request
import zipfile
from pathlib import Path

SHAPE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
HERE = Path(__file__).resolve().parent.parent


def polygon_centroid_area(pts):
    """Shoelace area + centroid of one ring. pts = [(x=lon, y=lat), ...]."""
    a2 = 0.0  # 2 * signed area
    cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a2) < 1e-12:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return sum(xs) / n, sum(ys) / n, 0.0
    return cx / (3 * a2), cy / (3 * a2), abs(a2) / 2


def main() -> None:
    import shapefile  # pyshp

    raw = HERE / "data_raw"
    raw.mkdir(exist_ok=True)
    zpath = raw / "taxi_zones.zip"
    if not zpath.exists():
        print("downloading shapefile ...")
        urllib.request.urlretrieve(SHAPE_URL, zpath)

    with zipfile.ZipFile(zpath) as z:
        shp_name = next(n for n in z.namelist() if n.endswith(".shp"))
        for suffix in (".shp", ".shx", ".dbf"):
            z.extract(shp_name.replace(".shp", suffix), raw)

    r = shapefile.Reader(str(raw / shp_name.replace(".shp", "")))
    # field order: first field is deletion flag; find LocationID index
    fields = [f[0] for f in r.fields[1:]]
    loc_idx = fields.index("LocationID")

    zones: dict[int, list] = {}  # loc -> [area_sum, lat*wsum, lon*wsum]
    for sr in r.iterShapeRecords():
        loc = int(sr.record[loc_idx])
        shape = sr.shape
        parts = list(shape.parts) + [len(shape.points)]
        for i in range(len(parts) - 1):
            ring = shape.points[parts[i]:parts[i + 1]]
            lon_c, lat_c, area = polygon_centroid_area(ring)
            if area <= 0:
                continue
            acc = zones.setdefault(loc, [0.0, 0.0, 0.0])
            acc[0] += area
            acc[1] += lat_c * area
            acc[2] += lon_c * area

    out = HERE / "artifacts" / "zones.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["LocationID", "lat", "lon"])
        for loc in sorted(zones):
            area, la, lo = zones[loc]
            w.writerow([loc, f"{la / area:.6f}", f"{lo / area:.6f}"])
    print(f"wrote {out} with {len(zones)} zones")


if __name__ == "__main__":
    main()
