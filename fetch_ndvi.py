import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.warp import transform

BBOX = [-96.6, 40.4, -90.1, 43.5]
F = 8  # shrink factor: 10 m pixels become 80 m, much faster to download
LABELS = ["jun1", "jun2", "jul1", "jul2", "aug1", "aug2", "sep1", "sep2"]
years = [int(a) for a in sys.argv[1:]] or [2018, 2019, 2020, 2021]

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
pts = pd.read_csv("points.csv")
pts["idx"] = pts.index


def pick_items(year):
    items = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=f"{year}-06-01/{year}-09-30",
        query={"eo:cloud_cover": {"lt": 30}},
    ).item_collection()
    best = {}
    for it in items:
        d = it.datetime
        half = d.month * 2 + (1 if d.day > 15 else 0)
        key = (it.properties["s2:mgrs_tile"], half)
        cc = it.properties["eo:cloud_cover"]
        if key not in best or cc < best[key][0]:
            best[key] = (cc, it)
    return [(k[1], v[1]) for k, v in best.items()]


def process(item, half, py):
    try:
        item = planetary_computer.sign(item)
        base = float(item.properties.get("s2:processing_baseline", "0"))
        off = 1000.0 if base >= 4.0 else 0.0
        with rasterio.open(item.assets["B04"].href) as r4:
            H, W = r4.height, r4.width
            xs, ys = transform("EPSG:4326", r4.crs, py["lon"].tolist(), py["lat"].tolist())
            rows, cols = rowcol(r4.transform, xs, ys)
            rows, cols = np.array(rows), np.array(cols)
            inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
            if not inside.any():
                return []
            oh, ow = H // F, W // F
            red = r4.read(1, out_shape=(oh, ow), resampling=Resampling.average).astype("float32")
        with rasterio.open(item.assets["B08"].href) as r8:
            nir = r8.read(1, out_shape=(oh, ow), resampling=Resampling.average).astype("float32")
        with rasterio.open(item.assets["SCL"].href) as rs:
            scl = rs.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
        red -= off
        nir -= off
        total = nir + red
        ndvi = np.where(total > 0, (nir - red) / np.where(total > 0, total, 1), np.nan)
        valid = np.isin(scl, [4, 5]) & (total > 0)
        ri = np.minimum(rows[inside] * oh // H, oh - 1)
        ci = np.minimum(cols[inside] * ow // W, ow - 1)
        ok = valid[ri, ci]
        idxs = py["idx"].values[inside][ok]
        vals = ndvi[ri, ci][ok]
        return [(int(i), half, float(v)) for i, v in zip(idxs, vals)]
    except Exception as e:
        print("skipped one scene:", str(e)[:120], flush=True)
        return []


for year in years:
    outfile = f"ndvi_{year}.csv"
    if os.path.exists(outfile):
        print(year, "already done, skipping", flush=True)
        continue
    py = pts[pts["year"] == year]
    scenes = pick_items(year)
    print(year, len(scenes), "scenes selected", flush=True)
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process, item, half, py) for half, item in scenes]
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % 20 == 0:
                print(year, done, "of", len(scenes), "scenes read", flush=True)
    obs = pd.DataFrame(rows, columns=["idx", "half", "ndvi"])
    wide = obs.groupby(["idx", "half"])["ndvi"].median().unstack()
    wide = wide.reindex(columns=range(12, 20))
    wide.columns = [f"ndvi_{l}" for l in LABELS]
    out = py.join(wide, on="idx").drop(columns=["idx"])
    out.to_csv(outfile, index=False)
    filled = float(out[list(wide.columns)].notna().mean().mean())
    print(year, "saved", outfile, "| share of readings filled:", round(filled, 2), flush=True)