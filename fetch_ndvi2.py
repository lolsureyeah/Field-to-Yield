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
F = 8          # shrink factor: 10 m pixels become 80 m
PER_SLOT = 2   # scenes kept per tile per half month
LABELS = ["jun1", "jun2", "jul1", "jul2", "aug1", "aug2", "sep1", "sep2"]
years = [int(a) for a in sys.argv[1:]] or [2018, 2019, 2020, 2021]

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
a = pd.read_csv("points.csv")
a["set"] = "train"
b = pd.read_csv("area_points.csv")
b["set"] = "area"
pts = pd.concat([a, b], ignore_index=True)
pts["idx"] = pts.index


def pick_items(year):
    items = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=f"{year}-06-01/{year}-09-30",
        query={"eo:cloud_cover": {"lt": 30}},
    ).item_collection()
    slots = {}
    for it in items:
        d = it.datetime
        half = d.month * 2 + (1 if d.day > 15 else 0)
        slots.setdefault((it.properties["s2:mgrs_tile"], half), []).append(it)
    chosen = []
    for (tile, half), lst in slots.items():
        lst.sort(key=lambda i: i.properties["eo:cloud_cover"])
        chosen.extend((half, i) for i in lst[:PER_SLOT])
    return chosen


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

        def band(name, method=Resampling.average):
            with rasterio.open(item.assets[name].href) as r:
                return r.read(1, out_shape=(oh, ow), resampling=method)

        nir = band("B08").astype("float32")
        swir = band("B11").astype("float32")
        scl = band("SCL", Resampling.nearest)
        red -= off
        nir -= off
        swir -= off
        t1 = nir + red
        t2 = nir + swir
        ndvi = np.where(t1 > 0, (nir - red) / np.where(t1 > 0, t1, 1), np.nan)
        ndmi = np.where(t2 > 0, (nir - swir) / np.where(t2 > 0, t2, 1), np.nan)
        valid = np.isin(scl, [4, 5]) & (t1 > 0) & (t2 > 0)
        ri = np.minimum(rows[inside] * oh // H, oh - 1)
        ci = np.minimum(cols[inside] * ow // W, ow - 1)
        ok = valid[ri, ci]
        idxs = py["idx"].values[inside][ok]
        return list(zip(idxs.tolist(), [half] * len(idxs),
                        ndvi[ri, ci][ok].tolist(), ndmi[ri, ci][ok].tolist()))
    except Exception as e:
        print("skipped one scene:", str(e)[:120], flush=True)
        return []


for year in years:
    outfile = f"feat_{year}.csv"
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
    obs = pd.DataFrame(rows, columns=["idx", "half", "ndvi", "ndmi"])
    g = obs.groupby(["idx", "half"])[["ndvi", "ndmi"]].median()
    parts = []
    for name in ["ndvi", "ndmi"]:
        w = g[name].unstack().reindex(columns=range(12, 20))
        w.columns = [f"{name}_{l}" for l in LABELS]
        parts.append(w)
    wide = pd.concat(parts, axis=1)
    out = py.join(wide, on="idx").drop(columns=["idx"])
    out.to_csv(outfile, index=False)
    filled = float(out[[f"ndvi_{l}" for l in LABELS]].notna().mean().mean())
    print(year, "saved", outfile, "| share of NDVI readings filled:", round(filled, 2), flush=True)