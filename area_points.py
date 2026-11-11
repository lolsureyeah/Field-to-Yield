import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
import planetary_computer
import pystac_client

YEARS = [2018, 2019, 2020, 2021]
N_DRAW = 20000      # random points drawn per year
PER_CLASS = 1000000   # max points kept per class per year
rng = np.random.default_rng(7)

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
items = cat.search(
    collections=["usda-cdl"], bbox=[-96.6, 40.4, -90.1, 43.5]
).item_collection()
tiles = [i for i in items if i.id.startswith("cropland_")]

with rasterio.open(tiles[0].assets["cropland"].href) as s:
    crs = s.crs

counties = gpd.read_file("counties_iowa.geojson").to_crs(crs)
iowa = shapely.union_all(counties.geometry.values)
minx, miny, maxx, maxy = iowa.bounds


def random_points(n):
    xs, ys = [], []
    while len(xs) < n:
        x = rng.uniform(minx, maxx, n)
        y = rng.uniform(miny, maxy, n)
        keep = shapely.contains_xy(iowa, x, y)
        xs.extend(x[keep])
        ys.extend(y[keep])
    return np.array(xs[:n]), np.array(ys[:n])


out = []
for year in YEARS:
    x, y = random_points(N_DRAW)
    code = np.zeros(N_DRAW, dtype=np.uint8)
    for t in [t for t in tiles if t.id.startswith(f"cropland_{year}_")]:
        with rasterio.open(t.assets["cropland"].href) as src:
            b = src.bounds
            inside = (x >= b.left) & (x < b.right) & (y >= b.bottom) & (y < b.top)
            if inside.any():
                vals = [v[0] for v in src.sample(zip(x[inside], y[inside]))]
                code[inside] = vals
    df = pd.DataFrame({"year": year, "x": x, "y": y, "cdl": code})
    df = df[df["cdl"] > 0]
    df["label"] = np.where(df["cdl"] == 1, "corn", np.where(df["cdl"] == 5, "soy", "other"))
    df = df.sample(frac=1, random_state=42).groupby("label").head(PER_CLASS)
    out.append(df)
    print(year, df["label"].value_counts().to_dict(), flush=True)

pts = pd.concat(out, ignore_index=True)
g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.x, pts.y), crs=crs)
g = gpd.sjoin(g, counties[["GEOID", "geometry"]], how="left", predicate="within")
g = g.rename(columns={"GEOID": "fips"}).dropna(subset=["fips"])
ll = g.to_crs(4326)
g["lon"] = ll.geometry.x
g["lat"] = ll.geometry.y
g[["year", "lon", "lat", "cdl", "label", "fips"]].to_csv("area_points.csv", index=False)
print(len(g), "points saved")