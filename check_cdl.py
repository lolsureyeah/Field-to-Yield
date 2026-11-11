import planetary_computer
import pystac_client

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
items = cat.search(
    collections=["usda-cdl"],
    bbox=[-96.6, 40.4, -90.1, 43.5],
).item_collection()
years = sorted({i.id.split("_")[1] for i in items if i.id.startswith("cropland_")})
print("Years available:", years)