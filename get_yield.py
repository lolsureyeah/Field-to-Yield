import requests
import pandas as pd

key = open("key.txt").read().strip()
crops = {
    "corn": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
    "soy": "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
}

rows = []
for crop, desc in crops.items():
    r = requests.get(
        "https://quickstats.nass.usda.gov/api/api_GET/",
        params={
            "key": key,
            "source_desc": "SURVEY",
            "short_desc": desc,
            "agg_level_desc": "COUNTY",
            "state_alpha": "IA",
            "year__GE": 2018,
            "year__LE": 2023,
            "format": "JSON",
        },
        timeout=60,
    )
    r.raise_for_status()
    for d in r.json()["data"]:
        if d["county_ansi"]:
            rows.append({
                "crop": crop,
                "year": int(d["year"]),
                "fips": d["state_ansi"] + d["county_ansi"],
                "county": d["county_name"],
                "yield": d["Value"],
            })

df = pd.DataFrame(rows)
df["yield"] = pd.to_numeric(df["yield"].str.replace(",", ""), errors="coerce")
df = df.dropna()
df = df[(df["year"] >= 2018) & (df["year"] <= 2021)]
df.to_csv("yield_iowa.csv", index=False)
print(df.groupby(["crop", "year"]).size())
print(df.head())