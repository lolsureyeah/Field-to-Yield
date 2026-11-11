import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LABELS = ["jun1", "jun2", "jul1", "jul2", "aug1", "aug2", "sep1", "sep2"]
import sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # half months used; 5 = up to early August
SIGNALS = [f"ndvi_{l}" for l in LABELS[:N]] + [f"ndmi_{l}" for l in LABELS[:N]]

pts = pd.read_csv("area_pred.csv", dtype={"fips": str})
yl = pd.read_csv("yield_iowa.csv", dtype={"fips": str})

rows = []
for (year, fips), g in pts.groupby(["year", "fips"]):
    if len(g) < 30:
        continue
    rec = {
        "year": year, "fips": fips,
        "corn_share": (g["pred"] == "corn").mean(),
        "soy_share": (g["pred"] == "soy").mean(),
        "corn_share_cdl": (g["label"] == "corn").mean(),
        "soy_share_cdl": (g["label"] == "soy").mean(),
    }
    for crop in ["corn", "soy"]:
        s = g[g["pred"] == crop]
        for c in SIGNALS:
            rec[f"{crop}_{c}"] = s[c].mean() if len(s) >= 10 else np.nan
    rows.append(rec)
cty = pd.DataFrame(rows)
print("county year rows:", len(cty))

print("\nStage 1 check: predicted vs official crop share per county")
for crop in ["corn", "soy"]:
    for yr in [2018, 2021]:
        d = cty[cty["year"] == yr]
        r = np.corrcoef(d[f"{crop}_share"], d[f"{crop}_share_cdl"])[0, 1]
        print(f"  {crop} {yr}: correlation {r:.3f}")


def hgb():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3, random_state=42
    )


def score(name, truth, pred):
    rmse = mean_squared_error(truth, pred) ** 0.5
    print(f"  {name:<34} RMSE {rmse:6.2f}  MAE {mean_absolute_error(truth, pred):6.2f}  R2 {r2_score(truth, pred):6.3f}")


out = []
for crop in ["corn", "soy"]:
    y_df = yl[yl["crop"] == crop][["year", "fips", "yield"]]
    d = cty.merge(y_df, on=["year", "fips"])
    feats = [f"{crop}_share"] + [f"{crop}_{c}" for c in SIGNALS]
    train = d[d["year"] <= 2020].copy()
    test = d[d["year"] == 2021].copy()

    g = train.groupby("fips")["yield"]
    s, n = g.transform("sum"), g.transform("count")
    train["prior"] = ((s - train["yield"]) / (n - 1)).where(n > 1)
    test["prior"] = test["fips"].map(train.groupby("fips")["yield"].mean())
    test = test.dropna(subset=["prior"])

    print(f"\n{crop.upper()} yield, bushels per acre | train rows {len(train)}, test rows (2021) {len(test)}")
    print(f"  test yield mean {test['yield'].mean():.1f}, spread (std) {test['yield'].std():.1f}")
    score("Baseline: county average 2018-2020", test["yield"], test["prior"])
    m1 = hgb().fit(train[feats], train["yield"])
    p1 = m1.predict(test[feats])
    score("Satellite features only", test["yield"], p1)
    m2 = hgb().fit(train[feats + ["prior"]], train["yield"])
    p2 = m2.predict(test[feats + ["prior"]])
    score("Satellite features + county average", test["yield"], p2)
    res = test[["year", "fips", "yield"]].copy()
    res["crop"] = crop
    res["pred_satellite"] = p1
    res["pred_with_prior"] = p2
    out.append(res)

pd.concat(out).to_csv(f"yield_pred_2021_{N}.csv", index=False)
print("\nsaved yield_pred_2021.csv")