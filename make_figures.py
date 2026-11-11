import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

os.makedirs("figures", exist_ok=True)
CLASSES = ["corn", "soy", "other"]

# 1. Confusion matrix on random 2021 points
pts = pd.read_csv("area_pred.csv", dtype={"fips": str})
t = pts[pts["year"] == 2021]
cm = confusion_matrix(t["label"], t["pred"], labels=CLASSES)
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(CLASSES)
ax.set_yticks(range(3))
ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("Official (USDA CDL)")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center")
ax.set_title("Crop classifier, 2021 random points")
fig.tight_layout()
fig.savefig("figures/confusion_2021.png", dpi=150)
plt.close(fig)

# 2. County yield maps, official vs predicted
yp = pd.read_csv("yield_pred_2021_8.csv", dtype={"fips": str})
counties = gpd.read_file("counties_iowa.geojson")
for crop in ["corn", "soy"]:
    d = yp[yp["crop"] == crop]
    m = counties.merge(d, left_on="GEOID", right_on="fips", how="left")
    vmin = min(m["yield"].min(), m["pred_satellite"].min())
    vmax = max(m["yield"].max(), m["pred_satellite"].max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in zip(axes, ["yield", "pred_satellite"],
                              ["official USDA", "predicted from satellite"]):
        m.plot(column=col, ax=ax, cmap="YlGn", vmin=vmin, vmax=vmax, legend=True,
               edgecolor="white", linewidth=0.3, missing_kwds={"color": "lightgrey"})
        ax.set_title(f"{crop.capitalize()} yield 2021, {title} (bu/acre)", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"figures/yield_map_{crop}.png", dpi=150)
    plt.close(fig)

# 3. Predicted vs official yield per county
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, crop in zip(axes, ["corn", "soy"]):
    d = yp[yp["crop"] == crop]
    ax.scatter(d["yield"], d["pred_satellite"], s=14, alpha=0.7)
    lo = min(d["yield"].min(), d["pred_satellite"].min())
    hi = max(d["yield"].max(), d["pred_satellite"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Official yield (bu/acre)")
    ax.set_ylabel("Predicted yield (bu/acre)")
    ax.set_title(f"{crop.capitalize()} 2021, one dot per county")
fig.tight_layout()
fig.savefig("figures/yield_scatter_2021.png", dpi=150)
print("saved 4 figures in the figures folder")