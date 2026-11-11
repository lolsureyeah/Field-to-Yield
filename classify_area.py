import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

LABELS = ["jun1", "jun2", "jul1", "jul2", "aug1", "aug2", "sep1", "sep2"]
NDVI = [f"ndvi_{l}" for l in LABELS]
NDMI = [f"ndmi_{l}" for l in LABELS]
CLASSES = ["corn", "soy", "other"]

df = pd.concat(
    [pd.read_csv(f"feat_{y}.csv", dtype={"fips": str}) for y in [2018, 2019, 2020, 2021]],
    ignore_index=True,
)
df = df[df["set"] == "area"].reset_index(drop=True)


def fill(v):
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return np.full(len(v), np.nan)
    return np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])


A = np.array([fill(r) for r in df[NDVI].values.astype(float)])
B = np.array([fill(r) for r in df[NDMI].values.astype(float)])
good = ~np.isnan(A).any(axis=1) & ~np.isnan(B).any(axis=1)
print("points kept:", int(good.sum()), "of", len(df))
df, A, B = df[good].reset_index(drop=True), A[good], B[good]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


net = Net()
net.load_state_dict(torch.load("cnn2.pt"))
net.eval()
S = torch.tensor(np.stack([A, B], axis=1), dtype=torch.float32)
with torch.no_grad():
    pred = net(S).argmax(1).numpy()
df["pred"] = [CLASSES[i] for i in pred]
for i, l in enumerate(LABELS):
    df[f"ndvi_{l}"] = A[:, i]
    df[f"ndmi_{l}"] = B[:, i]

t = df[df["year"] == 2021]
print("\nAccuracy on random 2021 points:", round(accuracy_score(t["label"], t["pred"]), 3))
print(classification_report(t["label"], t["pred"], digits=3))
df[["year", "fips", "label", "pred"] + NDVI + NDMI].to_csv("area_pred.csv", index=False)
print("saved area_pred.csv")