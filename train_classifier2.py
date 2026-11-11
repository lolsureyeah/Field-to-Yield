import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ["jun1", "jun2", "jul1", "jul2", "aug1", "aug2", "sep1", "sep2"]
NDVI = [f"ndvi_{l}" for l in LABELS]
NDMI = [f"ndmi_{l}" for l in LABELS]
CLASSES = ["corn", "soy", "other"]

df = pd.concat(
    [pd.read_csv(f"feat_{y}.csv") for y in [2018, 2019, 2020, 2021]],
    ignore_index=True,
)
df = df[df["set"] == "train"].reset_index(drop=True)


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

y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).values
train = (df["year"] <= 2020).values
test = (df["year"] == 2021).values
print("train points:", int(train.sum()), "| test points (2021):", int(test.sum()))


def report(name, truth, pred):
    print(f"\n{name}: accuracy {accuracy_score(truth, pred):.3f}")
    print(classification_report(truth, pred, target_names=CLASSES, digits=3))
    print(confusion_matrix(truth, pred))


# Model 1: random forest on NDVI only (same idea as before)
rf1 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf1.fit(A[train], y[train])
report("RANDOM FOREST, NDVI only", y[test], rf1.predict(A[test]))

# Model 2: random forest on NDVI and NDMI
X = np.concatenate([A, B], axis=1)
rf2 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf2.fit(X[train], y[train])
report("RANDOM FOREST, NDVI and NDMI", y[test], rf2.predict(X[test]))

# Model 3: 1D CNN on both signals as two channels
torch.manual_seed(42)


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


S = np.stack([A, B], axis=1)
Xt = torch.tensor(S[train], dtype=torch.float32)
yt = torch.tensor(y[train])
Xe = torch.tensor(S[test], dtype=torch.float32)
net = Net()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(40):
    net.train()
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        opt.zero_grad()
        loss_fn(net(Xt[b]), yt[b]).backward()
        opt.step()
net.eval()
with torch.no_grad():
    cnn_pred = net(Xe).argmax(1).numpy()
report("1D CNN, NDVI and NDMI", y[test], cnn_pred)

joblib.dump(rf2, "rf2.joblib")
torch.save(net.state_dict(), "cnn2.pt")
print("\nsaved rf2.joblib and cnn2.pt")