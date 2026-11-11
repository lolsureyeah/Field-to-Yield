import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ["jun1", "jun2", "jul1", "jul2", "aug1", "aug2", "sep1", "sep2"]
COLS = [f"ndvi_{l}" for l in LABELS]
CLASSES = ["corn", "soy", "other"]

df = pd.concat(
    [pd.read_csv(f"ndvi_{y}.csv") for y in [2018, 2019, 2020, 2021]],
    ignore_index=True,
)


def fill(v):
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return np.full(len(v), np.nan)
    return np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])


X = np.array([fill(r) for r in df[COLS].values.astype(float)])
good = ~np.isnan(X).any(axis=1)
print("points kept:", int(good.sum()), "of", len(df))
df, X = df[good].reset_index(drop=True), X[good]

y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).values
train = (df["year"] <= 2020).values
test = (df["year"] == 2021).values
print("train points:", int(train.sum()), "| test points (2021):", int(test.sum()))

# Model 1: random forest
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X[train], y[train])
rf_pred = rf.predict(X[test])
print("\nRANDOM FOREST accuracy on 2021:", round(accuracy_score(y[test], rf_pred), 3))
print(classification_report(y[test], rf_pred, target_names=CLASSES, digits=3))
print("confusion matrix (rows = truth, columns = prediction; order corn, soy, other)")
print(confusion_matrix(y[test], rf_pred))

# Model 2: 1D CNN
torch.manual_seed(42)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


Xt = torch.tensor(X[train], dtype=torch.float32).unsqueeze(1)
yt = torch.tensor(y[train])
Xe = torch.tensor(X[test], dtype=torch.float32).unsqueeze(1)
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
print("\n1D CNN accuracy on 2021:", round(accuracy_score(y[test], cnn_pred), 3))
print(classification_report(y[test], cnn_pred, target_names=CLASSES, digits=3))
print(confusion_matrix(y[test], cnn_pred))

# Save models and the 2021 predictions
joblib.dump(rf, "rf.joblib")
torch.save(net.state_dict(), "cnn.pt")
out = df[test].copy()
out["rf_pred"] = [CLASSES[i] for i in rf_pred]
out["cnn_pred"] = [CLASSES[i] for i in cnn_pred]
out.to_csv("preds_2021.csv", index=False)
print("\nsaved rf.joblib, cnn.pt, preds_2021.csv")