"""Lesson 04 - k-NN: choosing k, and why scaling matters.

k-NN classifies by distance to training points, so any feature with a large
numeric range dominates the distance calculation unless you scale.
Dataset: wine (13 chemical features, 3 cultivars) - ranges differ wildly.
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

X, y = load_wine(return_X_y=True, as_frame=True)
print("Feature ranges (max - min), note 'proline' dwarfs the others:")
print((X.max() - X.min()).round(1).sort_values(ascending=False).head(4), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)
scaler = StandardScaler().fit(X_train)  # fit on train only
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

ks = range(1, 31)
raw, scaled = [], []
for k in ks:
    raw.append(KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test))
    scaled.append(KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train).score(X_test_s, y_test))

print(f"Best raw    accuracy: {max(raw):.3f} (k={ks[raw.index(max(raw))]})")
print(f"Best scaled accuracy: {max(scaled):.3f} (k={ks[scaled.index(max(scaled))]})")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, raw, label="unscaled")
ax.plot(ks, scaled, label="scaled")
ax.set_xlabel("k (number of neighbours)")
ax.set_ylabel("test accuracy")
ax.set_title("k-NN on wine: scaling helps a lot")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "04_knn_scaling.png", dpi=120)
print(f"Saved plot to {OUT / '04_knn_scaling.png'}")

# Small k = flexible, jagged boundary (high variance, may overfit).
# Large k = smooth boundary (high bias, may underfit).

# --- Exercises ---------------------------------------------------------------
# 1. Pick k using the test set, as done above? That's subtly cheating -
#    the test set stops being unseen. Lesson 08/09 shows the proper way.
# 2. Try weights="distance" so closer neighbours count more.
# 3. Try MinMaxScaler instead of StandardScaler.
