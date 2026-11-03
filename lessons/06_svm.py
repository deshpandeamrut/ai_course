"""Lesson 06 - Support vector machines and kernels.

An SVM finds the boundary with the widest margin between classes. A kernel lets
it draw curved boundaries. Like k-NN it is distance-based, so scale features.
Dataset: two interleaving half-moons (not separable by a straight line).
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    "linear": SVC(kernel="linear"),
    "rbf (default)": SVC(kernel="rbf"),
    "rbf, C=100, gamma=10 (overfit)": SVC(kernel="rbf", C=100, gamma=10),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, (name, svc) in zip(axes, models.items()):
    # make_pipeline chains scaler -> model so they are fitted together.
    clf = make_pipeline(StandardScaler(), svc).fit(X_train, y_train)
    print(f"{name:32s} train={clf.score(X_train, y_train):.3f}  test={clf.score(X_test, y_test):.3f}")
    ax.contourf(xx, yy, clf.predict(grid).reshape(xx.shape), alpha=0.25)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=12)
    ax.set_title(name, fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "06_svm_boundaries.png", dpi=120)
print(f"Saved plot to {OUT / '06_svm_boundaries.png'}")

# Key hyperparameters:
#   C     - penalty for misclassified training points. High C = fit tightly.
#   gamma - (rbf) reach of a single point. High gamma = wiggly boundary.

# --- Exercises ---------------------------------------------------------------
# 1. Add kernel="poly" with degree=3. How does the boundary look?
# 2. Grid over C in [0.1, 1, 10, 100] with gamma fixed. Watch train vs test gap.
# 3. SVC gets slow on large datasets. Why? (Hint: it scales ~O(n^2) or worse.)
