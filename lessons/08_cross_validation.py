"""Lesson 08 - Cross-validation.

A single train/test split gives a noisy estimate that depends on which rows
happened to land in the test set. k-fold CV splits the data into k parts, trains
on k-1 and tests on the held-out one, rotating k times.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# --- How noisy is a single split? --------------------------------------------
single = []
for seed in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    single.append(model.fit(X_tr, y_tr).score(X_te, y_te))
print(f"20 different single splits: min={min(single):.3f}  max={max(single):.3f}  "
      f"std={np.std(single):.3f}\n")

# --- 5-fold stratified CV ----------------------------------------------------
# Stratified keeps class proportions in each fold. shuffle=True mixes row order.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("Per-fold accuracy:", scores.round(3))
print(f"Mean = {scores.mean():.3f}   Std = {scores.std():.3f}\n")

# --- Compare models fairly: same folds, same metric --------------------------
candidates = {
    "logistic": model,
    "random forest": RandomForestClassifier(n_estimators=200, random_state=0),
}
for name, m in candidates.items():
    s = cross_val_score(m, X, y, cv=cv, scoring="f1")
    print(f"{name:14s} F1 = {s.mean():.3f} +/- {s.std():.3f}")

# Because the scaler is INSIDE the pipeline, it is re-fitted on each training
# fold only - no leakage from the held-out fold.

# --- Exercises ---------------------------------------------------------------
# 1. Try cv=10. Does the mean change? Does the std?
# 2. Use scoring="recall" and "roc_auc". Is the model ranking the same?
# 3. Scale X up front with StandardScaler().fit_transform(X), then CV without
#    a pipeline. The score is barely different here, but it's still leakage.
