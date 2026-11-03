"""Lesson 13 - Gradient boosting.

Random forest (lesson 05) builds many trees INDEPENDENTLY and averages them.
Gradient boosting builds trees ONE AFTER ANOTHER, and each new small tree
tries to fix the mistakes the previous ones made. It is often the most accurate
model on tabular data.

Dataset: breast cancer (binary).
"""

import time

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# HistGradientBoostingClassifier is scikit-learn's fast implementation. It
# needs no feature scaling and handles missing values natively.
#   learning_rate - how much each new tree is trusted (small = slower, safer)
#   max_iter      - number of boosting rounds (= number of trees)
#   max_depth     - depth of each tree (shallow trees = "weak learners")
gb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_depth=3, random_state=0)
gb.fit(X_train, y_train)

print(f"Train accuracy: {gb.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {gb.score(X_test, y_test):.3f}\n")
print(classification_report(y_test, gb.predict(X_test), target_names=list(map(str, data.target_names))))

# --- Compare with the models from earlier lessons (same folds) ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
models = {
    "single tree": DecisionTreeClassifier(random_state=0),
    "random forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0),
}
print("model              | CV accuracy      | fit time (s)")
for name, m in models.items():
    t0 = time.perf_counter()
    scores = cross_val_score(m, X, y, cv=cv)
    secs = time.perf_counter() - t0
    print(f"{name:18s} | {scores.mean():.3f} +/- {scores.std():.3f} | {secs:6.2f}")

# --- Watching the ensemble grow ----------------------------------------------
# staged_predict gives predictions after 1, 2, 3, ... trees.
print("\ntrees | test accuracy")
stages = list(gb.staged_predict(X_test))
for n in [1, 5, 10, 25, 50, 100]:
    acc = (stages[n - 1] == y_test.to_numpy()).mean()
    print(f"{n:5d} | {acc:.3f}")

# --- Early stopping: let the model decide when to stop -----------------------
# Hold out 10% of the TRAINING data internally; stop when its score stops
# improving for 10 rounds. Prevents boosting from overfitting with too many trees.
es = HistGradientBoostingClassifier(
    max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=0
).fit(X_train, y_train)
print(f"\nEarly stopping used {es.n_iter_} of 1000 allowed trees; test accuracy {es.score(X_test, y_test):.3f}")

# --- Exercises ---------------------------------------------------------------
# 1. Try learning_rate=1.0 and 0.01 (with max_iter=100). What happens? Why?
# 2. Gradient boosting has many hyperparameters. Use GridSearchCV (lesson 09)
#    over learning_rate and max_depth.
# 3. When would you prefer random forest? (Hint: less tuning, harder to overfit.)
