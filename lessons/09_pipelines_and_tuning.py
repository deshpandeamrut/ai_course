"""Lesson 09 - Pipelines and hyperparameter tuning.

Hyperparameters (k in k-NN, C in SVM, max_depth in trees) are set BEFORE
training. GridSearchCV tries combinations using cross-validation on the
training data, so the test set stays untouched until the very end.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)

# 1. Lock the test set away FIRST.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 2. Pipeline = preprocessing + model as one estimator. Steps are named.
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svc", SVC()),
])

# 3. Parameter grid: "<step name>__<parameter>".
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    "svc__kernel": ["rbf", "linear"],
}

# 4. Search with 5-fold CV on the training set only.
search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
search.fit(X_train, y_train)

print(f"Combinations tried: {len(search.cv_results_['params'])} (x5 folds each)")
print("Best params:", search.best_params_)
print(f"Best CV F1: {search.best_score_:.3f}\n")

# 5. best_estimator_ is already refit on the full training set. Evaluate ONCE.
print(f"Held-out test accuracy: {search.best_estimator_.score(X_test, y_test):.3f}")

# Top 3 configurations, to see how flat or peaked the search landscape is.
import pandas as pd

res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(3)
print("\nTop 3 configurations:")
print(res[["params", "mean_test_score", "std_test_score"]].to_string(index=False))

# --- Exercises ---------------------------------------------------------------
# 1. Why must the scaler be inside the pipeline when using GridSearchCV?
# 2. Replace SVC with RandomForestClassifier and tune n_estimators / max_depth.
#    (Drop the scaler or keep it - does it matter for trees?)
# 3. Look up RandomizedSearchCV. When is it better than a full grid?
# 4. If you tune, look at the test score, then tune again - what went wrong?
