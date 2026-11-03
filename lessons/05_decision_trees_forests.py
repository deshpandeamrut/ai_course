"""Lesson 05 - Decision trees, overfitting, and random forests.

A tree asks a series of yes/no questions about features. Trees don't need
feature scaling, but an unconstrained tree memorises the training set.
Dataset: breast cancer.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# --- Overfitting in action ---------------------------------------------------
# Compare train vs test accuracy as the tree is allowed to grow deeper.
print("max_depth | train acc | test acc")
for depth in [1, 2, 3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    print(f"{str(depth):9s} | {tree.score(X_train, y_train):9.3f} | {tree.score(X_test, y_test):8.3f}")
# Train accuracy hits 1.0 with no depth limit, while test accuracy does not
# improve: that gap is overfitting.

# --- A tree is readable ------------------------------------------------------
small = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_train, y_train)
print("\nA depth-2 tree as rules:")
print(export_text(small, feature_names=list(X.columns)))

# --- Random forest: many decorrelated trees, vote ------------------------------
# Each tree sees a bootstrap sample of rows and a random subset of features at
# each split. Averaging many noisy trees cancels out their individual errors.
forest = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
print(f"Random forest: train={forest.score(X_train, y_train):.3f}  test={forest.score(X_test, y_test):.3f}")

print("\nTop 5 feature importances:")
imp = sorted(zip(X.columns, forest.feature_importances_), key=lambda t: -t[1])[:5]
for name, v in imp:
    print(f"  {name:25s} {v:.3f}")

# --- Exercises ---------------------------------------------------------------
# 1. Plot test accuracy vs n_estimators (1, 5, 20, 100, 500).
# 2. Try min_samples_leaf=5 on a single tree as another way to limit growth.
# 3. Impurity-based importances can be misleading for correlated features.
#    Look up sklearn.inspection.permutation_importance and compare.
