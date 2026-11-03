"""Lesson 11 - Multiclass classification: Wine Recognition.

Lesson 10 had two classes. Many real problems have more than two. Here we
identify which of 3 grape cultivars a wine came from using 13 chemical
measurements (alcohol, proline, flavanoids, colour intensity, ...).

  - 178 samples, 13 numeric features, 3 classes

Good news: the scikit-learn code is almost identical to the binary case.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# --- 1. Load -----------------------------------------------------------------
wine = load_wine()
X, y = wine.data, wine.target
print("X shape:", X.shape, "| y shape:", y.shape)
print("Classes:", list(map(str, wine.target_names)))
print("Samples per class:", np.bincount(y), "\n")

# --- 2. Split ----------------------------------------------------------------
# stratify=y keeps the class proportions equal in train and test. It matters
# more here: with 3 classes and only 36 test rows, a random split could easily
# under-represent one class.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, " Test:", X_test.shape, "\n")

# --- 3. Scale ----------------------------------------------------------------
# Proline is in the hundreds/thousands while hue is around 1, so scale.
# Fit on train only!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    # Now a 3x3 matrix: diagonal = correct, off-diagonal = which classes get mixed up.
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=list(map(str, wine.target_names))))
    return pred


# --- 4. Five models ----------------------------------------------------------
# LogisticRegression, SVC and the others handle several classes automatically.
results = {}
results["Logistic Regression"] = accuracy_score(
    y_test, evaluate("Logistic Regression", LogisticRegression(max_iter=1000))
)
results["KNN (k=5)"] = accuracy_score(y_test, evaluate("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)))
results["Decision Tree (depth 5)"] = accuracy_score(
    y_test, evaluate("Decision Tree (depth 5)", DecisionTreeClassifier(max_depth=5, random_state=42))
)
results["SVM (linear)"] = accuracy_score(y_test, evaluate("SVM (linear)", SVC(kernel="linear", C=1.0)))
results["MLP (64,)"] = accuracy_score(
    y_test,
    evaluate(
        "MLP (64,)",
        MLPClassifier(hidden_layer_sizes=(64,), activation="relu", max_iter=500, random_state=42),
    ),
)

# --- 5. Compare --------------------------------------------------------------
print("===== Comparison (test accuracy) =====")
for name, acc in sorted(results.items(), key=lambda t: -t[1]):
    print(f"{name:24s} {acc:.4f}")

# Caution: the test set has only 36 samples, so ONE mistake changes accuracy by
# about 2.8%. Models within a few percent of each other are effectively tied.
# Lesson 08 (cross-validation) is the right tool for a trustworthy ranking.

# --- Exercises ---------------------------------------------------------------
# 1. In the confusion matrices, which two classes are mixed up most often?
# 2. Remove the scaling step. Which models suffer most? (Compare with lesson 04
#    for KNN and lesson 12 for the neural network.)
# 3. Try MLP hidden_layer_sizes (64, 32) and activation="tanh". Does it help?
# 4. Run the models with 5-fold cross-validation instead of a single split.
