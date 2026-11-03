"""Lesson 10 - Binary classification: Breast Cancer Wisconsin (Diagnostic).

Goal: decide whether a tumour is malignant or benign from 30 measurements of
cell nuclei taken from a fine needle aspirate image.

  - 569 samples, 30 numeric features
  - label 0 = malignant, label 1 = benign   (note: malignant is 0!)

We train four classifiers on the same split and compare them, paying special
attention to the one mistake that matters most in medicine: a missed cancer.

The workflow (the same recipe works for almost any sklearn model):

  Step 1 - Load     get X (measurements) and y (0/1 label)
  Step 2 - Split    hold back a test set the model never sees
  Step 3 - Scale    put all features on the same scale (learn it from TRAIN only)
  Step 4 - Fit      the model learns the pattern from X_train and y_train
  Step 5 - Predict  the model guesses labels for the unseen X_test
  Step 6 - Evaluate compare the guesses with the true y_test

Steps 4-6 are repeated for each of the four models.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# --- 1. Load (Step 1) --------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
print("X shape:", X.shape, "| y shape:", y.shape)
print("Classes:", dict(enumerate(map(str, data.target_names))))
print("First 5 features:", list(data.feature_names[:5]), "...\n")

# --- 2. Split (Step 2) -------------------------------------------------------
# 80% train / 20% test. random_state=42 makes the split reproducible.
# Split BEFORE scaling, so the test rows cannot influence the scaler.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Scale (Step 3) -------------------------------------------------------
# StandardScaler: rescale every feature to mean 0, std 1.
# fit_transform on TRAIN, plain transform on TEST. Fitting on test data would
# leak information about it into training ("data leakage").
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- 4-6. Fit, Predict, Evaluate (Steps 4, 5, 6) -----------------------------
def evaluate(name, model):
    """Fit, predict and print the three standard evaluation views."""
    # Step 4 - Fit: the model learns the pattern from the training data only.
    model.fit(X_train, y_train)
    # Step 5 - Predict: guess labels for test rows the model has never seen.
    pred = model.predict(X_test)
    # Step 6 - Evaluate: compare the guesses (pred) with the truth (y_test).
    print(f"===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=list(map(str, data.target_names))))
    return pred


# --- 7. Four models ----------------------------------------------------------
# Run Steps 4-6 once per model. Same fit / predict API for every one - only the
# constructor changes.
preds = {
    "Logistic Regression": evaluate("Logistic Regression", LogisticRegression(max_iter=1000)),
    "KNN (k=5)": evaluate("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree (depth 6)": evaluate(
        "Decision Tree (depth 6)", DecisionTreeClassifier(max_depth=6, random_state=42)
    ),
    "SVM (linear)": evaluate("SVM (linear)", SVC(kernel="linear", C=1.0)),
}

# --- 8. Side-by-side comparison ---------------------------------------------
# In the confusion matrix, row 0 = truly malignant. cm[0, 1] = malignant
# tumours that the model called benign = FALSE NEGATIVES (the dangerous error).
print("===== Comparison =====")
print(f"{'Model':26s} {'Accuracy':>9s} {'Malignant recall':>17s} {'Missed malignant':>17s}")
for name, p in preds.items():
    cm = confusion_matrix(y_test, p)
    print(
        f"{name:26s} {accuracy_score(y_test, p):9.4f} "
        f"{recall_score(y_test, p, pos_label=0):17.4f} {cm[0, 1]:17d}"
    )

# Why recall of the MALIGNANT class?
#   False negative = a cancer is called benign -> patient goes untreated.
#   False positive = a benign tumour is called malignant -> extra tests.
# The first is far worse, and recall (of the malignant class) measures exactly
# "of all real cancers, how many did we catch?".

# --- Exercises ---------------------------------------------------------------
# 1. Try KNN with n_neighbors = 3, 7, 11. Which k is best on this split?
# 2. Try SVC(kernel="rbf") and kernel="poly". Does the ranking change?
# 3. Change random_state in train_test_split to 0, 1, 2. Does the winner change?
#    With only 114 test samples, one extra mistake moves accuracy by ~0.9%.
# 4. Add stratify=y to the split. Why might that be a good idea here?
