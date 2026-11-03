"""Lesson 07 - Beyond accuracy: confusion matrix, precision, recall, F1.

Accuracy lies on imbalanced data. If 95% of transactions are legitimate, a model
that always says "legitimate" scores 95% and catches zero fraud.
"""

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Synthetic data: ~5% positives (the rare class we care about, e.g. fraud).
X, y = make_classification(
    n_samples=5000, n_features=10, n_informative=5, weights=[0.95, 0.05], random_state=0
)
print(f"Positive rate: {y.mean():.3f}\n")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

# Baseline: always predict the majority class.
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"Dummy accuracy: {dummy.score(X_test, y_test):.3f}  <- looks great, is useless")
print(f"Dummy recall:   {recall_score(y_test, dummy.predict(X_test)):.3f}\n")

model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
pred = model.predict(X_test)

# Confusion matrix layout (rows = truth, cols = prediction):
#            pred 0   pred 1
#   true 0     TN       FP
#   true 1     FN       TP
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"TN={tn} FP={fp} FN={fn} TP={tp}\n")

# precision = TP / (TP + FP)  "of those I flagged, how many were real?"
# recall    = TP / (TP + FN)  "of the real ones, how many did I catch?"
# F1        = harmonic mean of the two
print(f"precision={precision_score(y_test, pred):.3f}  "
      f"recall={recall_score(y_test, pred):.3f}  f1={f1_score(y_test, pred):.3f}\n")
print(classification_report(y_test, pred, target_names=["negative", "positive"]))

# Moving the decision threshold trades precision against recall.
proba = model.predict_proba(X_test)[:, 1]
print("threshold | precision | recall")
for t in [0.2, 0.3, 0.5, 0.7]:
    p = (proba >= t).astype(int)
    print(f"{t:9.1f} | {precision_score(y_test, p, zero_division=0):9.3f} | {recall_score(y_test, p):6.3f}")

# --- Exercises ---------------------------------------------------------------
# 1. Which matters more for cancer screening, precision or recall? For spam?
# 2. Try RandomForestClassifier(class_weight="balanced"). Effect on recall?
# 3. Compute roc_auc_score(y_test, proba). Why is AUC threshold-independent?
