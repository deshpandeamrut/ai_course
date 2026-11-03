"""Lesson 03 - Logistic regression: classification with probabilities.

Despite the name, this is a classifier. It outputs the probability of each
class; predict() picks the most probable one.
Dataset: breast cancer (binary: malignant vs benign), 30 numeric features.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print("Classes:", dict(enumerate(map(str, data.target_names))), "| counts:", y.value_counts().to_dict(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Features live on very different scales (area ~ 1000, smoothness ~ 0.1).
# Logistic regression is optimised iteratively and behaves better when features
# are standardised to mean 0, std 1.
# IMPORTANT: fit the scaler on TRAIN only, then apply it to test. Fitting on
# test data would leak information about it into training.
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
print(f"Test accuracy: {model.score(X_test_s, y_test):.3f}\n")

# predict_proba returns one column per class, rows sum to 1.
proba = model.predict_proba(X_test_s[:5])
print("Probabilities for first 5 test samples (columns = malignant, benign):")
print(proba.round(3))
print("True labels:", y_test.to_numpy()[:5], "\n")

# Coefficients show which features push toward class 1 (benign) or class 0.
coefs = sorted(zip(X.columns, model.coef_[0]), key=lambda t: -abs(t[1]))[:5]
print("Top 5 features by |coefficient| (standardised, so comparable):")
for name, w in coefs:
    print(f"  {name:25s} {w:6.2f}")

# --- Exercises ---------------------------------------------------------------
# 1. Remove the scaling and fit again. Do you get a convergence warning?
# 2. LogisticRegression(C=0.01) regularises strongly; C=100 weakly. Compare.
# 3. Instead of predict(), threshold the probabilities yourself at 0.9 for
#    "benign". What happens to the errors? (Foreshadows lesson 07.)
