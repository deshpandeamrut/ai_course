"""Lesson 02 - Regression: predicting a number.

Classification predicts a category; regression predicts a continuous value.
Dataset: diabetes progression score from 10 baseline health measurements.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# A baseline that always predicts the training mean. Any real model must beat it.
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)

model = LinearRegression().fit(X_train, y_train)


def report(name, m):
    pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, pred)             # average absolute miss
    rmse = np.sqrt(mean_squared_error(y_test, pred))    # punishes big misses more
    r2 = r2_score(y_test, pred)                         # 1 = perfect, 0 = same as mean
    print(f"{name:10s} MAE={mae:6.1f}  RMSE={rmse:6.1f}  R2={r2:5.2f}")


report("baseline", baseline)
report("linear", model)

# Linear regression learns one coefficient per feature plus an intercept:
#   y_hat = intercept + w1*x1 + w2*x2 + ...
print("\nIntercept:", round(model.intercept_, 1))
print("Coefficients (largest magnitude first):")
for name, w in sorted(zip(X.columns, model.coef_), key=lambda t: -abs(t[1])):
    print(f"  {name:4s} {w:8.1f}")

# --- Exercises ---------------------------------------------------------------
# 1. Why is R2 only about 0.4-0.5 here? Is the model bad, or is the data noisy?
# 2. Try sklearn.linear_model.Ridge(alpha=1.0). How do the coefficients change?
# 3. Predict on X_train instead of X_test. Is the score better or worse? Why?
