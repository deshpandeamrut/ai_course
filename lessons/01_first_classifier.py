"""Lesson 01 - Your first supervised learning model.

Supervised learning = learn a mapping from inputs (X) to a known answer (y)
using labelled examples, then predict y for new, unseen inputs.

Here: classify iris flowers into 3 species from 4 measurements.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening a window
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

# --- 1. Load data -----------------------------------------------------------
# as_frame=True gives a pandas DataFrame, which is easier to inspect.
iris = load_iris(as_frame=True)
X = iris.data      # features: sepal/petal length and width (150 rows x 4 cols)
y = iris.target    # labels: 0, 1, 2 -> setosa, versicolor, virginica

print("Feature table (first 5 rows):")
print(X.head(), "\n")
print("Class names:", [str(n) for n in iris.target_names])
print("Rows per class:\n", y.value_counts().sort_index(), "\n")

# --- 2. Look at the data (always do this before modelling) -------------------
fig, ax = plt.subplots(figsize=(6, 4))
for label, name in enumerate(iris.target_names):
    mask = y == label
    ax.scatter(X.loc[mask, "petal length (cm)"], X.loc[mask, "petal width (cm)"], label=name)
ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.set_title("Iris: the classes are visibly separable")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "01_iris_scatter.png", dpi=120)
print(f"Saved plot to {OUT / '01_iris_scatter.png'}\n")

# --- 3. Split into train and test -------------------------------------------
# The model must be judged on data it has NEVER seen, otherwise we only measure
# memorisation. test_size=0.25 holds back 25% for the final exam.
# stratify=y keeps the class proportions the same in both sets.
# random_state fixes the shuffle so results are reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"Train size: {len(X_train)}, test size: {len(X_test)}\n")

# --- 4. Choose a model and fit ----------------------------------------------
# k-nearest neighbours: to classify a flower, look at the 5 most similar
# training flowers and take a majority vote.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)   # "learning" happens here

# --- 5. Predict --------------------------------------------------------------
y_pred = model.predict(X_test)
print("First 10 predictions:", y_pred[:10])
print("First 10 true labels:", y_test.to_numpy()[:10], "\n")

# --- 6. Evaluate -------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)  # fraction predicted correctly
print(f"Test accuracy: {acc:.3f}")
# model.score(X_test, y_test) is a shortcut for the same number.

# --- 7. Predict a brand-new flower ------------------------------------------
# Passing a DataFrame with the same column names avoids a feature-name warning.
import pandas as pd

new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
pred = model.predict(new_flower)[0]
print(f"New flower {new_flower.iloc[0].tolist()} -> {iris.target_names[pred]}")

# --- Exercises ---------------------------------------------------------------
# 1. Change n_neighbors to 1, then 50. What happens to accuracy? Why?
# 2. Change random_state to 0, 1, 2... Does accuracy change? What does that
#    tell you about trusting a single split? (Lesson 08 addresses this.)
# 3. Swap KNeighborsClassifier for sklearn.tree.DecisionTreeClassifier. The
#    fit/predict API is identical - that consistency is scikit-learn's big idea.
