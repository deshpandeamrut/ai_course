"""Lesson 12 - Neural networks: the Multi-Layer Perceptron (MLP).

An MLP is layers of small units. Each unit takes a weighted sum of its inputs
and passes it through a non-linear "activation". Stacking layers lets the model
learn curved, complex boundaries. Training adjusts the weights step by step to
reduce a loss (error) value.

Dataset: wine (13 chemical features, 3 cultivars).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)

X, y = load_wine(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

# Neural networks are VERY sensitive to feature scale, so always standardise.
# Fit on train only, then apply to test (no leakage).
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# --- A small network ---------------------------------------------------------
# hidden_layer_sizes=(16, 8): two hidden layers with 16 and 8 units.
# The input layer (13 features) and output layer (3 classes) are set for us.
# max_iter: max training passes (epochs). random_state: weights start random,
# so fix the seed for repeatable results.
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=0)
mlp.fit(X_train_s, y_train)

print(f"Train accuracy: {mlp.score(X_train_s, y_train):.3f}")
print(f"Test accuracy:  {mlp.score(X_test_s, y_test):.3f}")
print(f"Stopped after {mlp.n_iter_} iterations, final loss {mlp.loss_:.4f}\n")
print(classification_report(y_test, mlp.predict(X_test_s), target_names=["class_0", "class_1", "class_2"]))

# The learning curve: loss should fall steadily as the network learns.
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(mlp.loss_curve_)
ax.set_xlabel("iteration (epoch)")
ax.set_ylabel("training loss")
ax.set_title("MLP training loss on wine")
fig.tight_layout()
fig.savefig(OUT / "12_mlp_loss_curve.png", dpi=120)
print(f"Saved plot to {OUT / '12_mlp_loss_curve.png'}\n")

# --- Why scaling matters for MLP ---------------------------------------------
raw = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=0)
raw.fit(X_train, y_train)
print(f"Test accuracy WITHOUT scaling: {raw.score(X_test, y_test):.3f}")

# --- Network size: bigger is not always better --------------------------------
print("\nhidden layers | train acc | test acc")
for layers in [(2,), (8,), (16, 8), (100, 100)]:
    # The tiny (2,) network learns slowly and needs ~1500 iterations to converge.
    m = MLPClassifier(hidden_layer_sizes=layers, max_iter=3000, random_state=0)
    m.fit(X_train_s, y_train)
    print(f"{str(layers):13s} | {m.score(X_train_s, y_train):9.3f} | {m.score(X_test_s, y_test):8.3f}")

# Notes:
#  - Wine has only 178 rows. Neural nets shine with lots of data; on small
#    tabular data, simpler models (lesson 03/05) often do as well.
#  - Results vary a little with random_state because of random initial weights.

# --- Exercises ---------------------------------------------------------------
# 1. Change random_state to 1, 2, 3. How much does test accuracy move?
# 2. Try activation="tanh" instead of the default "relu".
# 3. Set max_iter=20. You'll see a ConvergenceWarning - what does it mean?
# 4. Try alpha=0.1 (L2 regularisation) on the (100, 100) network.
