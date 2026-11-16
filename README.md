# Supervised Learning with scikit-learn

A beginner-level Python course. Each lesson is a runnable script with comments
that explain every step. Read the script, run it, then do the exercises at the bottom.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a lesson

```bash
python lessons/01_first_classifier.py
```

Plots are saved to `outputs/` (no window pops up).

## Lessons

| # | File | Topic |
|---|------|-------|
| 01 | `01_first_classifier.py` | The full workflow on iris: load, split, fit, predict, evaluate |
| 02 | `02_linear_regression.py` | Regression basics, MAE / RMSE / R² |
| 03 | `03_logistic_regression.py` | Classification with probabilities, scaling |
| 04 | `04_knn_and_scaling.py` | k-NN, choosing k, why scaling matters |
| 05 | `05_decision_trees_forests.py` | Trees, overfitting, random forests, feature importance |
| 06 | `06_svm.py` | Support vector machines and kernels |
| 07 | `07_metrics.py` | Confusion matrix, precision / recall / F1, imbalanced data |
| 08 | `08_cross_validation.py` | Why one split isn't enough; k-fold CV |
| 09 | `09_pipelines_and_tuning.py` | Pipelines, GridSearchCV, avoiding data leakage |
| 10 | `10_breast_cancer_classification.py` | Binary classification: 4 models, false negatives, malignant recall |
| 11 | `11_wine_multiclass.py` | Multiclass classification: 5 models incl. MLP, 3x3 confusion matrix |
| 12 | `12_neural_network_mlp.py` | Neural networks (MLPClassifier) in depth, why scaling is essential |
| 13 | `13_gradient_boosting.py` | Gradient boosting vs random forest, early stopping |

## The workflow every lesson repeats

1. Load data and split into `X` (features) and `y` (target)
2. Split into train and test sets
3. Pick a model, `fit` on the training data
4. `predict` on the test data
5. Evaluate with a suitable metric

## Reading

### Evaluation metrics

Key metric definitions and formulas for classification
(TP = true positive, TN = true negative, FP = false positive, FN = false negative).

| Metric | What it measures | Formula | When to use / best suited for |
|---|---|---|---|
| **Accuracy** | Overall proportion of correct predictions out of all predictions | $\frac{TP+TN}{TP+TN+FP+FN}$ | Balanced datasets where all classes are equally important. |
| **Precision** | How many of the predicted positives were actually positive (exactness) | $\frac{TP}{TP+FP}$ | When false positives are costly (e.g. spam detection, false alarms). |
| **Recall** (Sensitivity) | How many of the actual positives were caught by the model (completeness) | $\frac{TP}{TP+FN}$ | When false negatives are dangerous or costly (e.g. disease diagnosis, fraud detection). |
| **F1 Score** | Harmonic mean of precision and recall (balances both metrics) | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Imbalanced datasets, or when you need a single score combining precision and recall. |

See `lessons/07_metrics.py` for these in code and `BasicMaths/Precision_Recall.md` for a worked example.
