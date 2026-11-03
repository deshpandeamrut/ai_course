# Correlation, Simply

## What it is
Correlation measures **how closely two columns move together**. It is a single number from **-1 to +1**.

| value | meaning | example |
|---|---|---|
| +1 | perfectly move together: one goes up, the other goes up | height in cm and height in inches |
| 0 | no link | shoe size and exam score |
| -1 | perfect opposites: one goes up, the other goes down | hours of battery used and battery left |

Values in between show how strong the link is. 0.9 is very strong, 0.3 is weak.

Correlation says two things move together. It does **not** say one causes the other.

## Real examples from the breast cancer data

| column 1 | column 2 | correlation | reading |
|---|---|---|---|
| mean radius | mean perimeter | **0.998** | essentially the same information |
| mean radius | mean area | **0.987** | very strong |
| mean concavity | mean concave points | **0.921** | very strong |
| mean radius | mean texture | 0.324 | weak |
| mean radius | mean smoothness | 0.171 | almost no link |

## Why radius, perimeter and area move together
They all describe the size of the same round shape. Area is roughly π × radius², and the first three rows of the data confirm it: area ÷ (π × radius²) comes out at 0.985, 0.998 and 0.988, very close to 1. So a tumour with a bigger radius has a bigger perimeter and a bigger area.

## Correlation with the answer
Correlation between mean radius and the label (0 = malignant, 1 = benign) is **-0.73**.

- It is negative because **malignant is 0**. Bigger radius means a lower label, which means more likely malignant.
- It is strong, which is why radius and area are useful for predicting cancer.
- The minus sign comes from how the labels are coded. It does not mean "bad". Flip the labels and the sign flips.

## Why this matters for the models
1. **Redundant features.** Many of the 30 columns are strongly correlated, so they repeat each other. That is fine for prediction.
2. **Hard-to-read coefficients.** When features overlap, Logistic Regression can split credit between them in odd ways, so don't over-read a single coefficient.
3. **Not everything correlates.** Radius and smoothness (0.171) carry different information, which is why using all 30 features helps.

## How to compute it
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

d = load_breast_cancer()
df = pd.DataFrame(d.data, columns=d.feature_names)
print(df["mean radius"].corr(df["mean area"]))   # 0.987
```

This is the Pearson correlation, which measures straight-line relationships. It can miss curved patterns.
