# Mean and Standard Deviation, Simply

## Mean = the average
Add everything up and divide by how many there are.

Say 5 friends scored these on a test: 70, 72, 68, 71, 69.
Total = 350, and 350 ÷ 5 = **70**. The mean is the "typical" value, the middle of the pack.

## Standard deviation (SD) = how spread out the numbers are
The mean alone doesn't tell you the whole story. Compare two classes that both have a mean of 70:

| | scores | mean | SD |
|---|---|---|---|
| Class A | 70, 72, 68, 71, 69 | 70 | **1.4** |
| Class B | 40, 100, 60, 90, 60 | 70 | **21.9** |

- In Class A everyone scored about the same, so the SD is small.
- In Class B the scores are all over the place, so the SD is big.

You can think of SD as **"the typical distance from the average"**. In Class A most people are within about 1-2 points of 70. In Class B they're 20 or more points away.

### How to calculate it (Class A)
1. Find the mean: 70.
2. Subtract the mean from each score: 0, +2, -2, +1, -1.
3. Square each difference (this removes the minus signs): 0, 4, 4, 1, 1.
4. Average the squares: (0 + 4 + 4 + 1 + 1) ÷ 5 = 2. This is called the **variance**.
5. Take the square root: √2 ≈ **1.4**. That is the SD.

(This is the "population" SD, dividing by n. It is the version `StandardScaler` uses. Some textbooks divide by n-1, which gives a slightly larger number.)

## Why it's useful: "how unusual is this value?"
Suppose someone scores 90.
- In Class B (SD 21.9), 90 is 20 points above the mean. That's less than one SD away, so it's pretty normal there.
- In Class A (SD 1.4), 90 is 20 points above the mean. That's about 14 SDs away, so it would be a huge outlier.

The same 90 means very different things depending on the SD. Measuring distance in SDs is fair across both classes.

## How this connects to the lesson
`StandardScaler` (used in `lessons/10_breast_cancer_classification.py`) does exactly this to every column:

```
scaled value = (value - mean) / SD
```

The answer says "how many SDs away from average is this value?"

Real numbers from the breast cancer data:

| feature | mean | SD |
|---|---|---|
| mean radius | 14.13 | 3.52 |
| mean area | 654.9 | 351.9 |
| mean smoothness | 0.10 | 0.01 |

The raw columns are on totally different scales, so you can't compare "area 250" with "smoothness 0.11" directly. After scaling they're all in the same unit, SDs:

- A tumour with area **250.5** becomes (250.5 - 654.4) ÷ 354.6 = **-1.14**. It's 1.14 SDs *below* average, so it's small. (These use the mean and SD of the training rows only, which is why they differ slightly from the whole-dataset table above.)
- That same tumour's worst concavity became **+4.67**. That is 4.67 SDs *above* average, which is very unusual. (This particular tumour is actually benign, so an extreme value is not automatically a sign of cancer. Models combine all 30 features.)

After scaling, a column has **mean 0 and SD 1**. Zero means "perfectly average", and positive or negative numbers say how far above or below you are. That's why every feature then gets an equal say in models like KNN.

## Rows after scaling
The first 6 training rows, using 5 of the 30 columns.

**Before scaling**

| | mean radius | mean texture | mean area | mean smoothness | worst concavity | label |
|---|---|---|---|---|---|---|
| 0 | 9.03 | 17.33 | 250.5 | 0.11 | 1.25 | benign |
| 1 | 21.09 | 26.57 | 1311.0 | 0.11 | 0.68 | malignant |
| 2 | 9.17 | 13.86 | 260.9 | 0.08 | 0.14 | benign |
| 3 | 10.65 | 25.22 | 347.0 | 0.10 | 0.11 | benign |
| 4 | 10.17 | 14.88 | 311.9 | 0.11 | 0.02 | benign |
| 5 | 14.54 | 27.54 | 658.8 | 0.11 | 0.70 | malignant |

**After scaling**

| | mean radius | mean texture | mean area | mean smoothness | worst concavity | label |
|---|---|---|---|---|---|---|
| 0 | -1.44 | -0.44 | -1.14 | 0.78 | 4.67 | benign |
| 1 | 1.97 | 1.73 | 1.85 | 1.32 | 1.93 | malignant |
| 2 | -1.40 | -1.25 | -1.11 | -1.33 | -0.64 | benign |
| 3 | -0.98 | 1.42 | -0.87 | 0.06 | -0.77 | benign |
| 4 | -1.12 | -1.01 | -0.97 | 1.27 | -1.21 | benign |
| 5 | 0.12 | 1.96 | 0.01 | 1.31 | 2.05 | malignant |

### What to notice
- **The columns are now comparable.** Area was in the hundreds and smoothness was about 0.1. Now every column is roughly between -2 and +2, so each one has a similar say in the model.
- **The sign tells you above or below average.** Row 1 (malignant) is positive everywhere: a bigger-than-average radius (+1.97) and area (+1.85). Rows 0, 2, 3 and 4 (benign) have mostly negative radius and area. This is the pattern the models learn.
- **Row 5 shows the mean.** Its radius of 14.54 is close to the average of about 14.1, so it scales to **0.12**, near zero.
- **Row 0 is an outlier.** Its concavity of **4.67** is far above average, yet it is **benign**. A high value doesn't automatically mean cancer, which is one reason models combine all 30 features instead of relying on one.

### Train vs test columns after scaling
Each column scaled on the **training** rows has mean 0 and SD 1 exactly. The **test** rows come out close to that, but not exactly:

| | mean radius | mean texture | mean area | mean smoothness | worst concavity |
|---|---|---|---|---|---|
| test mean | 0.01 | 0.12 | 0.01 | 0.22 | -0.06 |
| test SD | 0.98 | 1.04 | 0.96 | 1.03 | 0.98 |

This is expected, and it is the right behaviour. The scaler learned its mean and SD from the training rows only and then applied them to the test rows. If the test columns came out exactly 0 and 1, the scaler would have peeked at the test data (leakage).
