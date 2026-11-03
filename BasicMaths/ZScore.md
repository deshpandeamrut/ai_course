# Z-score, Simply

## What it is
A z-score answers one question: **how many standard deviations is this value from the average?**

```
z = (value - mean) / SD
```

- z = 0 means exactly average.
- z = +1 means one SD above average.
- z = -2 means two SDs below average.

You need [MeanAndSD.md](MeanAndSD.md) first.

## Why it's useful: comparing different things fairly
Two classes, both with a mean of 70:

| | mean | SD |
|---|---|---|
| Class A | 70 | 1.41 |
| Class B | 70 | 21.91 |

A student scores 90 in each class:

| | calculation | z-score | meaning |
|---|---|---|---|
| Class A | (90 - 70) ÷ 1.41 | **14.14** | astonishingly far above everyone |
| Class B | (90 - 70) ÷ 21.91 | **0.91** | a bit above average, quite normal |

The same raw score means very different things. The z-score makes it comparable.

## A rough guide
| z-score | how unusual |
|---|---|
| between -1 and +1 | very common |
| beyond ±2 | fairly rare |
| beyond ±3 | very rare, often called an outlier |

(This guide fits data shaped like a bell curve. Real data, like the cancer areas, is often skewed, so treat it as a rule of thumb.)

## Where it shows up in the lesson
`StandardScaler` turns **every value in every column into a z-score**. That is all it does.

Real rows from the breast cancer training data:

| feature | raw value | training mean | training SD | z-score |
|---|---|---|---|---|
| mean area | 250.5 | 654.4 | 354.6 | **-1.14** |
| mean area | 1311.0 | 654.4 | 354.6 | **+1.85** |
| worst concavity | 1.25 | not shown | not shown | **+4.67** |

- Area 250.5 is 1.14 SDs below average: a small tumour.
- Area 1311.0 is 1.85 SDs above average: a large tumour.
- The concavity of 4.67 is extreme. That row is benign, so extreme does not automatically mean cancer.

## What scaling gives you
After converting a column to z-scores:
- the **mean becomes 0**,
- the **SD becomes 1**,
- every feature uses the same unit ("number of SDs"), so area (hundreds) and smoothness (about 0.1) can be compared.

The scaler uses the **training rows only** to get the mean and SD. The test rows are converted with those same numbers, which is why the test columns come out close to 0 and 1 but not exactly.
