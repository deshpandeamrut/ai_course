# Absolute Deviation, Simply

## What it is
Absolute deviation measures **how far each value is from the centre, ignoring the direction**. "Absolute" means we drop the minus sign: |-2| = 2 and |+2| = 2.

Steps for the **mean absolute deviation (MAD)**:
1. Find the mean.
2. Subtract the mean from each value.
3. Drop the minus signs (take the absolute value).
4. Average the results.

It is a cousin of the SD. See [MeanAndSD.md](MeanAndSD.md) and [Variance.md](Variance.md).

## Worked example
Class A scores: 70, 72, 68, 71, 69 (mean = 70).

| score | difference from mean | absolute deviation |
|---|---|---|
| 70 | 0 | 0 |
| 72 | +2 | 2 |
| 68 | -2 | 2 |
| 71 | +1 | 1 |
| 69 | -1 | 1 |

Mean absolute deviation = (0 + 2 + 2 + 1 + 1) ÷ 5 = **1.2**.

Compare this with the SD of the same data, **1.41**. Both say "the scores are typically 1 to 2 points from the average", but they are not the same number.

## Absolute deviation vs SD

| | Mean absolute deviation | Standard deviation |
|---|---|---|
| How it removes minus signs | absolute value | squaring |
| Big gaps | count in proportion | count extra (squared) |
| Easy to explain? | very ("average distance from the mean") | a bit harder |
| Used by sklearn's `StandardScaler` | no | yes |

SD is usually a bit larger than the mean absolute deviation, because squaring gives big gaps extra weight.

| data | mean absolute deviation | SD |
|---|---|---|
| Class A (70, 72, 68, 71, 69) | 1.20 | 1.41 |
| Class B (40, 100, 60, 90, 60) | 20.00 | 21.91 |

For real cancer data (mean radius, 569 rows) the mean absolute deviation is 2.75 and the SD is 3.52, a ratio of 0.78. Mean area (0.75) and mean smoothness (0.79) are similar, but the heavily skewed area error column gives only 0.60. A ratio near 0.8 is typical of bell-shaped data, and skewed data with extreme values pushes it lower. It is not a law.

(The SD here is the population SD, dividing by n, so mean area shows 351.6. The 351.9 in other notes divides by n-1. See [Variance.md](Variance.md).)

## The robust version: median absolute deviation
Use the **median** instead of the mean at both steps:
1. Find the median.
2. Take the absolute difference of each value from the median.
3. Take the **median** of those.

Both versions are called "MAD", so check which one a source means.

### Why bother? Outliers
Add one extreme score (300) to Class A:

| scores | mean abs. deviation | SD | median abs. deviation |
|---|---|---|---|
| 70, 72, 68, 71, 69 | 1.20 | 1.41 | 1.0 |
| 70, 72, 68, 71, 69, **300** | **63.89** | **85.73** | **1.5** |

One outlier makes the SD jump from 1.41 to 85.73 and the mean absolute deviation jump from 1.2 to 63.89. The **median absolute deviation barely moves** (1.0 to 1.5). So it describes the typical spread of the other five scores.

## Real data: mean area in the cancer set

| measure | value |
|---|---|
| mean absolute deviation | 263.5 |
| SD | 351.6 |
| median absolute deviation | 153.3 |

The median absolute deviation (153) is much smaller than the SD (352). The area column is skewed by a few very large tumours, and the SD is more affected by them.

### The outlier count depends on the yardstick
The largest area in the data is 2501. How extreme it looks depends on how you measure:

| rule | what it uses | rows flagged as outliers in mean area |
|---|---|---|
| z-score beyond ±3 | mean and SD | **8** |
| modified z-score beyond ±3.5 | median and median absolute deviation | **24** |

The robust rule flags more rows because its spread estimate is smaller and isn't inflated by the big tumours themselves. The two rules disagree on which rows are outliers. Neither is "correct". Pick one on purpose and don't treat the count as an exact fact. The modified z-score is `0.6745 × (value - median) ÷ median absolute deviation`, a common convention.

## Where it shows up in the lesson
Nowhere directly. `StandardScaler` uses the mean and SD. sklearn's `RobustScaler` uses the median and the interquartile range, a related idea, which makes it less affected by outliers. This note is background for understanding "spread" in general.
