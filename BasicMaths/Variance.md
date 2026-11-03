# Variance, Simply

## What it is
Variance measures **how spread out** numbers are. It is the **average of the squared distances from the mean**.

**Standard deviation (SD) = √variance**, and **variance = SD²**. They carry the same information. See [MeanAndSD.md](MeanAndSD.md) for SD.

## Worked example
Two classes, both with a mean of 70:

| | scores | mean | variance | SD |
|---|---|---|---|---|
| Class A | 70, 72, 68, 71, 69 | 70 | **2** | 1.41 |
| Class B | 40, 100, 60, 90, 60 | 70 | **480** | 21.91 |

How Class A's variance is calculated:
1. Mean = 70.
2. Differences from the mean: 0, +2, -2, +1, -1.
3. Square each one: 0, 4, 4, 1, 1.
4. Average them: (0 + 4 + 4 + 1 + 1) ÷ 5 = **2**.

## Why square the differences?
- The differences add up to zero (+2 and -2 cancel out), so you can't just average them.
- Squaring removes the minus signs.
- It also makes big gaps count extra. A gap of 20 contributes 400, while a gap of 2 contributes only 4.

## Variance vs SD
| | Variance | SD |
|---|---|---|
| Units | squared (points²) | same as the data (points) |
| Easy to picture? | no | yes ("typical distance from average") |
| Used for | maths and formulas | explaining and scaling |

Variance is awkward to picture because a spread of "480 points²" has no real-world meaning. That is why people usually quote SD.

## Where it shows up in the lesson
`StandardScaler` computes each column's mean and variance on the training rows, then divides by the square root of the variance (the SD). You don't see variance directly, but it's inside the scaler.

## Population vs sample
- Dividing by **n** gives the population variance (what `StandardScaler` and numpy's default use).
- Dividing by **n - 1** gives the sample variance (what pandas' default uses).
- With 455 training rows the difference is tiny.
