# Distance (for KNN), Simply

## The idea
KNN predicts a tumour by finding the **closest** training tumours and letting them vote. To do that it needs a way to measure "how close".

## Straight-line (Euclidean) distance
Like measuring with a ruler. For two features:

```
distance = sqrt( (difference in feature 1)² + (difference in feature 2)² )
```

With 30 features, you add all 30 squared differences under the square root.

Small example, a 3-4-5 triangle: if the differences are 3 and 4, the distance is √(9 + 16) = √25 = **5**.

## Why scaling matters: a real example
Compare tumours using just two features: **mean area** and **mean smoothness**. These are real training rows.

| row | mean area | mean smoothness | label |
|---|---|---|---|
| 0 | 250.5 | 0.107 | benign |
| 1 | 1311.0 | 0.114 | malignant |
| 2 | 260.9 | 0.077 | benign |

### Raw (unscaled) distances
- Row 0 vs row 1: differences are 1060.5 (area) and 0.007 (smoothness). Distance = √(1060.5² + 0.007²) ≈ **1060.5**.
- Row 0 vs row 2: differences are 10.4 (area) and 0.030 (smoothness). Distance = √(10.4² + 0.030²) ≈ **10.4**.

The smoothness gap contributes essentially **nothing**. The distance is just the area gap.

### After scaling (z-scores, see [ZScore.md](ZScore.md))

| row | area (scaled) | smoothness (scaled) |
|---|---|---|
| 0 | -1.14 | 0.78 |
| 1 | 1.85 | 1.32 |
| 2 | -1.11 | -1.33 |

- Row 0 vs row 1: differences 2.99 and 0.54, so distance = **3.04**.
- Row 0 vs row 2: differences 0.03 and 2.11, so distance = **2.11**.

Now smoothness matters. Rows 0 and 2 have almost the same area, but their smoothness is quite different (0.78 against -1.33), and the scaled distance shows that gap. The raw distance hid it.

## With all 30 features
Row 0 vs row 2, using all 30 columns:

| | distance |
|---|---|
| raw | 19.56 |
| scaled | 12.30 |

Without scaling, the big-number columns (area, perimeter) dominate. With scaling, all 30 columns have a fair say.

## A caution
In this lesson, KNN on unscaled data scored 0.9561 against 0.9474 scaled. That is one patient out of 114 on one split, so it's probably luck. Area is itself very informative, so letting it dominate didn't hurt much here. Scaling matters most when a large-valued feature is **not** informative. Scaling is still the safe default.

## Other distances
Euclidean is KNN's default in sklearn. Another is **Manhattan** distance, where you add the absolute differences, like walking along city blocks. You don't need it for this lesson.
