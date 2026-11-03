# Median and Outliers, Simply

## Median = the middle value
Sort the numbers and pick the one in the middle. With an even count, average the two middle ones.

Scores: 70, 72, 68, 71, 69. Sorted: 68, 69, **70**, 71, 72. The median is **70**.

## Mean vs median: the outlier test
Now one friend gets a huge score:

| scores | mean | median |
|---|---|---|
| 70, 72, 68, 71, 69 | 70 | 70 |
| 70, 72, 68, 71, 69, **300** | **108.3** | **70.5** |

One extreme value dragged the mean from 70 up to 108.3. The median barely moved. The median is **robust** to extreme values, and the mean is not.

## What is an outlier?
A value far away from the rest. A common rule of thumb is a z-score beyond about ±3 (see [ZScore.md](ZScore.md)).

## Real data: the breast cancer set

| feature | mean | median | max |
|---|---|---|---|
| mean radius | 14.13 | 13.37 | 28.11 |
| mean area | 654.89 | 551.10 | 2501.00 |
| mean smoothness | 0.10 | 0.10 | 0.16 |
| area error | 40.34 | 24.53 | 542.20 |

How to read this:
- **Mean far above median** means the data is stretched toward big values. Mean area is 655 but the median is 551, so a few very large tumours pull the average up.
- **Mean about equal to median** means the data is fairly balanced. Smoothness is the example.
- **Area error** is the extreme case: a mean of 40 against a median of 25, with a max of 542.

## Outliers in mean area
Using a z-score beyond ±3 on the whole dataset:
- **8 rows** are outliers.
- The largest has an area of 2501, which is a z-score of **5.25**.
- **All 8 of them are malignant** (label 0).

So here, the outliers carry real information: very large tumours tend to be cancerous. Do not automatically delete outliers. First ask whether they are errors or real, important cases.

## When to use which
| situation | better summary |
|---|---|
| balanced data with no extremes | mean |
| skewed data or extreme values | median |
| you care about spread | SD (see [MeanAndSD.md](MeanAndSD.md)) |

## A note on scaling
`StandardScaler` uses the **mean and SD**, so outliers affect it. A few giant areas pull the mean and SD up, which squeezes the typical tumours into a narrower range after scaling. sklearn also has `RobustScaler`, which uses the median instead. The lesson sticks with `StandardScaler`.
