# Permutations and Combinations, Simply

Both are ways of **counting**: how many ways can I pick or arrange things?

The only question that separates them is: **does the order matter?**

| | order matters? | example |
|---|---|---|
| **Permutation** | yes | a race: gold, silver, bronze |
| **Combination** | no | picking a team: who is on it |

## Factorial (the building block)
`n!` ("n factorial") means n × (n-1) × ... × 1.

- 3! = 3 × 2 × 1 = **6**
- 5! = 5 × 4 × 3 × 2 × 1 = **120**

It counts the ways to line up n different things in a row.

## Permutations: order matters
Line up all of A, B, C. There are 3! = **6** ways:

ABC, ACB, BAC, BCA, CAB, CBA

To pick and arrange **k** items out of **n**:

```
P(n, k) = n! / (n - k)!
```

Example: gold, silver, bronze from 5 runners: P(5, 3) = 5 × 4 × 3 = **60**.

## Combinations: order doesn't matter
Choose 2 people from A, B, C, D, E for a team. AB is the same team as BA.

```
C(n, k) = n! / (k! × (n - k)!)
```

Example: C(5, 2) = 10 teams: AB, AC, AD, AE, BC, BD, BE, CD, CE, DE.

Combination count = permutation count ÷ k!, because each group of k items can be arranged in k! orders that we now treat as the same.

- Choosing 3 of 5 runners **in order**: P(5, 3) = 60.
- Choosing 3 of 5 runners **as a group**: C(5, 3) = 60 ÷ 3! = **10**.

## Python
```python
import math
math.factorial(5)   # 120
math.perm(5, 3)     # 60
math.comb(5, 3)     # 10
```

## Where this shows up in the lesson

### 1. Why a train/test split has so many possibilities
`train_test_split` chooses which 114 of the 569 rows go to the test set. Order doesn't matter, so that is a combination:

> C(569, 114) is about **2.5 × 10¹²²** possible splits.

That is a 123-digit number. `random_state=42` picks exactly one of them, and the same seed always picks the same one. See the explanation of `random_state` in the lesson.

### 2. How many malignant cases land in the test set?
Because the split is random, the number of malignant tumours in the test set varies (the data has 212 malignant out of 569, so we'd expect about 42.5 of the 114).

Counting with combinations (the hypergeometric calculation), for a random, non-stratified split:

| outcome in the 114 test rows | probability |
|---|---|
| exactly 43 malignant (what seed 42 gave) | about **8.6%** |
| between 38 and 47 malignant | about **72%** |

The seeds we tried earlier gave 43, 47, 42 and 45 malignant, all inside that range. This is why different splits give slightly different accuracy, and why `stratify=y` (which fixes the proportion) is exercise 4.

### 3. Comparing the four models
- **Permutation:** the number of possible rankings of 4 models is 4! = **24**. Exercise 3 asks "does the winner change?", and there are many orderings it could change into.
- **Combination:** the number of pairs of models to compare head-to-head is C(4, 2) = **6**.

### 4. Choosing features
The data has 30 features. Choosing which subset to use is a combination problem:

| choose | ways |
|---|---|
| 2 features | C(30, 2) = **435** |
| any subset at all | 2³⁰ = **1,073,741,824** |

That is why people don't try every subset of features by hand. There are too many.

### 5. KNN neighbours
With the 455 training rows, KNN with k=5 picks 5 neighbours. The number of possible groups of 5 is C(455, 5) = **158,964,146,341**. KNN doesn't test all of them. It just takes the 5 nearest by distance, which is why the neighbour vote is quick.

## Quick check
| question | permutation or combination? | answer |
|---|---|---|
| 8 runners, who gets gold, silver, bronze? | permutation | 8 × 7 × 6 = 336 |
| pick 3 of 8 for a relay team | combination | C(8, 3) = 56 |
| arrange 4 books on a shelf | permutation | 4! = 24 |
| pick 2 of 30 features | combination | C(30, 2) = 435 |

## Where to go next
Probability uses these counts: **probability = favourable ways ÷ total ways**. That is how the 8.6% above was computed.
