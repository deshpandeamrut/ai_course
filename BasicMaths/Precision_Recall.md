# Accuracy, Precision and Recall, Simply

## Start with the confusion matrix
Logistic Regression on the 114 test tumours (rows = true class, columns = predicted class):

|  | predicted malignant | predicted benign |
|---|---|---|
| **truly malignant (43)** | **41** caught (true positive) | **2** missed (false negative) |
| **truly benign (71)** | **1** false alarm (false positive) | **70** correct (true negative) |

In this lesson **malignant is the class we care about**, even though it is labelled 0.

## The three numbers

| Metric | The question it answers | Calculation |
|---|---|---|
| **Accuracy** | Of all 114 tumours, how many did we get right? | (41 + 70) ÷ 114 = **0.9737** |
| **Recall** (malignant) | Of the 43 real cancers, how many did we catch? | 41 ÷ 43 = **0.9535** |
| **Precision** (malignant) | Of the 42 we called cancer, how many really were? | 41 ÷ 42 = **0.9762** |

Recall and precision have the **same top number (41)**. They differ in what you divide by:
- **Recall** divides by the real cancers (the first **row**, 43).
- **Precision** divides by the predicted cancers (the first **column**, 42).

A simple way to remember it:
- Recall asks "did we **find** them all?"
- Precision asks "when we raise the alarm, are we **right**?"

## Why accuracy alone is not enough
Accuracy treats every mistake the same. Here they aren't the same:
- **Missed cancer (false negative):** the patient goes untreated. Dangerous.
- **False alarm (false positive):** extra tests. An inconvenience.

So the lesson focuses on **recall of the malignant class**.

## F1 score
F1 combines precision and recall into one number:

```
F1 = 2 × precision × recall / (precision + recall)
```

For malignant: 2 × 0.9762 × 0.9535 ÷ (0.9762 + 0.9535) = **0.9647**. It is only high when both are high.

## The trap: the label 0 problem
sklearn's functions treat **class 1 as "positive" by default**. Here class 1 is **benign**. Compare the two ways of calling it:

| call | what it measures | Logistic Regression |
|---|---|---|
| `recall_score(y, pred)` | recall of **benign** | 70 ÷ 71 = **0.9859** |
| `recall_score(y, pred, pos_label=0)` | recall of **malignant** | 41 ÷ 43 = **0.9535** |

The default would give you a flattering number for the wrong class. That is why the lesson passes `pos_label=0`.

## The trade-off between recall and precision
Logistic Regression really outputs a probability, and `predict` says "benign" when P(benign) is at least 0.5. You can change that cut-off. Here is what happens on the same 114 test tumours when we require P(benign) to be at least the value in the first column before calling a tumour benign:

| need P(benign) of at least | missed cancers | false alarms |
|---|---|---|
| 0.5 (the default) | 2 | 1 |
| 0.7 | 1 | 4 |
| 0.9 | 1 | 7 |

Being stricter catches more cancers (fewer misses), but it raises more false alarms. You are trading one mistake for the other. How far to go is a judgement call, and in medicine a few extra tests are usually preferred to a missed cancer.

Two of the real mistakes are worth knowing about:
- Test row 77 is a malignant tumour that the model thought was **51.5%** likely benign, so it was a near call. A stricter cut-off catches it.
- Test row 20 is a malignant tumour that the model thought was **91.6%** likely benign. It was confidently wrong, and even the 0.9 cut-off doesn't catch it.

## Summary
| want to know | use |
|---|---|
| overall right/wrong | accuracy |
| "did we catch all the cancers?" | recall (malignant) |
| "are our cancer alarms trustworthy?" | precision (malignant) |
| one balanced number | F1 |
