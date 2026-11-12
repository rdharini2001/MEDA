# Method

## Setting

A pretrained CT encoder maps each study to an embedding `z`. For each finding `t`, MEDA chooses a classifier `g_t` and a threshold `tau_t`, and predicts a positive when `g_t(z) >= tau_t`. Both are chosen to maximize balanced accuracy,

    BA = 1/2 * (TP / (TP + FN) + TN / (TN + FP)),

so sensitivity and specificity count equally regardless of how rare the finding is. The encoder is never updated.

## Selection

For one finding and its training cases:

1. Split the training cases into 3 stratified folds (seed 42).
2. For each candidate classifier, fit it on two folds and score the third. The result is one out-of-fold score per training case.
3. Search thresholds on those scores. The grid is 97 evenly spaced values from 0.02 to 0.98 together with 97 quantiles of the candidate's own scores. The threshold with the highest balanced accuracy wins. If several thresholds tie, the lowest one is used.
4. Rank candidates by balanced accuracy at their best threshold. Ties go to the higher inner AUROC, then to the candidate name.
5. Refit the winner on all training cases and store it with its threshold.

`meda.selection.select_candidate` implements steps 1 to 4 and `meda.MEDA.fit` adds step 5.

An unweighted logistic regression is scored alongside the candidates. It is the plain linear probe used in the comparisons and is never selected.

## MEDA-Lite

MEDA-Lite applies the same rule to the first K candidates in the order given in [candidate_bank.md](candidate_bank.md). K is 10 for Merlin and 16 for CT-NEXUS. These sizes were chosen after looking at held-out outer-fold results, so the Lite numbers are exploratory. The original challenge submission searched the full bank.

## Evaluation

Nested cross-validation measures the whole procedure, including selection:

- 5 outer stratified folds over the training split, seed 42.
- Inside each outer training split, selection uses 3 inner folds.
- The chosen classifier is refitted on the outer training split and applied to the outer test fold with the frozen threshold.
- Outer test labels are never seen during selection. `tests/test_selection.py` checks this by flipping the test labels and confirming the choice does not change.

Fourteen decision rules are scored on the same folds: the unweighted probe at 0.5, with a prior-corrected cutoff and with a tuned cutoff; the balanced probe at 0.5 and with a tuned cutoff; MEDA; MEDA's selected head at 0.5; MEDA's selection made at 0.5; and six fixed heads (PCA8 logistic, sparse logistic, ExtraTrees, random forest, histogram boosting, XGBoost) each with a threshold chosen on the training split.

`historical_validation` is the original challenge-style context: fit on the full training split, then score the provided validation split once.

## Uncertainty

`meda.stats.uncertainty` computes confidence intervals with a clustered conditional bootstrap. Each scan gets one exponential weight that is shared across findings, folds and methods, because AMOS scans appear in several findings. AUROC intervals use the influence function of the Mann-Whitney statistic. Paired differences against MEDA use a sign-flip permutation test on per-scan contributions, followed by Holm adjustment across comparators. Intervals condition on the fitted models and the splits, so they do not include variation from retraining.

## What was submitted to the challenge

The challenge trainer chose among the 23 candidates on the provided validation split and refitted on all labeled cases. It did not use the nested inner folds above. The paper's matched comparisons use the nested protocol, and the independent-cohort results refer to the submitted trainer.
