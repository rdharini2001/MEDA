# Results guide

All tables are plain CSV. Methods are named as in the code: `MEDA` is the full 23-candidate search, `LR_unweighted_05` the plain linear probe, `LR_balanced_tuned` the balanced probe with a training-derived cutoff, and `fixed_<head>_tuned` a fixed classifier with a training-derived cutoff.

## results/nested

`<encoder>/<finding>/<context>/` with contexts `outer_0` to `outer_4` and `historical_validation`.

| File | Content |
|---|---|
| `inner_candidates.csv` | every candidate: inner balanced accuracy, inner AUROC, inner balanced accuracy at 0.5, selected threshold |
| `metrics.csv` | the 14 decision rules scored on the held-out fold: balanced accuracy, AUROC, sensitivity, specificity, threshold |
| `predictions.csv` | per-study score, call and threshold for each rule |
| `COMPLETE.json` | selected candidate, threshold, candidate count |

Regenerate with `scripts/run_nested.py`.

## results/summary

Per encoder and protocol (`nested` or `historical_validation`): `*_macro.csv` (macro balanced accuracy and AUROC with 95% intervals), `*_paired.csv` (MEDA minus each other rule, with permutation and Holm-adjusted p-values), `*_targets.csv` and `*_target_intervals.csv` (per finding), `*_fold_metrics.csv`. Also `outer_family_threshold_stability.csv`, the score-only selection bootstrap tables, and `hypothetical_prior_shift.csv`, which recomputes accuracy and precision under assumed prevalences from the observed sensitivity and specificity. Regenerate the main tables with `scripts/summarize.py`.

## results/search_budget

`budget_metrics.csv` has held-out results for searches over 1, 6, 10, 16, 19 and 23 candidates (`scripts/run_search_budget.py`). `candidate_budget_macro.csv` averages them. `winners_curse_gap.csv` compares inner and outer balanced accuracy of the selected candidate. `selection_stability.csv` and `learning_curve_macro.csv` support the selection-consistency and label-efficiency analyses.

## results/metrics

Every metric for every rule: balanced accuracy, G-mean, F1, F2, MCC, kappa, AUPRC, sensitivity, specificity, precision, NPV, false-negative and false-positive rates, Brier score. `meda_rank_by_metric.csv` shows where MEDA ranks on each metric and `meda_significant_wins_losses.csv` counts the rules MEDA significantly beats and loses to on each metric; `meda_vs_all_extended.csv` has the paired differences. MEDA is not first on any single metric; the tables show where it ties and where it loses.

## results/efficiency

Training time per candidate, single-thread inference latency, serialized model size and arithmetic operations for each rule, plus the composition of the heads MEDA selects.

## results/external_test

`task2_per_target.csv` (per finding), `cohort_means_*.csv` (cohort means), `hidden_vs_validation_internal.csv` (validation versus challenge result on the internal cohort) and `hidden_test_meda_vs_all.csv` (paired differences between MEDA and each other system).
