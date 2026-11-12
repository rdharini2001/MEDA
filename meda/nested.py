"""Nested evaluation: classifier and threshold are chosen without outer-test labels."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .bank import family
from .metrics import decision_metrics
from .selection import CONTROL, select_candidate

FIXED_HEADS = (
    "pca8_logreg",
    "logreg_l1_C0.1",
    "extratrees_balanced",
    "rf_balanced",
    "histgb",
    "xgb_d2_lr0.03_n350",
)


def decision_rules(sel, y_train):
    """Every decision rule compared in the paper as ``(label, model name, threshold)``."""
    lookup = sel.table.set_index("name")
    weighted = "logreg_C0.5"
    rules = [
        ("LR_unweighted_05", CONTROL, 0.5),
        ("LR_prior_corrected", CONTROL, float(y_train.mean())),
        ("LR_unweighted_tuned", CONTROL, lookup.loc[CONTROL, "threshold"]),
        ("LR_balanced_05", weighted, 0.5),
        ("LR_balanced_tuned", weighted, lookup.loc[weighted, "threshold"]),
        ("MEDA", sel.chosen["name"], sel.chosen["threshold"]),
        ("MEDA_selected_head_05", sel.chosen["name"], 0.5),
        ("MEDA_selection_05", sel.chosen_at_half["name"], 0.5),
    ]
    for name in FIXED_HEADS:
        if name in lookup.index:
            rules.append((f"fixed_{name}_tuned", name, lookup.loc[name, "threshold"]))
    return rules


def evaluate_fold(X, y, ids, train, test, seed=42, inner=3, verbose=False):
    """Select on ``train`` and evaluate every decision rule on ``test``."""
    if set(ids[train]) & set(ids[test]):
        raise ValueError("a scan appears in both outer training and test")
    sel = select_candidate(X[train], y[train], ids[train], seed=seed, inner=inner, verbose=verbose)
    models = dict(sel.bank)
    fitted, reports, predictions = {}, [], []
    for method, name, tau in decision_rules(sel, y[train]):
        if name not in fitted:
            fitted[name] = clone(models[name]).fit(X[train], y[train])
        scores = fitted[name].predict_proba(X[test])[:, 1]
        reports.append(
            dict(method=method, selected_name=name, family=family(name), **decision_metrics(y[test], scores, tau))
        )
        predictions += [
            dict(case_id=c, label=int(z), score=float(s), prediction=int(s >= tau), threshold=tau, method=method)
            for c, z, s in zip(ids[test], y[test], scores)
        ]
    return sel, pd.DataFrame(reports), pd.DataFrame(predictions)


def run_nested(X, y, ids, train_index, outer=5, inner=3, seed=42, output=None, verbose=False):
    """Run ``outer`` stratified folds over the training split.

    Returns the concatenated metrics table. When ``output`` is given, each fold
    writes its candidate table, metrics and predictions under ``outer_<k>``.
    """
    outer_folds = StratifiedKFold(outer, shuffle=True, random_state=seed)
    tables = []
    for k, (a, b) in enumerate(outer_folds.split(X[train_index], y[train_index])):
        sel, metrics, predictions = evaluate_fold(X, y, ids, train_index[a], train_index[b], seed, inner, verbose)
        metrics.insert(0, "fold", k)
        tables.append(metrics)
        if output is not None:
            folder = Path(output) / f"outer_{k}"
            folder.mkdir(parents=True, exist_ok=True)
            sel.table.to_csv(folder / "inner_candidates.csv", index=False)
            metrics.to_csv(folder / "metrics.csv", index=False)
            predictions.to_csv(folder / "predictions.csv", index=False)
    return pd.concat(tables, ignore_index=True)
