"""Inner cross-validation: score every candidate, pick one classifier and threshold."""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .bank import build_bank, family, lite_names, unweighted_logistic
from .thresholds import select_threshold

CONTROL = "unweighted_logreg_C0.5"


@dataclass
class SelectionResult:
    bank: list
    table: pd.DataFrame
    oof: np.ndarray
    chosen: pd.Series
    chosen_at_half: pd.Series


def rank_candidates(table, key="inner_balacc"):
    """Order candidates by ``key``, then inner AUROC, then name."""
    return table.sort_values([key, "inner_auroc", "name"], ascending=[False, False, True])


def select_candidate(X, y, ids=None, seed=42, inner=3, budget=None, verbose=False):
    """Select a classifier and threshold using training data only.

    The candidates are fitted on ``inner`` stratified folds. Each candidate's
    out-of-fold scores give its threshold and balanced accuracy. With
    ``budget=K`` only the first K candidates in search order are fitted and
    can be chosen (MEDA-Lite).
    """
    folds = list(StratifiedKFold(inner, shuffle=True, random_state=seed).split(X, y))
    if ids is not None:
        for tr, va in folds:
            if set(ids[tr]) & set(ids[va]):
                raise ValueError("a scan appears in both inner training and validation")
    bank = build_bank(min(len(tr) for tr, _ in folds), X.shape[1], seed)
    if budget is not None:
        keep = set(lite_names([n for n, _ in bank], budget))
        bank = [(n, m) for n, m in bank if n in keep]
    bank.append((CONTROL, unweighted_logistic(seed)))
    rows, oof = [], []
    for name, model in bank:
        start = time.monotonic()
        scores = np.full(len(y), np.nan)
        for tr, va in folds:
            fitted = clone(model).fit(X[tr], y[tr])
            scores[va] = fitted.predict_proba(X[va])[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError(f"incomplete out-of-fold scores for {name}")
        tau, ba = select_threshold(y, scores)
        rows.append(
            dict(
                name=name,
                family=family(name),
                inner_balacc=ba,
                inner_auroc=float(roc_auc_score(y, scores)),
                inner_balacc_05=float(balanced_accuracy_score(y, scores >= 0.5)),
                threshold=tau,
                seconds=time.monotonic() - start,
            )
        )
        oof.append(scores)
        if verbose:
            print(f"{name:<28s} inner BA {ba:.4f}", flush=True)
    table = pd.DataFrame(rows)
    search = table[table.name != CONTROL]
    chosen = rank_candidates(search).iloc[0]
    chosen_at_half = rank_candidates(search, "inner_balacc_05").iloc[0]
    return SelectionResult(bank, table, np.asarray(oof), chosen, chosen_at_half)


def bootstrap_selection(y, oof, table, seed=42, n_boot=200):
    """Resample the stored out-of-fold scores and repeat the selection.

    Models are not refitted, so this only shows how stable the choice of
    classifier and threshold is given the inner scores.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    allowed = np.flatnonzero((table.name != CONTROL).to_numpy())
    records = []
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        found = [select_threshold(y[idx], oof[k, idx]) for k in allowed]
        aucs = [roc_auc_score(y[idx], oof[k, idx]) for k in allowed]
        j = max(range(len(allowed)), key=lambda i: (found[i][1], aucs[i], -i))
        row = table.iloc[allowed[j]]
        records.append(
            dict(bootstrap=b, name=row["name"], family=row["family"], threshold=found[j][0], inner_balacc=found[j][1])
        )
    return pd.DataFrame(records)
