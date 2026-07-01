from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def make_threshold_grid(scores, grid_min: float = 0.02, grid_max: float = 0.98, n_grid: int = 97):
    """Uniform probability grid plus empirical quantiles of validation scores."""
    scores = np.asarray(scores, dtype=float)
    uniform = np.linspace(grid_min, grid_max, n_grid)

    if scores.size == 0:
        return uniform

    quantiles = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    thresholds = np.unique(np.clip(np.concatenate([uniform, quantiles]), 0.0, 1.0))
    return thresholds


def select_threshold(y_true, scores, thresholds=None):
    """Select threshold that maximizes validation balanced accuracy."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if thresholds is None:
        thresholds = make_threshold_grid(scores)

    best_tau = 0.5
    best_ba = -np.inf

    for tau in thresholds:
        pred = (scores >= tau).astype(int)
        ba = balanced_accuracy_score(y_true, pred)
        if ba > best_ba:
            best_ba = ba
            best_tau = float(tau)

    return best_tau, float(best_ba)
