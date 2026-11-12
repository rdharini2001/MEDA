"""Operating-threshold search that maximizes balanced accuracy."""
import numpy as np


def threshold_grid(scores):
    """Uniform grid on [0.02, 0.98] joined with quantiles of the scores."""
    levels = np.linspace(0.02, 0.98, 97)
    return np.unique(np.r_[levels, np.quantile(scores, levels)])


def select_threshold(y, scores):
    """Return ``(threshold, balanced_accuracy)`` for the best cutoff.

    A case is called positive when its score is at least the threshold. Ties
    resolve to the lowest threshold.
    """
    y = np.asarray(y)
    scores = np.asarray(scores)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both classes are required to select a threshold")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    labels = y[order]
    cum_pos = np.r_[0, np.cumsum(labels)]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    grid = threshold_grid(scores)
    below = np.searchsorted(sorted_scores, grid, side="left")
    ba = 0.5 * ((n_pos - cum_pos[below]) / n_pos + (below - cum_pos[below]) / n_neg)
    best = int(np.argmax(ba))
    return float(grid[best]), float(ba[best])
