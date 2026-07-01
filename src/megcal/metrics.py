from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def balanced_accuracy(y_true, y_pred) -> float:
    """Balanced accuracy: average of sensitivity and specificity."""
    return float(balanced_accuracy_score(y_true, y_pred))


def safe_auroc(y_true, scores) -> float:
    """AUROC with NaN fallback when validation labels contain one class."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))
