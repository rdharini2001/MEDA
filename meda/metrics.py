"""Per-fold decision metrics."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def decision_metrics(y, scores, tau):
    pred = scores >= tau
    return dict(
        balacc=float(balanced_accuracy_score(y, pred)),
        auroc=float(roc_auc_score(y, scores)),
        sensitivity=float(pred[y == 1].mean()),
        specificity=float((~pred[y == 0]).mean()),
        threshold=float(tau),
        n=len(y),
        positives=int(y.sum()),
    )
