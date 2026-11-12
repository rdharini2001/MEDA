"""Estimator interface: ``MEDA().fit(X, y)`` selects and refits one classifier with its threshold."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from .bank import LITE_SIZE
from .selection import select_candidate


class MEDA(BaseEstimator, ClassifierMixin):
    """Select a classifier and operating threshold for one finding.

    Parameters
    ----------
    budget : int or None
        Number of candidates searched. ``None`` searches the full bank
        (MEDA-Full); ``10`` and ``16`` are the MEDA-Lite sizes for Merlin and
        CT-NEXUS features.
    inner : int
        Inner stratified folds used to score candidates and thresholds.
    seed : int
        Seed for folds and stochastic classifiers.
    """

    def __init__(self, budget=None, inner=3, seed=42):
        self.budget = budget
        self.inner = inner
        self.seed = seed

    @classmethod
    def lite(cls, backbone, **kwargs):
        return cls(budget=LITE_SIZE[backbone], **kwargs)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.selection_ = select_candidate(X, y, seed=self.seed, inner=self.inner, budget=self.budget)
        self.name_ = self.selection_.chosen["name"]
        self.threshold_ = float(self.selection_.chosen["threshold"])
        self.model_ = clone(dict(self.selection_.bank)[self.name_]).fit(X, y)
        return self

    def decision_scores(self, X):
        return self.model_.predict_proba(np.asarray(X))[:, 1]

    def predict_proba(self, X):
        s = self.decision_scores(X)
        return np.c_[1 - s, s]

    def predict(self, X):
        return (self.decision_scores(X) >= self.threshold_).astype(int)
