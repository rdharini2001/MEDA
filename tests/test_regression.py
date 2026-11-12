"""Re-run one stored outer fold and compare with the shipped results."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from meda.data import load_features
from meda.nested import evaluate_fold

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_ctnexus_covid_fold_matches_stored_results():
    X, y, ids, split = load_features(ROOT / "data/features", "ctnexus", "covid")
    train = np.flatnonzero(split == "train")
    a, b = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X[train], y[train]))[1]
    with threadpool_limits(limits=1):
        _, metrics, _ = evaluate_fold(X, y, ids, train[a], train[b])
    stored = pd.read_csv(ROOT / "results/nested/ctnexus/covid/outer_1/metrics.csv")
    merged = metrics.merge(stored, on="method", suffixes=("", "_stored"))
    assert len(merged) == len(stored) == 14
    assert (merged.selected_name == merged.selected_name_stored).all()
    for column in ("balacc", "auroc", "sensitivity", "specificity", "threshold"):
        assert np.allclose(merged[column], merged[f"{column}_stored"], atol=1e-9)
