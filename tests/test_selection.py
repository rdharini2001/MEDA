import numpy as np
import pytest

from meda import MEDA
from meda.nested import evaluate_fold
from meda.selection import CONTROL, select_candidate


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n, d = 150, 40
    y = (rng.uniform(size=n) < 0.25).astype(int)
    X = rng.normal(size=(n, d))
    X[:, :5] += 1.2 * y[:, None]
    return X, y, np.array([f"s{i}" for i in range(n)])


def test_selection_is_deterministic_and_excludes_control(data):
    X, y, ids = data
    a = select_candidate(X, y, ids, budget=10)
    b = select_candidate(X, y, ids, budget=10)
    assert a.chosen["name"] == b.chosen["name"] and a.chosen["threshold"] == b.chosen["threshold"]
    assert a.chosen["name"] != CONTROL
    assert len(a.table) == len(a.bank)


def test_lite_budget_restricts_choice(data):
    X, y, ids = data
    sel = select_candidate(X, y, ids, budget=10)
    assert sel.chosen["family"] in {"Balanced logistic", "Sparse logistic"}


def test_outer_labels_do_not_influence_selection(data):
    X, y, ids = data
    train, test = np.arange(100), np.arange(100, 150)
    first, _, _ = evaluate_fold(X, y, ids, train, test)
    y_changed = y.copy()
    y_changed[test] = 1 - y_changed[test]
    second, _, _ = evaluate_fold(X, y_changed, ids, train, test)
    assert first.chosen["name"] == second.chosen["name"]
    assert first.chosen["threshold"] == second.chosen["threshold"]


def test_overlapping_scans_are_rejected(data):
    X, y, ids = data
    with pytest.raises(ValueError):
        evaluate_fold(X, y, ids, np.arange(100), np.arange(90, 150))


def test_estimator_interface(data):
    X, y, _ = data
    model = MEDA(budget=10).fit(X[:100], y[:100])
    scores = model.decision_scores(X[100:])
    assert scores.shape == (50,)
    assert set(np.unique(model.predict(X[100:]))) <= {0, 1}
    assert np.array_equal(model.predict(X[100:]), (scores >= model.threshold_).astype(int))
