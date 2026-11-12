import numpy as np
import pytest

from meda.thresholds import select_threshold, threshold_grid


def test_grid_contains_uniform_levels_and_quantiles():
    scores = np.random.default_rng(0).uniform(size=200)
    grid = threshold_grid(scores)
    assert np.all(np.diff(grid) > 0)
    assert np.isclose(grid, 0.5).any()
    assert np.isclose(grid, np.quantile(scores, 0.5)).any()


def test_perfectly_separable_scores_reach_one():
    y = np.array([0] * 20 + [1] * 20)
    scores = np.r_[np.linspace(0.05, 0.4, 20), np.linspace(0.6, 0.95, 20)]
    tau, ba = select_threshold(y, scores)
    assert ba == pytest.approx(1.0)
    assert 0.4 < tau <= 0.6


def test_matches_direct_balanced_accuracy():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 150)
    scores = np.clip(0.3 * y + rng.normal(0.35, 0.2, 150), 0, 1)
    tau, ba = select_threshold(y, scores)
    pred = scores >= tau
    direct = 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
    assert ba == pytest.approx(direct)
    best = max(
        0.5 * ((scores >= t)[y == 1].mean() + (~(scores >= t))[y == 0].mean()) for t in threshold_grid(scores)
    )
    assert ba == pytest.approx(best)


def test_single_class_is_rejected():
    with pytest.raises(ValueError):
        select_threshold(np.zeros(10), np.linspace(0, 1, 10))
