import numpy as np
import pandas as pd
import pytest

from meda.stats import auc_influence, holm, uncertainty


def test_holm_is_monotone_and_bounded():
    adjusted = holm([0.01, 0.04, 0.03, 0.5])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_auc_influence_sums_to_zero_per_class():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    s = rng.normal(size=80) + y
    auc, infl = auc_influence(y, s)
    assert 0 < auc < 1
    assert infl[y == 1].sum() == pytest.approx(0, abs=1e-12)
    assert infl[y == 0].sum() == pytest.approx(0, abs=1e-12)


def test_uncertainty_recovers_point_estimates():
    rng = np.random.default_rng(1)
    rows = []
    for method, shift in (("MEDA", 1.0), ("LR", 0.4)):
        for t in ("a", "b"):
            for fold in range(2):
                y = rng.integers(0, 2, 60)
                s = 1 / (1 + np.exp(-(shift * (2 * y - 1) + rng.normal(size=60))))
                rows.append(
                    pd.DataFrame(
                        dict(
                            method=method, target=t, fold=fold, label=y, score=s, prediction=(s >= 0.5).astype(int),
                            cluster=[f"{t}:{fold}:{i}" for i in range(60)],
                        )
                    )
                )
    pred = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame(
        [
            dict(method=m, balacc=g.assign(c=g.prediction == g.label).groupby(["target", "fold", "label"]).c.mean().groupby(["target", "fold"]).mean().mean())
            for m, g in pred.groupby("method")
        ]
    )
    summary, paired = uncertainty(pred, metrics, n_boot=200, permutations=500)
    assert set(summary.method) == {"MEDA", "LR"}
    assert (summary.balacc_low <= summary.balacc).all() and (summary.balacc <= summary.balacc_high).all()
    assert len(paired) == 1 and 0 < paired.p_holm.iloc[0] <= 1
