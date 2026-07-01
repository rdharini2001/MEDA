import numpy as np

from megcal.thresholds import select_threshold


def test_select_threshold_runs():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    tau, ba = select_threshold(y, s)
    assert 0 <= tau <= 1
    assert ba >= 0.5
