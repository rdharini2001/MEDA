import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cohort_means_match_the_paper():
    spec = importlib.util.spec_from_file_location("external_table", ROOT / "scripts/external_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tables = module.build(pd.read_csv(ROOT / "results/external_test/task2_per_target.csv"))
    ba = tables["test_balanced_acc"]
    assert ba.loc["MEDA (Merlin)", "All cohorts"] == pytest.approx(0.6479, abs=1e-4)
    assert ba.loc["MEDA (Merlin)", "External-Abd-China"] == pytest.approx(0.6673, abs=1e-4)
    assert ba.loc["MEDA (Merlin)", "External-Chest-China"] == pytest.approx(0.5466, abs=1e-4)
    assert ba.loc["Merlin + LP", "All cohorts"] == pytest.approx(0.6537, abs=1e-4)
    assert ba["External-Abd-China"].idxmax() == "MEDA (Merlin)"
    assert ba["External-Chest-China"].idxmax() == "MEDA (Merlin)"
