import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_threshold_table_matches_stored_selections():
    table = pd.read_csv(ROOT / "results/reviewer/thresholds_all.csv")
    full = table[table.variant == "MEDA-Full"]
    assert len(full) == 2 * 17 * 6
    for row in full.itertuples():
        stored = json.loads((ROOT / "results/nested" / row.backbone / row.finding / row.context / "COMPLETE.json").read_text())
        assert stored["selected"] == row.classifier
        assert abs(stored["threshold"] - row.threshold) < 1e-12


def test_lite_selections_match_search_budget_table():
    table = pd.read_csv(ROOT / "results/reviewer/thresholds_all.csv")
    budget = pd.read_csv(ROOT / "results/search_budget/budget_metrics.csv")
    size = {"merlin": 10, "ctnexus": 16}
    lite = table[(table.variant == "MEDA-Lite") & (table.context != "historical_validation")]
    for row in lite.itertuples():
        b = budget[
            (budget.backbone == row.backbone)
            & (budget.target == row.finding)
            & (budget.fold == int(row.context[-1]))
            & (budget.requested_budget == size[row.backbone])
        ].iloc[0]
        assert b.selected_name == row.classifier
