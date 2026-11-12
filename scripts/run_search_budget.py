"""Held-out accuracy as a function of the number of candidates searched.

Reads the inner candidate tables written by run_nested.py. Restricted searches
choose among the first K candidates in search order; the classifier is refitted
on the outer training split unless it is already among the evaluated rules.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meda.bank import build_bank, family, lite_names
from meda.data import BACKBONES, TARGETS, load_features
from meda.metrics import decision_metrics
from meda.selection import CONTROL, rank_candidates

BUDGETS = (1, 6, 10, 16, 19, 23)


def run(args, backbone, target):
    X, y, ids, split = load_features(args.features, backbone, target)
    train = np.flatnonzero(split == "train")
    rows = []
    folds = StratifiedKFold(args.outer, shuffle=True, random_state=args.seed).split(X[train], y[train])
    for k, (a, b) in enumerate(folds):
        tr, te = train[a], train[b]
        table = pd.read_csv(args.nested / backbone / target / f"outer_{k}" / "inner_candidates.csv")
        table = table[table.name != CONTROL]
        inner_min = min(len(i) for i, _ in StratifiedKFold(args.inner, shuffle=True, random_state=args.seed).split(X[tr], y[tr]))
        bank = dict(build_bank(inner_min, X.shape[1], args.seed))
        cache = {}
        for budget in BUDGETS:
            allowed = lite_names(table.name, budget)
            chosen = rank_candidates(table[table.name.isin(allowed)]).iloc[0]
            name, tau = chosen["name"], float(chosen["threshold"])
            if name not in cache:
                with threadpool_limits(limits=1):
                    cache[name] = clone(bank[name]).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            rows.append(
                dict(
                    backbone=backbone,
                    target=target,
                    fold=k,
                    requested_budget=budget,
                    candidates=len(allowed),
                    selected_name=name,
                    family=family(name),
                    inner_balacc=float(chosen.inner_balacc),
                    **decision_metrics(y[te], cache[name], tau),
                )
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="all", choices=["all", *BACKBONES])
    ap.add_argument("--target", default="all", choices=["all", *TARGETS])
    ap.add_argument("--features", type=Path, default=ROOT / "data/features")
    ap.add_argument("--nested", type=Path, default=ROOT / "results/nested")
    ap.add_argument("--output", type=Path, default=ROOT / "results/search_budget/budget_metrics.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    args = ap.parse_args()
    backbones = BACKBONES if args.backbone == "all" else [args.backbone]
    targets = TARGETS if args.target == "all" else [args.target]
    tables = []
    for b in backbones:
        for t in targets:
            tables.append(run(args, b, t))
            print("done", b, t, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
