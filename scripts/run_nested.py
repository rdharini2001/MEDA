"""Five-fold nested evaluation of every decision rule.

Example:
    python scripts/run_nested.py --backbone merlin --target fatty_liver
    python scripts/run_nested.py --backbone all --target all --jobs 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meda.data import BACKBONES, TARGETS, load_features
from meda.nested import evaluate_fold
from meda.selection import bootstrap_selection


def run_context(args, backbone, target, name, train, test):
    folder = args.output / backbone / target / name
    if (folder / "COMPLETE.json").exists() and not args.force:
        return f"skip {backbone}/{target}/{name}"
    X, y, ids, _ = load_features(args.features, backbone, target)
    with threadpool_limits(limits=1):
        sel, metrics, predictions = evaluate_fold(X, y, ids, train, test, args.seed, args.inner)
    folder.mkdir(parents=True, exist_ok=True)
    sel.table.to_csv(folder / "inner_candidates.csv", index=False)
    metrics.to_csv(folder / "metrics.csv", index=False)
    predictions.to_csv(folder / "predictions.csv", index=False)
    if args.selection_bootstrap:
        boot = bootstrap_selection(y[train], sel.oof, sel.table, args.seed, args.selection_bootstrap)
        boot.to_csv(folder / "selection_bootstrap.csv", index=False)
    summary = dict(
        seed=args.seed,
        inner=args.inner,
        selected=sel.chosen["name"],
        threshold=float(sel.chosen["threshold"]),
        candidate_count=len(sel.bank) - 1,
        selection_uses_evaluation_labels=False,
    )
    (folder / "COMPLETE.json").write_text(json.dumps(summary, indent=2))
    return f"done {backbone}/{target}/{name}"


def contexts(args, backbone, target):
    X, y, ids, split = load_features(args.features, backbone, target)
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "val")
    folds = StratifiedKFold(args.outer, shuffle=True, random_state=args.seed)
    for k, (a, b) in enumerate(folds.split(X[train], y[train])):
        yield backbone, target, f"outer_{k}", train[a], train[b]
    if args.validation:
        yield backbone, target, "historical_validation", train, val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="all", choices=["all", *BACKBONES])
    ap.add_argument("--target", default="all", choices=["all", *TARGETS])
    ap.add_argument("--features", type=Path, default=ROOT / "data/features")
    ap.add_argument("--output", type=Path, default=ROOT / "results/nested")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--selection-bootstrap", type=int, default=0, help="score-only resamples of the selection (0 to skip)")
    ap.add_argument("--no-validation", dest="validation", action="store_false", help="skip the train-to-validation context")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    backbones = BACKBONES if args.backbone == "all" else [args.backbone]
    targets = TARGETS if args.target == "all" else [args.target]
    jobs = [c for b in backbones for t in targets for c in contexts(args, b, t)]
    for line in Parallel(n_jobs=args.jobs, return_as="generator")(delayed(run_context)(args, *j) for j in jobs):
        print(line, flush=True)


if __name__ == "__main__":
    main()
