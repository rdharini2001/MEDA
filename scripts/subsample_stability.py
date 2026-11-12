"""How stable is the selected classifier when the training cases change?

Each outer training split is subsampled (80% of cases, stratified, without
replacement) and the whole selection is repeated, including the inner
cross-validation. Subsampling avoids duplicated scans leaking between inner
folds. The default searches the MEDA-Lite bank; the full bank takes much longer.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meda.bank import LITE_SIZE
from meda.data import BACKBONES, TARGETS, load_features
from meda.selection import select_candidate


def run_context(args, backbone, target, fold):
    out = args.output / f"{backbone}_{target}_outer{fold}.csv"
    if out.exists():
        return str(out)
    X, y, ids, split = load_features(args.features, backbone, target)
    train = np.flatnonzero(split == "train")
    a, _ = list(StratifiedKFold(args.outer, shuffle=True, random_state=args.seed).split(X[train], y[train]))[fold]
    tr = train[a]
    budget = None if args.full else LITE_SIZE[backbone]
    rows = []
    with threadpool_limits(limits=1):
        ref = select_candidate(X[tr], y[tr], ids[tr], seed=args.seed, inner=args.inner, budget=budget).chosen
        splitter = StratifiedShuffleSplit(args.resamples, train_size=args.fraction, random_state=args.seed + fold)
        for b, (keep, _) in enumerate(splitter.split(X[tr], y[tr])):
            sub = tr[keep]
            chosen = select_candidate(X[sub], y[sub], ids[sub], seed=args.seed, inner=args.inner, budget=budget).chosen
            rows.append(
                dict(
                    backbone=backbone,
                    target=target,
                    fold=fold,
                    resample=b,
                    reference_name=ref["name"],
                    reference_family=ref["family"],
                    reference_threshold=float(ref["threshold"]),
                    name=chosen["name"],
                    family=chosen["family"],
                    threshold=float(chosen["threshold"]),
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="all", choices=["all", *BACKBONES])
    ap.add_argument("--target", default="all", choices=["all", *TARGETS])
    ap.add_argument("--features", type=Path, default=ROOT / "data/features")
    ap.add_argument("--output", type=Path, default=ROOT / "results/reviewer/subsample")
    ap.add_argument("--resamples", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=0.8)
    ap.add_argument("--full", action="store_true", help="search all 23 candidates instead of MEDA-Lite")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    backbones = BACKBONES if args.backbone == "all" else [args.backbone]
    targets = TARGETS if args.target == "all" else [args.target]
    jobs = [(b, t, k) for b in backbones for t in targets for k in range(args.outer)]
    for line in Parallel(n_jobs=args.jobs, return_as="generator")(delayed(run_context)(args, *j) for j in jobs):
        print("wrote", line, flush=True)


if __name__ == "__main__":
    main()
