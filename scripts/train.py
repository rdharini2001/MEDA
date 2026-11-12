"""Fit MEDA on one finding's training features and save the selected classifier and threshold.

Example:
    python scripts/train.py --backbone merlin --target fatty_liver --out models/fatty_liver_merlin.joblib
    python scripts/train.py --backbone ctnexus --target covid --lite --out models/covid_ctnexus.joblib
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meda import MEDA
from meda.bank import LITE_SIZE
from meda.data import BACKBONES, TARGETS, load_features
from meda.metrics import decision_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=BACKBONES)
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--features", type=Path, default=ROOT / "data/features")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lite", action="store_true", help="search the restricted MEDA-Lite bank")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    X, y, ids, split = load_features(args.features, args.backbone, args.target)
    train, val = split == "train", split == "val"
    budget = LITE_SIZE[args.backbone] if args.lite else None
    model = MEDA(budget=budget, seed=args.seed).fit(X[train], y[train])
    scores = model.decision_scores(X[val])
    report = decision_metrics(y[val], scores, model.threshold_)
    print(f"selected {model.name_} at threshold {model.threshold_:.4f}")
    print("validation:", {k: round(v, 4) if isinstance(v, float) else v for k, v in report.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        dict(model=model.model_, name=model.name_, threshold=model.threshold_, backbone=args.backbone, target=args.target),
        args.out,
    )


if __name__ == "__main__":
    main()
