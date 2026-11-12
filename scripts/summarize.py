"""Macro metrics, intervals and paired tests from the per-fold results.

Reads results/nested and writes one macro table, one paired-test table and one
per-finding table per backbone and protocol under results/summary.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meda.data import BACKBONES, TARGETS
from meda.stats import uncertainty

STANDALONE_COHORTS = ("covid", "lung_nodule_malignancy")


def collect(nested, backbone, protocol):
    contexts = [f"outer_{k}" for k in range(5)] if protocol == "nested" else ["historical_validation"]
    preds, metrics = [], []
    for target in TARGETS:
        for fold, context in enumerate(contexts):
            folder = nested / backbone / target / context
            if not (folder / "COMPLETE.json").exists():
                raise FileNotFoundError(f"missing {folder}; run scripts/run_nested.py first")
            p = pd.read_csv(folder / "predictions.csv", float_precision="round_trip").assign(target=target, fold=fold)
            cohort = target if target in STANDALONE_COHORTS else "amos"
            p["cluster"] = cohort + ":" + p.case_id.astype(str)
            preds.append(p)
            metrics.append(pd.read_csv(folder / "metrics.csv").assign(target=target, fold=fold))
    return pd.concat(preds, ignore_index=True), pd.concat(metrics, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nested", type=Path, default=ROOT / "results/nested")
    ap.add_argument("--output", type=Path, default=ROOT / "results/summary")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for backbone in BACKBONES:
        for protocol in ("nested", "historical_validation"):
            pred, met = collect(args.nested, backbone, protocol)
            if pred.duplicated(["target", "case_id", "method"]).any():
                raise ValueError("a case has more than one score for a method")
            stem = f"{backbone}_{protocol}"
            met.to_csv(args.output / f"{stem}_fold_metrics.csv", index=False)
            summary, paired = uncertainty(pred, met, args.bootstrap)
            summary.to_csv(args.output / f"{stem}_macro.csv", index=False)
            paired.to_csv(args.output / f"{stem}_paired.csv", index=False)
            per_target = (
                met.groupby(["target", "method"])
                .agg(
                    balacc=("balacc", "mean"),
                    balacc_sd=("balacc", "std"),
                    auroc=("auroc", "mean"),
                    auroc_sd=("auroc", "std"),
                    threshold_median=("threshold", "median"),
                    threshold_min=("threshold", "min"),
                    threshold_max=("threshold", "max"),
                )
                .reset_index()
            )
            per_target.to_csv(args.output / f"{stem}_targets.csv", index=False)
            print("wrote", stem, flush=True)


if __name__ == "__main__":
    main()
