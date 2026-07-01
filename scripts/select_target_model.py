#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from megcal.learners import build_candidate_bank
from megcal.metrics import safe_auroc
from megcal.thresholds import select_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Select MEGCAL learner and threshold for one target."
    )
    parser.add_argument("--features-csv", required=True, help="CSV with split,label,case_id and f_* feature columns.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    if not feature_cols:
        raise ValueError("Expected feature columns named f_0, f_1, ...")

    if "split" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain at least split, label, and f_* columns.")

    train = df[df["split"].str.lower().eq("train")].copy()
    val = df[df["split"].str.lower().isin(["val", "valid", "validation"])].copy()

    if train.empty or val.empty:
        raise ValueError("Both train and validation rows are required.")

    Xtr = train[feature_cols].to_numpy()
    ytr = train["label"].to_numpy().astype(int)
    Xva = val[feature_cols].to_numpy()
    yva = val["label"].to_numpy().astype(int)

    max_pca_rank = min(256, max(2, min(Xtr.shape[0] - 1, Xtr.shape[1])))
    candidates = build_candidate_bank(max_pca_rank=max_pca_rank)

    rows = []
    best_row = None
    best_model = None

    for cand in candidates:
        try:
            model = cand.model
            model.fit(Xtr, ytr)

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(Xva)[:, 1]
            else:
                scores = model.decision_function(Xva)
                scores = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-8)

            tau, ba = select_threshold(yva, scores)
            ba_05 = balanced_accuracy_score(yva, (scores >= 0.5).astype(int))

            row = {
                "target": args.target,
                "name": cand.name,
                "family": cand.family,
                "n_train": int(len(ytr)),
                "n_val": int(len(yva)),
                "pos_train": int(ytr.sum()),
                "pos_val": int(yva.sum()),
                "best_threshold": float(tau),
                "best_balacc": float(ba),
                "balacc_0p5": float(ba_05),
                "auroc": safe_auroc(yva, scores),
            }
            rows.append(row)

            if best_row is None or row["best_balacc"] > best_row["best_balacc"]:
                best_row = row
                best_model = model

        except Exception as exc:
            rows.append({
                "target": args.target,
                "name": cand.name,
                "family": cand.family,
                "error": repr(exc),
            })

    pd.DataFrame(rows).to_csv(out_dir / "val_report.csv", index=False)

    if best_model is None:
        raise RuntimeError("No candidate model fit successfully.")

    joblib.dump(best_model, out_dir / "best_model.joblib")
    with open(out_dir / "best_model.json", "w") as f:
        json.dump(best_row, f, indent=2)

    print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
