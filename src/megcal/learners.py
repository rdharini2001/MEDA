from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Candidate:
    name: str
    family: str
    model: Any


def build_candidate_bank(random_state: int = 7, max_pca_rank: int = 256):
    """Build candidate scoring functions for target-wise decision adaptation."""
    candidates = []

    for c in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        candidates.append(
            Candidate(
                name=f"balanced_logreg_C{c}",
                family="Balanced logistic",
                model=Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                    )),
                ]),
            )
        )

    for c in [0.05, 0.1, 0.25, 0.5]:
        candidates.append(
            Candidate(
                name=f"sparse_logreg_C{c}",
                family="Sparse logistic",
                model=Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        penalty="l1",
                        solver="liblinear",
                        max_iter=5000,
                    )),
                ]),
            )
        )

    for r in [8, 16, 32, 64, 128, 256]:
        if r <= max_pca_rank:
            candidates.append(
                Candidate(
                    name=f"pca{r}_logreg",
                    family="PCA-logistic",
                    model=Pipeline([
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=r, random_state=random_state)),
                        ("clf", LogisticRegression(
                            class_weight="balanced",
                            max_iter=5000,
                            solver="lbfgs",
                        )),
                    ]),
                )
            )

    candidates.extend([
        Candidate(
            name="extratrees_balanced",
            family="ExtraTrees",
            model=ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        Candidate(
            name="rf_balanced",
            family="Random forest",
            model=RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        Candidate(
            name="hist_gradient_boosting",
            family="HistGradientBoosting",
            model=HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.03,
                random_state=random_state,
            ),
        ),
    ])

    try:
        from xgboost import XGBClassifier

        xgb_specs = [
            ("xgb_d2_lr0.03_n350", 2, 0.03, 350),
            ("xgb_d3_lr0.025_n450", 3, 0.025, 450),
            ("xgb_d2_lr0.05_n250", 2, 0.05, 250),
            ("xgb_d4_lr0.02_n500", 4, 0.02, 500),
        ]

        for name, depth, lr, n_estimators in xgb_specs:
            candidates.append(
                Candidate(
                    name=name,
                    family="XGBoost",
                    model=XGBClassifier(
                        max_depth=depth,
                        learning_rate=lr,
                        n_estimators=n_estimators,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=4,
                    ),
                )
            )
    except Exception:
        pass

    return candidates
