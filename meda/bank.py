"""Candidate classifier bank.

The bank has 23 settings in a fixed order. Restricted banks (MEDA-Lite) take
the first K candidates in that order.
"""
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

LOGREG_C = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
SPARSE_C = (0.05, 0.1, 0.25, 0.5)
PCA_RANKS = (8, 16, 32, 64, 128, 256)
XGB_SETTINGS = (
    (2, 0.03, 350, 2.0),
    (3, 0.025, 450, 3.0),
    (2, 0.05, 250, 1.0),
    (4, 0.02, 500, 5.0),
)
LITE_SIZE = {"merlin": 10, "ctnexus": 16}


def _logistic(C, seed, penalty="l2", class_weight="balanced"):
    return LogisticRegression(
        C=C,
        penalty=penalty,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )


def family(name):
    if name.startswith("logreg_l1"):
        return "Sparse logistic"
    if name.startswith("logreg"):
        return "Balanced logistic"
    if name.startswith("pca"):
        return "PCA-logistic"
    if name.startswith("xgb"):
        return "XGBoost"
    if name.startswith("extratrees"):
        return "ExtraTrees"
    if name.startswith("rf"):
        return "Random forest"
    if name == "histgb":
        return "HistGradientBoosting"
    return "Unweighted logistic"


def build_bank(n_samples, n_features, seed=42, n_jobs=1):
    """Return the ordered list of ``(name, estimator)`` pairs.

    ``n_samples`` is the smallest inner training size. A PCA rank is only
    included when ``n_samples > rank + 5`` and ``n_features > rank``.
    """
    from xgboost import XGBClassifier

    bank = []
    for C in LOGREG_C:
        bank.append((f"logreg_C{C}", make_pipeline(StandardScaler(), _logistic(C, seed))))
    for C in SPARSE_C:
        bank.append((f"logreg_l1_C{C}", make_pipeline(StandardScaler(), _logistic(C, seed, "l1"))))
    for rank in PCA_RANKS:
        if n_samples > rank + 5 and n_features > rank:
            bank.append(
                (
                    f"pca{rank}_logreg",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=rank, random_state=seed)),
                            ("clf", _logistic(0.5, seed)),
                        ]
                    ),
                )
            )
    bank.append(
        (
            "extratrees_balanced",
            ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            ),
        )
    )
    bank.append(
        (
            "rf_balanced",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
            ),
        )
    )
    bank.append(
        (
            "histgb",
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.03,
                l2_regularization=0.1,
                random_state=seed,
            ),
        )
    )
    for depth, lr, n_est, reg in XGB_SETTINGS:
        bank.append(
            (
                f"xgb_d{depth}_lr{lr}_n{n_est}",
                XGBClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=lr,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_lambda=reg,
                    reg_alpha=0.05,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            )
        )
    return bank


def unweighted_logistic(seed=42):
    """Standardized logistic regression without class weights (a control, not a candidate)."""
    return make_pipeline(StandardScaler(), _logistic(0.5, seed, class_weight=None))


def search_order(names):
    """Bank order used by restricted searches: balanced logistic C=0.5 first, then the rest."""
    names = list(names)
    first = "logreg_C0.5"
    return ([first] if first in names else []) + [n for n in names if n != first]


def lite_names(names, k):
    """Names of the first ``k`` candidates in search order."""
    return search_order(names)[:k]
