"""Clustered conditional bootstrap, paired differences and Holm adjustment.

Resampling weights are attached to scans, so a scan shared by several findings
(AMOS) moves together. Intervals condition on the fitted models and splits.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

BOOTSTRAP_SEED = 20261031


def auc_influence(y, s):
    """AUROC and the per-case influence of each case on it (with ties)."""
    pos, neg = s[y == 1], s[y == 0]
    auc = roc_auc_score(y, s)
    sn, sp = np.sort(neg), np.sort(pos)
    vp = (np.searchsorted(sn, pos, "left") + np.searchsorted(sn, pos, "right")) / (2 * len(neg))
    vn = 1 - (np.searchsorted(sp, neg, "left") + np.searchsorted(sp, neg, "right")) / (2 * len(pos))
    out = np.zeros(len(y))
    out[y == 1] = (vp - auc) / len(pos)
    out[y == 0] = (vn - auc) / len(neg)
    return auc, out


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p[i] * (len(p) - rank)))
        adjusted[i] = running
    return adjusted


def uncertainty(pred, metrics, n_boot=2000, reference="MEDA", run_tests=True, permutations=10000):
    """Macro balanced accuracy and AUROC with intervals, plus paired tests against ``reference``.

    ``pred`` needs columns method, target, fold, label, score, prediction and
    cluster; ``metrics`` needs method and balacc.
    """
    methods = sorted(pred.method.unique())
    groups = sorted(set(zip(pred.target, pred.fold, pred.label)))
    scans = sorted(pred.cluster.unique())
    scan_ix = {c: i for i, c in enumerate(scans)}
    group_ix = {c: i for i, c in enumerate(groups)}
    ref = pred[pred.method == methods[0]]
    rows = np.array([scan_ix[c] for c in ref.cluster])
    cols = np.array([group_ix[c] for c in zip(ref.target, ref.fold, ref.label)])
    A = csr_matrix((np.ones(len(ref)), (rows, cols)), shape=(len(scans), len(groups)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    W = rng.exponential(size=(n_boot, len(scans))).astype(np.float32)
    denom = np.asarray((A.T @ W.T).T)
    summary, boot, coef = [], {}, {}
    for method in methods:
        m = pred[pred.method == method]
        r = np.array([scan_ix[c] for c in m.cluster])
        c = np.array([group_ix[z] for z in zip(m.target, m.fold, m.label)])
        correct = (m.prediction.to_numpy() == m.label.to_numpy()).astype(float)
        C = csr_matrix((correct, (r, c)), shape=A.shape)
        draws = (np.asarray((C.T @ W.T).T) / denom).mean(axis=1)
        influence, aucs = np.zeros(len(scans)), []
        for _, sub in m.groupby(["target", "fold"]):
            auc, infl = auc_influence(sub.label.to_numpy(), sub.score.to_numpy())
            aucs.append(auc)
            for case, v in zip(sub.cluster, infl):
                influence[scan_ix[case]] += v / (len(groups) / 2)
        auc_base = float(np.mean(aucs))
        auc_draws = auc_base + (W - 1) @ influence
        counts = np.asarray(A.sum(axis=0)).ravel()
        coef[method] = np.bincount(r, weights=correct / counts[c] / len(groups), minlength=len(scans))
        boot[method] = draws
        lo, hi = np.quantile(draws, [0.025, 0.975])
        alo, ahi = np.quantile(auc_draws, [0.025, 0.975])
        summary.append(
            dict(
                method=method,
                balacc=float(metrics[metrics.method == method].balacc.mean()),
                balacc_low=lo,
                balacc_high=hi,
                auroc=auc_base,
                auroc_low=max(0.0, alo),
                auroc_high=min(1.0, ahi),
            )
        )
    pairs = []
    if run_tests:
        for other in [m for m in methods if m != reference]:
            diff = boot[reference] - boot[other]
            lo, hi = np.quantile(diff, [0.025, 0.975])
            v = coef[reference] - coef[other]
            observed = float(v.sum())
            hits = 0
            for start in range(0, permutations, 500):
                signs = rng.choice([-1, 1], size=(min(500, permutations - start), len(scans)))
                hits += int((np.abs(signs @ v) >= abs(observed) - 1e-12).sum())
            pairs.append(dict(comparison=f"{reference} - {other}", delta=observed, low=lo, high=hi, p=(hits + 1) / (permutations + 1)))
        if pairs:
            adjusted = holm([p["p"] for p in pairs])
            for p, a in zip(pairs, adjusted):
                p["p_holm"] = a
    return pd.DataFrame(summary), pd.DataFrame(pairs)
