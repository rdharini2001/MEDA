"""Threshold tables, prevalence sensitivity, selection stability and head-comparison tables.

Everything is computed from files already in results/ and data/, so it runs in seconds.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meda.bank import family, lite_names
from meda.data import BACKBONES, TARGETS
from meda.selection import CONTROL, rank_candidates

LITE = {"merlin": 10, "ctnexus": 16}
PREVALENCE = (0.01, 0.05, 0.10, 0.25, 0.50)
RULES = (
    "LR_unweighted_05",
    "LR_prior_corrected",
    "LR_balanced_tuned",
    "fixed_logreg_l1_C0.1_tuned",
    "fixed_extratrees_balanced_tuned",
    "MEDA",
)
FIXED = [
    "fixed_pca8_logreg_tuned",
    "fixed_logreg_l1_C0.1_tuned",
    "fixed_extratrees_balanced_tuned",
    "fixed_rf_balanced_tuned",
    "fixed_histgb_tuned",
    "fixed_xgb_d2_lr0.03_n350_tuned",
]


def thresholds(nested):
    """Every selected classifier and threshold, for MEDA-Full and MEDA-Lite."""
    rows = []
    for b in BACKBONES:
        for t in TARGETS:
            for context in [f"outer_{k}" for k in range(5)] + ["historical_validation"]:
                table = pd.read_csv(nested / b / t / context / "inner_candidates.csv")
                table = table[table.name != CONTROL]
                full = rank_candidates(table).iloc[0]
                lite = rank_candidates(table[table.name.isin(lite_names(table.name, LITE[b]))]).iloc[0]
                for variant, row in (("MEDA-Full", full), ("MEDA-Lite", lite)):
                    rows.append(
                        dict(
                            backbone=b,
                            finding=t,
                            context=context,
                            variant=variant,
                            classifier=row["name"],
                            family=family(row["name"]),
                            threshold=float(row["threshold"]),
                            inner_balacc=float(row.inner_balacc),
                        )
                    )
    return pd.DataFrame(rows)


def prevalence_table(fold_metrics):
    rows = []
    for b, met in fold_metrics.items():
        for rule in RULES:
            m = met[met.method == rule]
            for pi in PREVALENCE:
                se, sp = m.sensitivity.to_numpy(), m.specificity.to_numpy()
                flagged = pi * se + (1 - pi) * (1 - sp)
                rows.append(
                    dict(
                        backbone=b,
                        rule=rule,
                        prevalence=pi,
                        accuracy=float(np.mean(pi * se + (1 - pi) * sp)),
                        ppv=float(np.mean(np.where(flagged > 0, pi * se / np.where(flagged > 0, flagged, 1), np.nan))),
                        false_alarms_per_100=float(100 * np.mean((1 - pi) * (1 - sp))),
                        missed_per_100=float(100 * np.mean(pi * (1 - se))),
                        balanced_accuracy=float(np.mean(0.5 * (se + sp))),
                    )
                )
    return pd.DataFrame(rows)


def crossover(fold_metrics):
    rows = []
    for b, met in fold_metrics.items():
        macro = met.groupby("method")[["sensitivity", "specificity"]].mean()
        for ref in ("LR_unweighted_05", "LR_prior_corrected"):
            se_m, sp_m = macro.loc["MEDA"]
            se_r, sp_r = macro.loc[ref]
            denom = (se_m - se_r) + (sp_r - sp_m)
            rows.append(dict(backbone=b, reference=ref, meda_sens=se_m, meda_spec=sp_m, ref_sens=se_r, ref_spec=sp_r,
                             prevalence_where_accuracy_equal=float((sp_r - sp_m) / denom) if denom else np.nan))
    return pd.DataFrame(rows)


def threshold_vs_prevalence(nested, audit):
    train = audit[(audit.split == "train") & (audit.backbone == "merlin")].set_index("target")
    prevalence = train.positives / train.n
    rows = []
    for b in BACKBONES:
        for t in TARGETS:
            taus = [
                json.loads((nested / b / t / f"outer_{k}" / "COMPLETE.json").read_text())["threshold"] for k in range(5)
            ]
            rows.append(dict(backbone=b, finding=t, training_prevalence=prevalence.get(t, np.nan), median_threshold=np.median(taus)))
    out = pd.DataFrame(rows)
    corr = [
        dict(backbone=b, spearman=spearmanr(g.training_prevalence, g.median_threshold)[0], n=len(g.dropna()))
        for b, g in out.dropna().groupby("backbone")
    ]
    return out, pd.DataFrame(corr)


def bootstrap_stability(nested, summary):
    counts = pd.read_csv(summary / "conditional_bootstrap_family_counts.csv")
    quant = pd.read_csv(summary / "conditional_bootstrap_threshold_quantiles.csv")
    rows = []
    for (b, t, k), g in counts.groupby(["backbone", "target", "outer_fold"]):
        chosen = json.loads((nested / b / t / f"outer_{k}" / "COMPLETE.json").read_text())["selected"]
        fam = family(chosen)
        total = g["count"].sum()
        rows.append(
            dict(
                backbone=b,
                finding=t,
                fold=k,
                selected_family=fam,
                same_family_fraction=float(g.loc[g.family == fam, "count"].sum() / total),
                modal_family_fraction=float(g["count"].max() / total),
                modal_is_selected=bool(g.loc[g["count"].idxmax(), "family"] == fam),
            )
        )
    out = pd.DataFrame(rows)
    out = out.merge(
        quant.rename(columns={"target": "finding", "outer_fold": "fold", "0.025": "tau_low", "0.5": "tau_median", "0.975": "tau_high"}),
        on=["backbone", "finding", "fold"],
    )
    out["tau_range"] = out.tau_high - out.tau_low
    return out


def inner_margin(nested):
    """Inner balanced accuracy of the chosen candidate over the best candidate of another family."""
    rows = []
    for b in BACKBONES:
        for t in TARGETS:
            for k in range(5):
                table = pd.read_csv(nested / b / t / f"outer_{k}" / "inner_candidates.csv")
                table = table[table.name != CONTROL]
                best = rank_candidates(table).iloc[0]
                other = table[table.family != best["family"]].inner_balacc.max()
                within = table[table.inner_balacc >= best.inner_balacc - 0.01]
                rows.append(
                    dict(
                        backbone=b,
                        finding=t,
                        fold=k,
                        margin_over_other_family=float(best.inner_balacc - other),
                        candidates_within_0p01=int(len(within)),
                        families_within_0p01=int(within.family.nunique()),
                    )
                )
    return pd.DataFrame(rows)


def head_comparison(summary):
    rows = []
    for b in BACKBONES:
        t = pd.read_csv(summary / f"{b}_nested_targets.csv")
        wide = t.pivot(index="target", columns="method", values="balacc")
        for head in FIXED:
            diff = wide["MEDA"] - wide[head]
            rows.append(
                dict(
                    backbone=b,
                    fixed_head=head,
                    findings_meda_higher=int((diff > 0).sum()),
                    findings_tied=int((diff == 0).sum()),
                    findings_meda_lower=int((diff < 0).sum()),
                    mean_difference=float(diff.mean()),
                    largest_meda_gain=float(diff.max()),
                    largest_meda_loss=float(diff.min()),
                )
            )
        best = wide[FIXED].max(axis=1)
        rows.append(
            dict(
                backbone=b,
                fixed_head="best fixed head per finding (chosen after the fact)",
                findings_meda_higher=int((wide["MEDA"] > best).sum()),
                findings_tied=int((wide["MEDA"] == best).sum()),
                findings_meda_lower=int((wide["MEDA"] < best).sum()),
                mean_difference=float((wide["MEDA"] - best).mean()),
                largest_meda_gain=float((wide["MEDA"] - best).max()),
                largest_meda_loss=float((wide["MEDA"] - best).min()),
            )
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "results/reviewer")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nested, summary = ROOT / "results/nested", ROOT / "results/summary"

    th = thresholds(nested)
    th.to_csv(args.output / "thresholds_all.csv", index=False)
    for (b, v), g in th.groupby(["backbone", "variant"]):
        g.pivot(index="finding", columns="context", values="threshold").to_csv(args.output / f"thresholds_wide_{b}_{v}.csv")

    folds = {b: pd.read_csv(summary / f"{b}_nested_fold_metrics.csv") for b in BACKBONES}
    prevalence_table(folds).to_csv(args.output / "prevalence_sensitivity.csv", index=False)
    crossover(folds).to_csv(args.output / "prevalence_crossover.csv", index=False)
    tp, corr = threshold_vs_prevalence(nested, pd.read_csv(ROOT / "data/cohort_audit.csv"))
    tp.to_csv(args.output / "threshold_vs_prevalence.csv", index=False)
    corr.to_csv(args.output / "threshold_vs_prevalence_correlation.csv", index=False)
    bootstrap_stability(nested, summary).to_csv(args.output / "selection_bootstrap_stability.csv", index=False)
    inner_margin(nested).to_csv(args.output / "inner_margin.csv", index=False)
    head_comparison(summary).to_csv(args.output / "meda_vs_fixed_heads_by_finding.csv", index=False)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
