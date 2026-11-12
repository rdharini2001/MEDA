"""Mean balanced accuracy and AUROC per cohort for the seven challenge systems."""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {
    "dharini24_vl_eao": "MEDA (Merlin)",
    "merlin_lp": "Merlin + LP",
    "medical_image_insights_vl_eao": "Merlin + EAO",
    "ctnexus_lp": "CT-NEXUS + LP",
    "gooducas_vo_eao": "CT-NEXUS + EAO",
    "voco_lp": "VoCo-L + LP",
    "fomofo_vo_eao": "VoCo-L + EAO",
}
COHORTS = ["Internal-Abd-China", "External-Abd-China", "External-Chest-China", "External-WB-Germany"]


def build(table):
    table = table[table.team.isin(SYSTEMS)].copy()
    table["system"] = table.team.map(SYSTEMS)
    out = {}
    for metric in ("test_balanced_acc", "test_auroc"):
        by_cohort = table.pivot_table(index="system", columns="dataset", values=metric, aggfunc="mean")[COHORTS]
        by_cohort["All cohorts"] = table.groupby("system")[metric].mean()
        out[metric] = by_cohort.loc[list(SYSTEMS.values())].round(4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "results/external_test/task2_per_target.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "results/external_test")
    args = ap.parse_args()
    tables = build(pd.read_csv(args.input))
    for metric, name in (("test_balanced_acc", "balanced_accuracy"), ("test_auroc", "auroc")):
        tables[metric].to_csv(args.output / f"cohort_means_{name}.csv")
        print(f"\n{name}\n{tables[metric].to_string()}")


if __name__ == "__main__":
    main()
