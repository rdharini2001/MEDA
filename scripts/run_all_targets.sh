#!/usr/bin/env bash
set -euo pipefail

FEATURE_ROOT="${1:-features}"
OUT_ROOT="${2:-runs/megcal}"

targets=(
  adrenal_hyperplasia
  ascites
  atherosclerosis
  cholecystitis
  colorectal_cancer
  covid19
  fatty_liver
  gallstone
  hydronephrosis
  kidney_stone
  liver_calcifications
  liver_cyst
  liver_lesion
  lung_nodule_malignancy
  lymphadenopathy
  renal_cyst
  splenomegaly
)

for t in "${targets[@]}"; do
  echo "==== ${t} ===="
  python scripts/select_target_model.py \
    --target "${t}" \
    --features-csv "${FEATURE_ROOT}/${t}.csv" \
    --out-dir "${OUT_ROOT}/${t}"
done
