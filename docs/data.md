# Data

## Findings and cohorts

Seventeen findings: 15 abdominal findings from AMOS-based CT data, COVID-19 CT, and lung-nodule malignancy from LUNA25. The training sets contain 308 to 1,107 studies per finding, and positive prevalence ranges from about 6% to 50%.

Abdominal findings: adrenal hyperplasia, ascites, atherosclerosis, cholecystitis, colorectal cancer, fatty liver, gallstone, hydronephrosis, kidney stone, liver calcifications, liver cyst, liver lesion, lymphadenopathy, renal cyst, splenomegaly.

`data/cohort_details.csv` lists training and validation sizes and prevalence per finding. `data/cohort_audit.csv` lists the size, positives and feature dimension of every packaged feature file together with its SHA-256.

## Feature files

`data/features/<encoder>/<finding>.npz` holds four arrays:

| Key | Content |
|---|---|
| `X` | float32 embeddings, one row per study (2,048 columns for Merlin, 1,120 for CT-NEXUS) |
| `y` | 0/1 label |
| `case_id` | study identifier, unique within a file |
| `split` | `train` or `val`, the split provided by the challenge |

Both encoders contain the same studies for a finding. Load them with `meda.data.load_features`.

## Feature extraction

CT volumes are not distributed with this repository. The embeddings were produced as follows.

**Merlin.** The released Merlin image encoder (`Merlin(ImageEmbedding=True)`, I3D-ResNet-152) with checkpoint `i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt`. The feature is the 2,048-dimensional global-average-pooled output of the final residual stage, taken before the 512-dimensional contrastive and 1,692-dimensional phenotype heads. Each volume was processed once without test-time augmentation, using Merlin's standard transforms: reorientation to RAS, resampling to 1.5 x 1.5 x 3 mm, clipping to [-1000, 1000] HU, rescaling to [0, 1], then padding and center-cropping to 224 x 224 x 160 voxels. No organ or lesion masks were used.

**CT-NEXUS.** Features came from the organizer-provided container, whose residual-encoder network gives skip-connection features at six stages (32, 64, 128, 256, 320 and 320 channels). These are adaptively pooled and concatenated into 1,120 channels, then averaged over the spatial axes. Volumes were z-score normalized and resampled to 1 mm isotropic spacing. Atherosclerosis, colorectal cancer, ascites, lymphadenopathy, COVID-19 and lung-nodule malignancy used a 320^3 crop. For the eleven organ-specific abdominal findings (splenomegaly, adrenal hyperplasia, fatty liver, cholecystitis, liver calcifications, hydronephrosis, gallstone, liver lesion, kidney stone, liver cyst and renal cyst), the organizer's foreground organ mask was supplied to the extractor and a 160^3 crop was centered on the mask. Organ masks therefore entered CT-NEXUS extraction for those eleven findings but not the Merlin features. The CT-NEXUS checkpoint is the container's default `fold_all/checkpoint_final.pth`; its hash was not recorded.

## Independent challenge results

`results/external_test/task2_per_target.csv` has the per-finding balanced accuracy and AUROC of the seven challenge systems on four independent cohorts: Internal-Abd-China (15 findings), External-Abd-China (7), External-Chest-China (6) and External-WB-Germany (2). The system names in the file are team identifiers; `scripts/external_table.py` maps them to the labels used in the paper.
