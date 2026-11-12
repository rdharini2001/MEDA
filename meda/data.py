"""Loading of the packaged feature matrices."""
from pathlib import Path

import numpy as np

TARGETS = (
    "adrenal_hyperplasia",
    "ascites",
    "atherosclerosis",
    "cholecystitis",
    "colorectal_cancer",
    "covid",
    "fatty_liver",
    "gallstone",
    "hydronephrosis",
    "kidney_stone",
    "liver_calcifications",
    "liver_cyst",
    "liver_lesion",
    "lung_nodule_malignancy",
    "lymphadenopathy",
    "renal_cyst",
    "splenomegaly",
)
BACKBONES = ("merlin", "ctnexus")
FEATURE_DIM = {"merlin": 2048, "ctnexus": 1120}


def load_features(root, backbone, target):
    """Return ``X, y, case_id, split`` for one finding."""
    path = Path(root) / backbone / f"{target}.npz"
    with np.load(path) as a:
        X, y, ids, split = (a[k] for k in ("X", "y", "case_id", "split"))
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate scan identifiers in {path}")
    return X, y, ids, split
