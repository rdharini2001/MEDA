"""MEDA: finding-specific classifier and threshold selection on frozen CT embeddings."""
from .bank import build_bank, family, lite_names, search_order
from .model import MEDA
from .thresholds import threshold_grid, select_threshold
from .selection import select_candidate, SelectionResult
from .nested import run_nested, evaluate_fold

__all__ = [
    "build_bank",
    "lite_names",
    "search_order",
    "MEDA",
    "family",
    "threshold_grid",
    "select_threshold",
    "select_candidate",
    "SelectionResult",
    "run_nested",
    "evaluate_fold",
]
__version__ = "1.0.0"
