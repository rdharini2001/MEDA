from collections import Counter

from meda.bank import build_bank, family, lite_names, search_order


def test_full_bank_has_23_candidates_in_documented_order():
    names = [n for n, _ in build_bank(500, 2048)]
    assert len(names) == 23
    assert names[:6] == [f"logreg_C{c}" for c in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)]
    assert names[6:10] == [f"logreg_l1_C{c}" for c in (0.05, 0.1, 0.25, 0.5)]
    assert names[10:16] == [f"pca{r}_logreg" for r in (8, 16, 32, 64, 128, 256)]
    assert names[16:19] == ["extratrees_balanced", "rf_balanced", "histgb"]
    assert Counter(family(n) for n in names) == {
        "Balanced logistic": 6,
        "Sparse logistic": 4,
        "PCA-logistic": 6,
        "ExtraTrees": 1,
        "Random forest": 1,
        "HistGradientBoosting": 1,
        "XGBoost": 4,
    }


def test_pca_ranks_are_dropped_when_infeasible():
    names = [n for n, _ in build_bank(100, 1120)]
    assert "pca64_logreg" in names and "pca128_logreg" not in names
    assert len(names) == 21


def test_search_order_puts_balanced_c05_first():
    names = [n for n, _ in build_bank(500, 2048)]
    order = search_order(names)
    assert order[0] == "logreg_C0.5" and sorted(order) == sorted(names)


def test_lite_banks_contain_only_linear_candidates():
    names = [n for n, _ in build_bank(500, 2048)]
    assert {family(n) for n in lite_names(names, 10)} == {"Balanced logistic", "Sparse logistic"}
    assert {family(n) for n in lite_names(names, 16)} == {"Balanced logistic", "Sparse logistic", "PCA-logistic"}
