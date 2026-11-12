# Candidate bank

Every candidate receives the same standardized embedding. Stochastic components use seed 42.

| Family | Count | Settings varied | Fixed settings |
|---|---:|---|---|
| Balanced logistic | 6 | C in {0.05, 0.1, 0.25, 0.5, 1, 2} | L2, class-balanced weights, liblinear, 5000 iterations |
| Sparse logistic | 4 | C in {0.05, 0.1, 0.25, 0.5} | L1, otherwise as above |
| PCA + logistic | 6 | rank in {8, 16, 32, 64, 128, 256} | standardize, PCA, balanced L2 logistic with C = 0.5; used only if n > rank + 5 and d > rank |
| ExtraTrees | 1 | | 700 trees, sqrt(d) features per split, minimum leaf 2, balanced weights |
| Random forest | 1 | | 500 trees, sqrt(d) features, minimum leaf 2, balanced weights |
| Histogram boosting | 1 | | 250 iterations, learning rate 0.03, L2 0.1 |
| XGBoost | 4 | (depth, learning rate, trees, lambda) in {(2, 0.03, 350, 2), (3, 0.025, 450, 3), (2, 0.05, 250, 1), (4, 0.02, 500, 5)} | row subsample 0.85, column subsample 0.75, alpha 0.05, histogram trees, logistic loss, no class weighting |

The 23 candidates are listed in this order: balanced logistic with C = 0.5, the other five balanced settings, the four sparse settings, the six PCA ranks, ExtraTrees, random forest, histogram boosting and the four XGBoost models. See `meda.bank.search_order`.

Restricted banks take the first K candidates in that order. MEDA-Lite with 10 candidates is the six balanced and four sparse logistic settings. With 16 candidates it adds the six PCA ranks. Neither contains tree models, so the restricted banks differ from the full bank in model family as well as in size. The one exception is COVID-19, where the rank-256 PCA model is infeasible and the sixteenth slot is ExtraTrees.

The full bank has 23 candidates, or 22 for COVID-19 where the rank-256 PCA model is infeasible.
