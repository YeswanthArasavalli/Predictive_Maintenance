"""Phase 2 model layer.

Baseline models only. This package enforces the Phase 2 hard rule:
NO LSTM, NO GRU, NO Transformer, NO deep neural networks.

The public surface is:
    features.build_row_matrix  — assemble X for row-level classical models
    baselines.NaiveMeanRUL, NaiveAgeRUL, MajorityClass
    linear_degradation.LinearDegradationRUL (causal per-engine OLS)
    ridge.RidgeRegressor
    gradient_boosting.HistGBRegressor, HistGBClassifier
"""
