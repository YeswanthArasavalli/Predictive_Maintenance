"""Ridge regression wrapper for Phase 2 (Task A).

Wraps `sklearn.linear_model.Ridge` with a fixed solver choice and a
deterministic random_state where relevant. This is a thin, inspectable
wrapper that:

- Never touches the raw target (`raw_RUL`). The caller supplies y.
- Records the fitted coefficients and intercept for the report.
- Uses `alpha` from hyperparameters (default 1.0). No grid search here —
  Phase 2 section 8 says "no large hyperparameter search".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class RidgeRegressor:
    alpha: float = 1.0
    _model: Ridge | None = field(default=None, repr=False)
    feature_names: list[str] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self._model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RidgeRegressor.fit() must be called before predict().")
        return self._model.predict(X)

    def describe(self) -> dict[str, Any]:
        if self._model is None:
            return {}
        return {
            "alpha": float(self.alpha),
            "n_features": int(self._model.coef_.shape[0]),
            "coef_mean_abs": float(np.mean(np.abs(self._model.coef_))),
            "intercept": float(self._model.intercept_),
        }
