"""Logistic Regression wrapper for Phase 2 (Task B).

Uses `sklearn.linear_model.LogisticRegression` with `class_weight='balanced'`
by default (Phase 2 section 16: prefer class weighting over synthetic
oversampling). Solver is fixed to 'lbfgs' — deterministic, no grid search.
`max_iter=1000` is a modest default so convergence is not silently clipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class LogisticRegressionWrapper:
    C: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 1000
    random_state: int = 42
    _model: LogisticRegression | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionWrapper":
        self._model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def positive_class_probability(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LogisticRegressionWrapper.fit() must be called before predict.")
        pos_idx = int(np.searchsorted(self._model.classes_, 1))
        return self._model.predict_proba(X)[:, pos_idx]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.positive_class_probability(X) >= threshold).astype(np.int8)

    def describe(self) -> dict[str, Any]:
        if self._model is None:
            return {}
        return {
            "C": float(self.C),
            "class_weight": self.class_weight,
            "max_iter": int(self.max_iter),
            "n_features": int(self._model.n_features_in_),
            "random_state": int(self.random_state),
            "classes_seen": [int(c) for c in self._model.classes_],
        }
