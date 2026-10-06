"""HistGradientBoosting wrappers for Phase 2 (Tasks A and B).

Preferred tree implementation per Phase 2 section 8: sklearn's
`HistGradientBoostingRegressor` / `HistGradientBoostingClassifier`.
XGBoost / LightGBM are deliberately NOT used here (section 8: "Do NOT
immediately introduce XGBoost/LightGBM unless there is a strong reason").

Class-imbalance policy (Phase 2 section 16):
    `HistGradientBoostingClassifier` does not expose a `class_weight`
    parameter directly. Instead we compute inverse-frequency sample
    weights from the training labels and pass them to `fit(sample_weight=)`.
    This satisfies "prefer class weighting if supported / sample weighting"
    without inventing synthetic observations (no SMOTE). The user-facing
    YAML key `class_weight: balanced` triggers this behavior; setting
    `class_weight: null` fits with equal weights.

Determinism:
    `random_state` is fixed. Both wrappers use modest default
    `max_iter=200, learning_rate=0.1` unless the YAML overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _inverse_freq_sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights: w_i = N / (K * n_{y_i}) (sklearn's 'balanced' formula)."""
    y = np.asarray(y).astype(np.int8)
    N = len(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 0 or N == 0:
        raise ValueError("Cannot compute balanced sample weights on empty labels.")
    weight_per_class = {int(c): N / (len(classes) * int(cnt)) for c, cnt in zip(classes, counts)}
    return np.array([weight_per_class[int(v)] for v in y], dtype=np.float64)


@dataclass
class HistGBRegressorWrapper:
    max_iter: int = 200
    learning_rate: float = 0.1
    random_state: int = 42
    _model: HistGradientBoostingRegressor | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGBRegressorWrapper":
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HistGBRegressorWrapper.fit() must be called before predict().")
        return self._model.predict(X)

    def describe(self) -> dict[str, Any]:
        if self._model is None:
            return {}
        return {
            "max_iter": int(self.max_iter),
            "learning_rate": float(self.learning_rate),
            "n_features": int(self._model.n_features_in_),
            "random_state": int(self.random_state),
        }


@dataclass
class HistGBClassifierWrapper:
    max_iter: int = 200
    learning_rate: float = 0.1
    random_state: int = 42
    class_weight: str | None = "balanced"
    _model: HistGradientBoostingClassifier | None = field(default=None, repr=False)
    _sample_weight_applied: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGBClassifierWrapper":
        self._model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        if self.class_weight == "balanced":
            sw = _inverse_freq_sample_weights(y)
            self._model.fit(X, y, sample_weight=sw)
            self._sample_weight_applied = True
        elif self.class_weight is None:
            self._model.fit(X, y)
            self._sample_weight_applied = False
        else:
            raise ValueError(
                f"HistGBClassifierWrapper.class_weight must be 'balanced' or None; got {self.class_weight!r}"
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("HistGBClassifierWrapper.fit() must be called before predict_proba().")
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        # Positive class is class index for label 1; sklearn stores classes_ in sorted order.
        pos_idx = int(np.searchsorted(self._model.classes_, 1))
        return (proba[:, pos_idx] >= threshold).astype(np.int8)

    def positive_class_probability(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        pos_idx = int(np.searchsorted(self._model.classes_, 1))
        return proba[:, pos_idx]

    def describe(self) -> dict[str, Any]:
        if self._model is None:
            return {}
        return {
            "max_iter": int(self.max_iter),
            "learning_rate": float(self.learning_rate),
            "n_features": int(self._model.n_features_in_),
            "random_state": int(self.random_state),
            "class_weight": self.class_weight,
            "sample_weight_applied": bool(self._sample_weight_applied),
        }
