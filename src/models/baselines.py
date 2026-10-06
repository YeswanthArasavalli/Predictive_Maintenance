"""Trivial deterministic baselines required by Phase 2 sections 5 and 14.

NaiveMeanRUL
    Predicts a constant equal to the mean raw_RUL over the TRAIN rows.
    This is the "zero-signal" reference every model must beat.

NaiveAgeRUL
    Predicts (mean_train_lifespan - cycle), floored at 0. Requires the
    `cycle` column (justified in the report; the mean lifespan is a
    train-only constant, not target leakage).

MajorityClass
    Predicts the label that is most frequent in TRAIN. Used to expose how
    misleading accuracy alone can be on imbalanced failure data.

All three are deterministic: `fit()` and `predict()` never call any RNG,
and calling `fit()` twice on the same data yields identical state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.loader import CYCLE, UNIT_ID


@dataclass
class NaiveMeanRUL:
    """Constant predictor equal to the training-set mean raw_RUL."""

    mean_rul: float | None = None

    def fit(self, df_train: pd.DataFrame, *, target_col: str = "raw_RUL") -> "NaiveMeanRUL":
        self.mean_rul = float(df_train[target_col].mean())
        return self

    def predict(self, df_val: pd.DataFrame) -> np.ndarray:
        if self.mean_rul is None:
            raise RuntimeError("NaiveMeanRUL.fit() must be called before predict().")
        return np.full(len(df_val), self.mean_rul, dtype=np.float64)

    def describe(self) -> dict[str, float]:
        return {"mean_rul": float(self.mean_rul)} if self.mean_rul is not None else {}


@dataclass
class NaiveAgeRUL:
    """Age-based predictor: RUL_hat = mean_train_lifespan - cycle, clipped at 0.

    `mean_train_lifespan` is the mean of max_cycle(unit_id) over TRAIN
    engines. It is a train-partition constant, NOT a target feature.

    Justification for using `cycle`: this baseline is inherently a
    "lifecycle position" heuristic, so the current age (cycle) is its only
    input. Phase 2 section 5 explicitly permits "a simple age-based estimate".
    """

    mean_lifespan: float | None = None

    def fit(self, df_train: pd.DataFrame) -> "NaiveAgeRUL":
        max_cycles = df_train.groupby(UNIT_ID)[CYCLE].max()
        self.mean_lifespan = float(max_cycles.mean())
        return self

    def predict(self, df_val: pd.DataFrame) -> np.ndarray:
        if self.mean_lifespan is None:
            raise RuntimeError("NaiveAgeRUL.fit() must be called before predict().")
        cycles = df_val[CYCLE].to_numpy(dtype=np.float64)
        return np.maximum(self.mean_lifespan - cycles, 0.0)

    def describe(self) -> dict[str, float]:
        return {"mean_lifespan": float(self.mean_lifespan)} if self.mean_lifespan is not None else {}


@dataclass
class MajorityClass:
    """Predict the majority label observed in training.

    The Phase 2 spec requires reporting which metrics are degenerate for
    this baseline. `describe()` returns the observed prior; the evaluation
    layer will mark PR-AUC/ROC-AUC as `None` because this classifier emits
    hard labels only.
    """

    majority_label: int | None = None
    positive_rate: float | None = None

    def fit(self, y_train: np.ndarray) -> "MajorityClass":
        y = np.asarray(y_train).astype(np.int8)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos == 0 and neg == 0:
            raise ValueError("MajorityClass.fit() received an empty label array.")
        # Tie-break to 0 (the non-failure majority is the safer default
        # in an imbalance scenario; ties never occur for realistic H).
        self.majority_label = 1 if pos > neg else 0
        self.positive_rate = pos / (pos + neg)
        return self

    def predict(self, n_rows: int) -> np.ndarray:
        if self.majority_label is None:
            raise RuntimeError("MajorityClass.fit() must be called before predict().")
        return np.full(int(n_rows), self.majority_label, dtype=np.int8)

    def describe(self) -> dict[str, float | int]:
        return {
            "majority_label": int(self.majority_label) if self.majority_label is not None else -1,
            "positive_rate": float(self.positive_rate) if self.positive_rate is not None else -1.0,
        }
