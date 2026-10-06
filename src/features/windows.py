"""Causal sliding-window generation for sequence models.

For an engine with ordered cycles 1, 2, ..., T and lookback W:
    X_t = [x_{t-W+1}, x_{t-W+2}, ..., x_t]   (W rows, same engine only)

Valid prediction timepoints: t >= W (the first full window).
Early cycles (t < W) are discarded; this warm-up requirement is documented.

Leakage controls:
- Windows NEVER span two engines (split by unit_id before windowing).
- No future observation (cycle > t) enters X_t.
- Target columns (raw_RUL, model_RUL_target, failure_risk_target_H*) are
  NEVER included in the feature tensor X.
- unit_id and cycle are NEVER included in X.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.contract import FORBIDDEN_FEATURE_COLS
from ..data.loader import CYCLE, UNIT_ID


@dataclass
class WindowMetadata:
    """Per-window provenance record."""

    unit_id: int
    target_cycle: int    # the "current" observation t (last in window)
    start_cycle: int     # t - W + 1
    raw_rul: int         # the regression target at time t
    failure_label: int   # the classification target at time t


def generate_windows(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    lookback: int = 30,
    target_col: str = "raw_RUL",
    failure_target_col: str = "failure_risk_target_H30",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[WindowMetadata]]:
    """Generate causal sliding windows within each engine.

    Parameters
    ----------
    df: pd.DataFrame
        A partition of engines (train OR validation) with a regime column,
        target columns, and the feature columns already scaled.
        Must contain unit_id and cycle columns.
    feature_cols: list[str]
        Columns to include in X. Must NOT contain any target or identifier.
    lookback: int
        Number of time-steps in each window (W).
    target_col: str
        Column name for the regression target (raw_RUL).
    failure_target_col: str
        Column name for the failure-risk binary target.

    Returns
    -------
    tuple:
        X         : np.ndarray shape (n_samples, W, n_features), float64
        y_rul     : np.ndarray shape (n_samples,), int64
        y_failure : np.ndarray shape (n_samples,), int8
        metadata  : list[WindowMetadata]

    Raises
    ------
    ValueError
        If feature_cols contains any forbidden column.
    """
    # Validate feature columns against the data contract
    violations = set(feature_cols) & set(FORBIDDEN_FEATURE_COLS)
    if violations:
        raise ValueError(
            f"generate_windows: feature_cols contain forbidden columns: {sorted(violations)}"
        )

    n_features = len(feature_cols)
    all_X: list[np.ndarray] = []
    all_y_rul: list[int] = []
    all_y_fail: list[int] = []
    all_meta: list[WindowMetadata] = []

    # Sort by engine and cycle to ensure correct temporal ordering
    df_sorted = df.sort_values([UNIT_ID, CYCLE]).reset_index(drop=True)

    for unit_id, engine_grp in df_sorted.groupby(UNIT_ID, sort=True):
        cycles = engine_grp[CYCLE].to_numpy()
        features = engine_grp[feature_cols].to_numpy(dtype=np.float64)
        rul_values = engine_grp[target_col].to_numpy(dtype=np.int64)
        fail_values = engine_grp[failure_target_col].to_numpy(dtype=np.int8)

        n_obs = len(engine_grp)

        # Skip engines shorter than the lookback window
        if n_obs < lookback:
            continue

        # Slide: start at index (lookback - 1) so the window is [start..current]
        for i in range(lookback - 1, n_obs):
            window = features[i - lookback + 1 : i + 1]  # shape (W, n_features)
            t_cycle = int(cycles[i])

            all_X.append(window)
            all_y_rul.append(int(rul_values[i]))
            all_y_fail.append(int(fail_values[i]))
            all_meta.append(
                WindowMetadata(
                    unit_id=int(unit_id),
                    target_cycle=t_cycle,
                    start_cycle=int(cycles[i - lookback + 1]),
                    raw_rul=int(rul_values[i]),
                    failure_label=int(fail_values[i]),
                )
            )

    if not all_X:
        return (
            np.empty((0, lookback, n_features), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int8),
            [],
        )

    X = np.stack(all_X, axis=0)
    y_rul = np.array(all_y_rul, dtype=np.int64)
    y_failure = np.array(all_y_fail, dtype=np.int8)

    return X, y_rul, y_failure, all_meta
