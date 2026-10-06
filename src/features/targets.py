"""Target generation: raw RUL, model RUL target, and failure-risk labels.

Definitions (Phase 0 gate, sections 3 and 5):

raw_RUL
    Per-row: max_cycle(engine) - current_cycle.
    This is the *physical* remaining operational cycles to failure.
    The engine's max_cycle is target-generation metadata; it MUST NEVER
    appear in the feature set.

model_RUL_target
    Primary experiment: model_RUL_target = raw_RUL (no clipping).
    Optional ablation: model_RUL_target = min(raw_RUL, cap) where cap=125.
    A clipped value is a training convenience, NOT the physical RUL.

failure_risk_target_H
    Binary: 1 if raw_RUL <= H, else 0.
    H is in *operational cycles*, NOT days.
    Primary horizon: H=30.  Sensitivity: H=14 and H=50.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import CYCLE, UNIT_ID


def compute_raw_rul(df: pd.DataFrame) -> pd.Series:
    """Compute raw_RUL for every row using only the engine's own history.

    Parameters
    ----------
    df: pd.DataFrame
        Must contain unit_id and cycle columns. Typically the full FD004
        train frame (all 249 engines).

    Returns
    -------
    pd.Series
        Integer raw_RUL aligned to df's index.

    Note
    ----
    This function must NOT be called on the official test set during
    preprocessing. Test engines are truncated (they don't reach failure).
    Test RUL is available only in RUL_FD004.txt and is used only by the
    final evaluator after model selection is frozen.
    """
    max_cycle = df.groupby(UNIT_ID)[CYCLE].transform("max")
    rul = (max_cycle - df[CYCLE]).astype(np.int64)
    rul.name = "raw_RUL"
    return rul


def compute_model_rul_target(
    raw_rul: pd.Series,
    *,
    cap: int | None = None,
) -> pd.Series:
    """Derive the model_RUL_target from raw_RUL.

    Parameters
    ----------
    raw_rul: pd.Series
        The physical raw RUL values.
    cap: int | None
        None (default) — no clipping; returns raw_RUL unchanged.
        int (e.g. 125) — clips the target: min(raw_RUL, cap).
        The clipped result MUST be labelled "clipped RUL (cap=N)" and
        MUST NEVER be described as the physical/true RUL.

    Returns
    -------
    pd.Series
        model_RUL_target values aligned to raw_rul's index.
    """
    if cap is None:
        out = raw_rul.copy()
        out.name = "model_RUL_target"
        return out
    out = raw_rul.clip(upper=cap).astype(np.int64)
    out.name = "model_RUL_target"
    return out


def compute_failure_risk_target(
    raw_rul: pd.Series,
    *,
    horizon: int = 30,
) -> pd.Series:
    """Binary failure-risk label.

    1 if raw_RUL <= horizon else 0

    Parameters
    ----------
    raw_rul: pd.Series
        Integer raw_RUL values.
    horizon: int
        Number of *operational cycles* (NOT days) defining "imminent failure".

    Returns
    -------
    pd.Series
        Integer 0/1 label named "failure_risk_target_H{horizon}".
    """
    label = (raw_rul <= horizon).astype(np.int8)
    label.name = f"failure_risk_target_H{horizon}"
    return label
