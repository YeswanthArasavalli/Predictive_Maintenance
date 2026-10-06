"""Engine-grouped, regime-stratified train/validation splitting.

Design rationale (Phase 0 gate, section 7):
- C-MAPSS engine IDs carry NO temporal meaning. An engine with ID 200 is
  not "later" than engine 100. Forward-chaining in the classical time-series
  sense is not applicable here.
- The split is made at the ENGINE level: whole engines are assigned to
  train or validation. No engine appears in both partitions.
- Regime stratification ensures each partition contains a representative
  sample of all six FD004 operating regimes.
- The official test set (test_FD004.txt + RUL_FD004.txt) is NEVER loaded
  or referenced by this module.

Implementation:
- Uses sklearn StratifiedGroupKFold (available in sklearn >= 1.3) to
  perform a single split (train vs val) stratified on regime with
  groups=unit_id. This guarantees:
  1. No engine crosses the train/val boundary.
  2. Regime proportions are approximately preserved in both partitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .loader import UNIT_ID


def engine_split(
    train_df: pd.DataFrame,
    *,
    regime_col: str = "operating_regime",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split FD004 training engines into train and validation sets.

    Parameters
    ----------
    train_df: pd.DataFrame
        The full FD004 training frame (all 249 engines). Must contain the
        regime_col and UNIT_ID columns.
    regime_col: str
        Column name holding the integer regime label (0-5).
    val_fraction: float
        Approximate fraction of engines to assign to validation.
    seed: int
        Random seed. StratifiedGroupKFold is deterministic given the
        same data and n_splits; the seed is used for an initial shuffle
        of engine order before the split.

    Returns
    -------
    tuple[list[int], list[int]]
        (train_engine_ids, val_engine_ids) — sorted lists of unit_id integers.

    Raises
    ------
    ValueError
        If an engine appears in both returned lists (should be impossible
        by construction, but checked).
    """
    # Build per-engine regime label for stratification.
    # In FD004, each engine traverses ALL 6 regimes during its lifecycle;
    # we use the first-encountered regime as the stratification label.
    # (Since every engine has all regimes, stratification is effectively
    # a balanced group split — but the mechanism is general and safe.)
    engine_regimes = (
        train_df.groupby(UNIT_ID)[regime_col]
        .first()  # one label per engine for stratification
        .reset_index()
    )

    unique_engines = engine_regimes[UNIT_ID].to_numpy()
    regime_labels = engine_regimes[regime_col].to_numpy()

    # Shuffle engine order before splitting to break any incidental file-order
    # correlation. This makes the split independent of how the data file lists
    # engines. np.random.RandomState for cross-version determinism.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_engines))
    unique_engines = unique_engines[perm]
    regime_labels = regime_labels[perm]

    # n_splits=5 means 1/5 = 20% validation. For other fractions, approximate.
    n_splits = max(2, round(1.0 / val_fraction))

    # Use StratifiedGroupKFold to get one split
    # groups = unit_id so no engine is split across folds
    groups = unique_engines.copy()

    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    # StratifiedGroupKFold uses X=None (not needed), y=regime_labels, groups=engine_ids
    fold_indices = list(sgk.split(X=np.zeros(len(unique_engines)), y=regime_labels, groups=groups))

    # Use the first fold as validation, rest as train
    _, val_idx = fold_indices[0]

    val_engines = sorted(unique_engines[val_idx].tolist())
    train_engines = sorted(
        [e for e in unique_engines.tolist() if e not in set(val_engines)]
    )

    # Safety check: no overlap
    overlap = set(train_engines) & set(val_engines)
    if overlap:
        raise ValueError(
            f"Engine overlap detected between train and validation: {sorted(overlap)}"
        )

    return train_engines, val_engines
