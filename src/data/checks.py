"""Data integrity checks for parsed C-MAPSS tables.

Each function takes a DataFrame loaded by :mod:`src.data.loader` and returns
``(passed, message)`` so the audit can report pass/fail with an explanation.

Note on operational settings: columns 3-5 are *continuous* operating-condition
values (e.g. sea-level altitude, Mach number, throttle angle), NOT integers
1..3. "SIX conditions" in the readme refers to six discrete (setting1, setting2)
regimes, not to integer-coded columns. These checks therefore only verify that
the settings are numeric and finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import (
    OPS_COLS,
    UNIT_ID,
    CYCLE,
    N_COLUMNS,
    _sensor_names,
    _expected_columns,
)

__all__ = [
    "check_column_schema",
    "check_unit_cycle_uniqueness",
    "check_cyclic_order",
    "check_dataset_inventory",
    "check_test_rul_alignment",
]


def check_column_schema(df: pd.DataFrame, fd_id: str | None = None) -> tuple[bool, str]:
    """Check that the parsed frame matches the expected C-MAPSS schema."""
    expected = _expected_columns()
    if df.shape[1] != N_COLUMNS:
        return False, f"column count mismatch: expected {N_COLUMNS}, got {df.shape[1]}"
    if list(df.columns) != expected:
        return False, f"column order/name mismatch: got {list(df.columns)}"
    if df[UNIT_ID].dtype != np.int64 or df[CYCLE].dtype != np.int64:
        return False, "unit_id and cycle must be int64"
    for c in OPS_COLS + _sensor_names():
        if not pd.api.types.is_numeric_dtype(df[c]):
            return False, f"expected numeric dtype for {c}, got {df[c].dtype}"
    return True, "schema matches 2 keys + 3 settings + 21 sensors (26 columns)"


def check_unit_cycle_uniqueness(df: pd.DataFrame) -> tuple[bool, str]:
    """Every (unit_id, cycle) pair is unique and cycles are consecutive 1..n."""
    n_pairs = len(df)
    n_unique = df[[UNIT_ID, CYCLE]].drop_duplicates().shape[0]
    if n_pairs != n_unique:
        return False, f"duplicate (unit_id, cycle) rows: {n_pairs - n_unique} extra"
    for u, grp in df.groupby(UNIT_ID, sort=False):
        cyc = np.sort(grp[CYCLE].to_numpy())
        expected = np.arange(1, len(cyc) + 1)
        if not np.array_equal(cyc, expected):
            return False, (
                f"engine {u}: cycles not consecutive 1..{len(cyc)} "
                f"(min={cyc.min()}, max={cyc.max()}, n={len(cyc)})"
            )
    return True, "all (unit_id, cycle) pairs unique; cycles consecutive 1..n"


def check_cyclic_order(df: pd.DataFrame) -> tuple[bool, str]:
    """Within each engine, ``cycle`` is monotonically increasing (no dup/out-of-order)."""
    for u, grp in df.groupby(UNIT_ID, sort=False):
        cyc = grp[CYCLE].to_numpy()
        if not np.all(np.diff(cyc) > 0):
            return False, f"engine {u}: cycles not strictly increasing"
    return True, "cycles strictly increasing per engine (in file order)"


def check_dataset_inventory(df: pd.DataFrame) -> tuple[bool, str]:
    """Engine ids form 1..N, key columns finite, sensors have no NaN/inf."""
    units = np.sort(df[UNIT_ID].unique())
    n = int(df[UNIT_ID].nunique())
    if not np.array_equal(units, np.arange(1, n + 1)):
        return False, f"unit ids must form 1..{n}, got min={units.min()} max={units.max()}"
    for c in [UNIT_ID, CYCLE] + OPS_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any() or np.isinf(v).any():
            return False, f"NaN/inf present in key column {c}"
    for c in _sensor_names():
        v = df[c].to_numpy(dtype=float)
        if np.isnan(v).any() or np.isinf(v).any():
            return False, f"NaN/inf present in sensor column {c}"
    if (df[CYCLE].to_numpy() <= 0).any():
        return False, "non-positive cycles detected"
    return True, f"inventory OK for {n} engines, {len(df)} rows"


def check_test_rul_alignment(test: pd.DataFrame, rul: pd.DataFrame) -> tuple[bool, str]:
    """Verify test-frame engine ids align one-to-one with ``RUL_{fd_id}.txt``."""
    test_ids = np.sort(test[UNIT_ID].unique())
    rul_ids = np.sort(rul.reset_index()["unit_id"].to_numpy())
    if len(test_ids) != len(rul_ids):
        return False, f"count mismatch: test={len(test_ids)} rul={len(rul_ids)}"
    if not np.array_equal(test_ids, rul_ids):
        missing = sorted(set(test_ids) ^ set(rul_ids))
        return False, f"test/RUL engine id mismatch: {missing}"
    return True, f"{len(test_ids)} test engines align one-to-one with RUL file"
