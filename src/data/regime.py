"""Deterministic regime assignment from operational settings.

The six FD004 operating regimes are defined by fixed rounding rules on the
three continuous operational-setting columns. No learned parameters are used.
The same function must work at inference time without access to any target
or validation data.

Methodology (Phase 0 approved):
    regime_key = f"{s1:.1f}|{s2:.2f}|{s3:.0f}"
    where s1 = operational_setting_1, etc.
    Rounding collapses sensor noise to the discrete regime boundaries
    established by the C-MAPSS design of experiments.

The mapping table below is derived from FD004 train data but contains NO
information from validation or test sets.
"""

from __future__ import annotations

import pandas as pd

from .loader import OPS_COLS

# ---------------------------------------------------------------------------
# Fixed regime mapping (6 regimes observed in FD004 train data)
# ---------------------------------------------------------------------------
REGIME_MAPPING: dict[str, int] = {
    "0.0|0.0|100.0": 0,
    "10.0|0.25|100.0": 1,
    "20.0|0.7|100.0": 2,
    "25.0|0.62|60.0": 3,
    "35.0|0.84|100.0": 4,
    "42.0|0.84|100.0": 5,
}

# Reverse mapping for human-readable reporting
REGIME_NAMES: dict[int, str] = {v: k for k, v in REGIME_MAPPING.items()}


def _regime_key(df: pd.DataFrame) -> pd.Series:
    """Build the discretized regime key string from operational settings.

    Rounding: setting1 -> 1 decimal, setting2 -> 2 decimals, setting3 -> 0.
    """
    s1 = df[OPS_COLS[0]].round(1)
    s2 = df[OPS_COLS[1]].round(2)
    s3 = df[OPS_COLS[2]].round(0)
    return s1.astype(str) + "|" + s2.astype(str) + "|" + s3.astype(str)


def assign_regime(df: pd.DataFrame) -> pd.Series:
    """Assign an integer regime ID (0-5) to each row.

    Parameters
    ----------
    df: pd.DataFrame
        Must contain the three operational_setting_* columns.

    Returns
    -------
    pd.Series
        Integer regime IDs aligned to df's index.

    Raises
    ------
    ValueError
        If a regime key is observed that is not in REGIME_MAPPING.
    """
    keys = _regime_key(df)
    unknown = set(keys.unique()) - set(REGIME_MAPPING.keys())
    if unknown:
        raise ValueError(
            f"Unseen regime keys detected (not in Phase-0 mapping): {sorted(unknown)}. "
            f"The fixed regime table may need extension for new datasets."
        )
    return keys.map(REGIME_MAPPING).astype(int)


def assign_regime_safe(df: pd.DataFrame) -> pd.Series:
    """Like assign_regime but maps unknown keys to -1 instead of raising.

    Use at inference time when you want to flag unexpected regimes rather
    than crash. Callers should check for -1 values.
    """
    keys = _regime_key(df)
    return keys.map(REGIME_MAPPING).fillna(-1).astype(int)
