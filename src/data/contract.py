"""Formal data contract for the predictive-maintenance pipeline.

Defines column-level roles: source, derived, target, forbidden, and
inference-time availability. All downstream code (splitter, normalizer,
window generator, manifest) MUST reference these constants rather than
hard-coding column names.

Leakage rule enforced by this contract:
    raw_RUL, model_RUL_target, and failure_risk_target_* MUST NEVER
    appear in the model input feature set.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Identifier columns
# ---------------------------------------------------------------------------
IDENTIFIER_COLS: list[str] = ["unit_id", "cycle"]

# ---------------------------------------------------------------------------
# Operational settings (known at inference time, from flight data)
# ---------------------------------------------------------------------------
OPERATIONAL_COLS: list[str] = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]

# ---------------------------------------------------------------------------
# Sensor columns
# ---------------------------------------------------------------------------
SENSOR_COLS: list[str] = [f"sensor_{i:02d}" for i in range(1, 22)]

# ---------------------------------------------------------------------------
# Derived columns (computed by the pipeline; not present in raw files)
# ---------------------------------------------------------------------------
DERIVED_COLS: list[str] = [
    "raw_RUL",
    "model_RUL_target",
    "operating_regime",
    "failure_risk_target_H30",
]

# ---------------------------------------------------------------------------
# Target columns (used as y; MUST NEVER enter X)
# ---------------------------------------------------------------------------
TARGET_COLS: list[str] = [
    "raw_RUL",
    "model_RUL_target",
    "failure_risk_target_H30",
]

# ---------------------------------------------------------------------------
# Columns forbidden from feature input (targets + identifiers)
# ---------------------------------------------------------------------------
FORBIDDEN_FEATURE_COLS: list[str] = [
    "raw_RUL",
    "model_RUL_target",
    "failure_risk_target_H30",
    "unit_id",
    "cycle",
]

# ---------------------------------------------------------------------------
# Columns available at inference time (no targets)
# ---------------------------------------------------------------------------
INFERENCE_AVAILABLE_COLS: list[str] = (
    IDENTIFIER_COLS + OPERATIONAL_COLS + SENSOR_COLS
)

# ---------------------------------------------------------------------------
# Feature columns that may enter X (sensors + operational settings)
# ---------------------------------------------------------------------------
# Note: operating_regime is derived but may enter X in some model
# configurations; it is NOT a target and is leakage-free (deterministic
# function of operational settings). We list it as conditionally allowed.
CONDITIONAL_FEATURE_COLS: list[str] = ["operating_regime"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def feature_columns(
    sensors: list[str],
    include_operational: bool = True,
    include_regime: bool = False,
) -> list[str]:
    """Return the ordered list of column names allowed in the feature matrix.

    Parameters
    ----------
    sensors: list[str]
        Sensor column names selected by the sensor configuration.
    include_operational: bool
        Whether to include the 3 operational-setting columns.
    include_regime: bool
        Whether to include the derived operating_regime column.

    Returns
    -------
    list[str]
        Deterministic feature column list.
    """
    cols: list[str] = []
    if include_operational:
        cols.extend(OPERATIONAL_COLS)
    cols.extend(sensors)
    if include_regime:
        cols.append("operating_regime")
    return cols


def validate_feature_columns(feature_cols: list[str]) -> None:
    """Raise ValueError if any forbidden column appears in the feature list.

    This is a data-contract guard, not a model check. It ensures the
    pipeline cannot accidentally pass targets or identifiers as features.
    """
    violations = [c for c in feature_cols if c in FORBIDDEN_FEATURE_COLS]
    if violations:
        raise ValueError(
            f"Data contract violation: forbidden columns in feature set: {violations}"
        )
