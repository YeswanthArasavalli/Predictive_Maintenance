"""Row-level feature matrix builder for classical (non-sequence) models.

Phase 2 uses row-level (per-cycle) features for Ridge / HistGB / Logistic
Regression. Unlike the Phase 1 window tensor, this layer must support an
explicit controlled exception: the `cycle` column, normally forbidden, is
re-admitted only for the ablation experiments demanded by Phase 2 sections 7
and 22 ("Ridge + cycle" vs "Ridge without cycle").

Gate policy:
    - Every caller must pass `allow_cycle=True` explicitly to admit `cycle`
      into X. Otherwise `cycle` is rejected (Phase 1 contract enforced).
    - `unit_id`, `raw_RUL`, `model_RUL_target`, and any `failure_risk_target_*`
      column are ALWAYS rejected — there is no override for target leakage.
    - This module is only for Phase 2 row-level models. Phase 1 window
      generation still uses `src.data.contract.validate_feature_columns`
      directly and remains strict.

The `cycle` column is stored UNSCALED (integer age). This is intentional:
it is already on a small ordinal scale (1..~500), and applying a z-score to
it would make the "Ridge with cycle" ablation harder to interpret and would
mix cycle-position information into the per-regime scaler for Mode B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.contract import OPERATIONAL_COLS
from ..data.loader import CYCLE, UNIT_ID
from ..data.contract import SENSOR_COLS
from ..features.sensors import get_sensor_config

# Absolute never-allowed list (target/identifier leakage).
HARD_FORBIDDEN: set[str] = {
    UNIT_ID,
    "raw_RUL",
    "model_RUL_target",
    "operating_regime",  # categorical; not a model input in Phase 2 defaults
}


def _failure_target_columns(df: pd.DataFrame) -> set[str]:
    return {c for c in df.columns if str(c).startswith("failure_risk_target_")}


def validate_row_features(
    feature_cols: list[str],
    *,
    allow_cycle: bool,
) -> None:
    """Raise if `feature_cols` contains a hard-forbidden column.

    `cycle` is permitted only when `allow_cycle=True`. Everything else
    in `HARD_FORBIDDEN` plus any `failure_risk_target_*` column is always
    rejected.
    """
    hard = HARD_FORBIDDEN | {c for c in feature_cols if str(c).startswith("failure_risk_target_")}
    violations = [c for c in feature_cols if c in hard]
    if violations:
        raise ValueError(
            f"Row-level feature contract violation: {sorted(set(violations))}"
        )
    if (not allow_cycle) and (CYCLE in feature_cols):
        raise ValueError(
            "Row-level feature contract violation: 'cycle' requested but "
            "allow_cycle=False. Pass allow_cycle=True explicitly for the "
            "Phase 2 cycle-ablation experiments."
        )


def build_row_matrix(
    df_scaled: pd.DataFrame,
    *,
    sensor_config: str,
    include_cycle: bool,
) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names) for a row-level classical model.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        A partition (train OR val) of FD004 with sensors/settings already
        normalized in the desired mode (A global or B regime-conditioned).
        Must contain the 3 operational setting columns and 21 sensor columns,
        plus unit_id and cycle.
    sensor_config : str
        "A" (21 sensors), "B" (18, duplicates removed), or "B+" (17, also
        drops sensor_16).
    include_cycle : bool
        Explicit opt-in for the "with cycle" ablation. When False, `cycle`
        is excluded (Phase 1 contract enforced).

    Returns
    -------
    (X, feature_names) : tuple[np.ndarray, list[str]]
        X shape (n_rows, n_features), dtype float64.
        Feature ordering is deterministic: [ops_1..3] + [sensors in canonical
        order] + (["cycle"] if include_cycle else []).
    """
    sensors = get_sensor_config(sensor_config)
    names: list[str] = OPERATIONAL_COLS + sensors
    if include_cycle:
        names = names + [CYCLE]

    validate_row_features(names, allow_cycle=include_cycle)

    X = df_scaled[names].to_numpy(dtype=np.float64)
    return X, names


def build_row_matrix_from_raw(
    df: pd.DataFrame,
    *,
    sensor_config: str,
    include_cycle: bool,
) -> tuple[np.ndarray, list[str]]:
    """Same as `build_row_matrix` but assumes `df` columns are already in the
    desired normalization state (callers control this upstream). Convenience
    alias used by the driver so it can pass raw row-level frames directly."""
    return build_row_matrix(
        df, sensor_config=sensor_config, include_cycle=include_cycle
    )


def build_temporal_matrix(
    df_scaled: pd.DataFrame,
    *,
    sensor_config: str,
    include_cycle: bool,
    temporal_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Phase 2.5 row + causal-temporal feature matrix builder.

    Base features are exactly the Phase 2 row-level set (operational settings +
    selected sensors + optional cycle). ``temporal_names`` are the causal
    temporal columns already materialised in ``df_scaled`` by
    :func:`src.features.temporal.add_causal_temporal_features`; they are derived
    from sensor/setting columns only, so they inherit the same leakage
    guarantees (no target, no identifier, no future observation).

    Feature ordering is deterministic: [ops] + [sensors] + ([cycle] if included)
    + [temporal names in the given order].
    """
    sensors = get_sensor_config(sensor_config)
    base: list[str] = OPERATIONAL_COLS + sensors
    if include_cycle:
        base = base + [CYCLE]
    names = base + list(temporal_names)

    validate_row_features(names, allow_cycle=include_cycle)

    missing = [c for c in names if c not in df_scaled.columns]
    if missing:
        raise ValueError(f"build_temporal_matrix: missing columns {missing[:5]}")

    X = df_scaled[names].to_numpy(dtype=np.float64)
    return X, names


def select_temporal_monitors(
    train_scaled: pd.DataFrame,
    candidate_sensors: list[str],
    *,
    top_n: int,
    min_engine_length: int = 8,
) -> list[str]:
    """Rank candidate sensors by train-only within-engine degradation signal.

    Selection is performed EXCLUSIVELY on the training partition (spec section
    6: "If feature selection is performed, it must be based exclusively on
    training data"). Validation performance is never inspected.

    Score for a sensor = mean, over training engines with at least
    ``min_engine_length`` observations, of ``|corr(sensor, within_engine_position)|``.
    This measures how strongly the sensor drifts monotonically across an engine's
    own life after normalization, which is precisely the trajectory signal the
    temporal features are meant to capture. Ties are broken by sensor name for
    full determinism.

    Parameters
    ----------
    train_scaled : pd.DataFrame
        Training rows already normalized (Mode A or B). Must contain unit_id,
        cycle and the candidate sensor columns.
    candidate_sensors : list[str]
        Sensor columns eligible to receive temporal features (from the chosen
        sensor config).
    top_n : int
        How many monitors to keep.

    Returns
    -------
    list[str]
        The selected sensor names, in descending signal order (deterministic).
    """
    scores: dict[str, float] = {}
    grouped = train_scaled.groupby(UNIT_ID, sort=True)
    # Pre-compute each engine's within-engine position once.
    pos_by_engine = {
        uid: np.arange(len(sub), dtype=np.float64)
        for uid, sub in grouped
    }
    valid_engines = {uid: pos for uid, pos in pos_by_engine.items() if len(pos) >= min_engine_length}
    for sensor in candidate_sensors:
        abs_corrs: list[float] = []
        for uid, pos in valid_engines.items():
            sub = grouped.get_group(uid)
            x = sub[sensor].to_numpy(dtype=np.float64)
            if np.std(x) < 1e-12 or np.std(pos) < 1e-12:
                continue
            r = np.corrcoef(x, pos)[0, 1]
            if np.isfinite(r):
                abs_corrs.append(abs(float(r)))
        scores[sensor] = float(np.mean(abs_corrs)) if abs_corrs else 0.0
    ranked = sorted(candidate_sensors, key=lambda s: (-scores[s], s))
    return ranked[: max(0, int(top_n))]
