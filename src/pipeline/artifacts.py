"""Preprocessing artifact persistence.

Stores fitted scalers, regime mapping, feature schema, sensor configuration,
and preprocessing metadata to `results/preprocessing/`.

Artifacts generated from validation/test data are NEVER stored as training
artifacts. This module only persists objects that were fitted on training data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
PREPROCESSING_DIR = ROOT / "results" / "preprocessing"


def persist_artifacts(
    *,
    global_scaler: Any | None = None,
    regime_scaler: Any | None = None,
    feature_cols: list[str],
    sensor_config: str,
    sensors: list[str],
    normalization_mode: str,
    seed: int,
    lookback: int,
    failure_horizon: int,
    rul_cap: int | None,
) -> Path:
    """Write all preprocessing artifacts to disk.

    Returns the directory path where artifacts are stored.
    """
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Serialize scalers
    if global_scaler is not None and hasattr(global_scaler, "scaler") and global_scaler.scaler is not None:
        joblib.dump(global_scaler.scaler, PREPROCESSING_DIR / "global_scaler.joblib")

    if regime_scaler is not None and regime_scaler.scalers:
        joblib.dump(regime_scaler.scalers, PREPROCESSING_DIR / "regime_scalers.joblib")

    # 2. Regime mapping configuration (JSON)
    from ..data.regime import REGIME_MAPPING, REGIME_NAMES

    regime_config = {
        "mapping": {k: v for k, v in REGIME_MAPPING.items()},
        "names": {str(k): v for k, v in REGIME_NAMES.items()},
        "method": "deterministic rounding of operational settings",
        "rounding": {"setting1": 1, "setting2": 2, "setting3": 0},
    }
    _write_json(PREPROCESSING_DIR / "regime_mapping.json", regime_config)

    # 3. Feature schema
    feature_schema = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "sensors": sensors,
        "operational_settings_included": any(
            c.startswith("operational_setting") for c in feature_cols
        ),
    }
    _write_json(PREPROCESSING_DIR / "feature_schema.json", feature_schema)

    # 4. Sensor configuration
    sensor_cfg = {
        "config_name": sensor_config,
        "sensors_used": sensors,
        "n_sensors": len(sensors),
    }
    _write_json(PREPROCESSING_DIR / "sensor_config.json", sensor_cfg)

    # 5. Preprocessing metadata
    metadata = {
        "normalization_mode": normalization_mode,
        "normalization_fit_scope": "train_engines_only",
        "seed": seed,
        "lookback_W": lookback,
        "failure_horizon_H": failure_horizon,
        "rul_cap": rul_cap,
        "causal_scaler_note": (
            "Scaler parameters are deployment-known constants learned from "
            "the training partition. They are NOT computed from the future "
            "trajectory of any single engine being scored."
        ),
    }
    _write_json(PREPROCESSING_DIR / "preprocessing_metadata.json", metadata)

    return PREPROCESSING_DIR


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
