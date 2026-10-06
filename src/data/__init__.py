"""Data access layer: loading, parsing, and integrity checking of C-MAPSS."""

from .loader import (
    load_dataset,
    load_rul,
    derive_train_rul,
    final_cycle_rul,
    get_dataset_metadata,
    raw_dir,
    UNIT_ID,
    CYCLE,
    OPS_COLS,
    N_SENSORS,
    N_COLUMNS,
)
from .checks import (
    check_column_schema,
    check_unit_cycle_uniqueness,
    check_cyclic_order,
    check_dataset_inventory,
    check_test_rul_alignment,
)
from .contract import (
    IDENTIFIER_COLS,
    OPERATIONAL_COLS,
    SENSOR_COLS,
    DERIVED_COLS,
    TARGET_COLS,
    FORBIDDEN_FEATURE_COLS,
    INFERENCE_AVAILABLE_COLS,
    feature_columns,
    validate_feature_columns,
)
from .regime import (
    REGIME_MAPPING,
    REGIME_NAMES,
    assign_regime,
    assign_regime_safe,
)
from .splitter import engine_split

__all__ = [
    # loader
    "load_dataset",
    "load_rul",
    "derive_train_rul",
    "final_cycle_rul",
    "get_dataset_metadata",
    "raw_dir",
    "UNIT_ID",
    "CYCLE",
    "OPS_COLS",
    "N_SENSORS",
    "N_COLUMNS",
    # checks
    "check_column_schema",
    "check_unit_cycle_uniqueness",
    "check_cyclic_order",
    "check_dataset_inventory",
    "check_test_rul_alignment",
    # contract
    "IDENTIFIER_COLS",
    "OPERATIONAL_COLS",
    "SENSOR_COLS",
    "DERIVED_COLS",
    "TARGET_COLS",
    "FORBIDDEN_FEATURE_COLS",
    "INFERENCE_AVAILABLE_COLS",
    "feature_columns",
    "validate_feature_columns",
    # regime
    "REGIME_MAPPING",
    "REGIME_NAMES",
    "assign_regime",
    "assign_regime_safe",
    # splitter
    "engine_split",
]
