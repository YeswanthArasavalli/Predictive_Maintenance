"""Feature engineering layer: sensors, normalization, targets, windows."""

from .sensors import ALL_SENSORS, get_sensor_config
from .normalization import GlobalScaler, RegimeScaler
from .targets import (
    compute_raw_rul,
    compute_model_rul_target,
    compute_failure_risk_target,
)
from .windows import WindowMetadata, generate_windows

__all__ = [
    "ALL_SENSORS",
    "get_sensor_config",
    "GlobalScaler",
    "RegimeScaler",
    "compute_raw_rul",
    "compute_model_rul_target",
    "compute_failure_risk_target",
    "WindowMetadata",
    "generate_windows",
]
