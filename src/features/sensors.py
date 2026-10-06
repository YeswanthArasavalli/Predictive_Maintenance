"""Sensor selection configuration.

Provides two named configurations for ablation studies:

Config A (default)
    All 21 sensors. No assumptions about redundancy.

Config B
    Removes exact-duplicate sensors identified in Phase 0:
    - sensor_07  ≡ sensor_12  (Pearson r = 1.0)
    - sensor_08  ≡ sensor_18  (Pearson r = 1.0)
    - sensor_13  ≡ sensor_19  (Pearson r = 1.0)
    Optionally removes sensor_16 (near-constant; between-regime variance
    fraction = 0.9084, lowest of all 21 sensors).

Phase 1 does NOT declare Config B superior; both are configurable so that
later ablation can measure the impact empirically.
"""

from __future__ import annotations

# All 21 sensor column names (canonical order)
ALL_SENSORS: list[str] = [f"sensor_{i:02d}" for i in range(1, 22)]

# Exact duplicates identified in Phase 0 (remove the second of each pair)
_DUPLICATE_PAIRS: list[tuple[str, str]] = [
    ("sensor_07", "sensor_12"),
    ("sensor_08", "sensor_18"),
    ("sensor_13", "sensor_19"),
]

# Sensors to drop in Config B
_DROP_SET_B: set[str] = {b for _, b in _DUPLICATE_PAIRS}  # sensor_12, sensor_18, sensor_19

# sensor_16 is near-constant; optional additional removal
_OPTIONAL_DROP: set[str] = {"sensor_16"}


def get_sensor_config(config: str = "A", drop_optional: bool = False) -> list[str]:
    """Return the ordered list of sensor column names for the chosen configuration.

    Parameters
    ----------
    config: str
        "A" = all 21 sensors (default).
        "B" = remove exact duplicates (sensor_12, sensor_18, sensor_19).
        "B+" = same as B plus optionally sensor_16.
    drop_optional: bool
        If True and config="B", also drop sensor_16.

    Returns
    -------
    list[str]
        Sensor column names in deterministic canonical order.
    """
    if config == "A":
        return ALL_SENSORS.copy()
    elif config in ("B", "B+"):
        to_drop = _DROP_SET_B.copy()
        if config == "B+" or drop_optional:
            to_drop |= _OPTIONAL_DROP
        return [s for s in ALL_SENSORS if s not in to_drop]
    else:
        raise ValueError(f"Unknown sensor config {config!r}: must be 'A', 'B', or 'B+'")
