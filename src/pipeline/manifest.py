"""Generate the reproducible Phase 1 dataset manifest.

The manifest is a machine-readable JSON file stored at:
    results/audits/phase1_dataset_manifest.json

It captures every parameter, hash, and environment detail needed to
reproduce the exact processed dataset that any future model will consume.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent


def _get_version(pkg: str) -> str | None:
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _checksums() -> dict[str, str]:
    """Read the data checksum file if it exists."""
    path = ROOT / "data" / ".checksums.txt"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def generate_manifest(
    *,
    fd_id: str,
    train_engine_ids: list[int],
    val_engine_ids: list[int],
    sensor_config: str,
    sensors: list[str],
    normalization_mode: str,
    normalization_fit_scope: str,
    rul_target_definition: str,
    rul_cap: int | None,
    failure_horizon: int,
    lookback: int,
    seed: int,
    scaler_dict: dict[str, Any],
    n_train_rows: int,
    n_val_rows: int,
    n_train_windows: int,
    n_val_windows: int,
    regime_distribution_train: dict[str, int],
    regime_distribution_val: dict[str, int],
) -> dict[str, Any]:
    """Build the full manifest dictionary."""
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": 1,
        "dataset_version": "CMAPSS_v1.0",
        "primary_dataset": fd_id,
        "engine_split": {
            "train_engine_ids": train_engine_ids,
            "val_engine_ids": val_engine_ids,
            "n_train_engines": len(train_engine_ids),
            "n_val_engines": len(val_engine_ids),
            "no_overlap": len(set(train_engine_ids) & set(val_engine_ids)) == 0,
        },
        "official_test_status": "UNTOUCHED (never loaded during this pipeline)",
        "sensor_configuration": {
            "config_name": sensor_config,
            "sensors_used": sensors,
            "n_sensors": len(sensors),
        },
        "normalization": {
            "mode": normalization_mode,
            "fit_scope": normalization_fit_scope,
            "parameters_summary": scaler_dict,
        },
        "rul_target": {
            "definition": rul_target_definition,
            "cap": rul_cap,
            "raw_RUL_never_overwritten": True,
        },
        "failure_risk_target": {
            "horizon_H": failure_horizon,
            "horizon_unit": "operational_cycles (NOT days)",
            "definition": f"1 if raw_RUL <= {failure_horizon} else 0",
        },
        "window_generation": {
            "lookback_W": lookback,
            "n_train_windows": n_train_windows,
            "n_val_windows": n_val_windows,
            "warm_up_discarded": True,
            "causal": True,
            "single_engine": True,
        },
        "row_counts": {
            "n_train_rows": n_train_rows,
            "n_val_rows": n_val_rows,
        },
        "regime_distribution": {
            "train": regime_distribution_train,
            "val": regime_distribution_val,
        },
        "reproducibility": {
            "random_seed": seed,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": {
                "numpy": _get_version("numpy"),
                "pandas": _get_version("pandas"),
                "scikit_learn": _get_version("sklearn"),
                "pyarrow": _get_version("pyarrow"),
                "joblib": _get_version("joblib"),
                "scipy": _get_version("scipy"),
            },
            "git_commit": _git_commit(),
            "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "data_integrity": {
            "checksums": _checksums(),
            "raw_data_modified": False,
        },
    }
    return manifest


def write_manifest(manifest: dict[str, Any], path: Path | None = None) -> Path:
    """Serialize the manifest to disk."""
    if path is None:
        path = ROOT / "results" / "audits" / "phase1_dataset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
