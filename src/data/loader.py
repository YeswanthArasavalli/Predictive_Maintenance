"""Load and parse the C-MAPSS Turbofan Engine Degradation Simulation files.

The dataset package is the bundle of 13 ``.txt`` files (4 train, 4 test,
4 RUL, 1 readme) plus the PDF. Each row is one snapshot taken during a single
operational cycle. Each file has 26 space-separated numeric columns:

1.  unit number                -> ``unit_id``
2.  time, in cycles            -> ``cycle``
3.  operational setting 1      -> ``operational_setting_1``
4.  operational setting 2      -> ``operational_setting_2``
5.  operational setting 3      -> ``operational_setting_3``
6-26. sensor measurement 1..21 -> ``sensor_01``..``sensor_21``

The official ``readme.txt`` words the last column as "26) sensor measurement
26"; this is a well-known documentation quirk. There are 26 *columns* total, of
which 21 are *sensors* (columns 6-26). The Phase-0 methodology fixes the schema
at ``sensor_01``..``sensor_21``; that is what this parser produces.

No unnamed columns are retained: the C parser is driven by ``sep=r"\\s+"`` with
explicit ``names``, so trailing whitespace on a line does not create a spurious
empty final column.

Constants
---------
UNIT_ID, CYCLE, OPS_COLS, SENSOR_PREFIX, N_SENSORS, N_COLUMNS
    Column metadata derived from the official readme + Phase-0 schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

UNIT_ID = "unit_id"
CYCLE = "cycle"
OPS_COLS = ["operational_setting_1", "operational_setting_2", "operational_setting_3"]
SENSOR_PREFIX = "sensor_{:02d}"
N_SENSORS = 21
N_COLUMNS = N_SENSORS + 5  # 26

# Default raw location if config is unavailable. Matches data/raw/README.md.
_DEFAULT_RAW_DIR = "data/raw/CMAPSSData_v1.0"


@dataclass
class DatasetMetadata:
    """Metadata describing a loaded C-MAPSS dataset (readme-level facts)."""

    fd_id: str                    # e.g. "FD004"
    split: str                    # "train" or "test"
    source_path: Path
    n_rows: int
    n_engines: int
    n_columns: int
    columns: list[str] = field(default_factory=list)
    conditions: int | None = None     # number of distinct operating conditions (readme)
    fault_modes: int | None = None    # number of fault modes (readme)
    readme_engines: int | None = None  # engine count the readme *claims*


def _sensor_names() -> list[str]:
    """sensor_01 .. sensor_21."""
    return [SENSOR_PREFIX.format(i) for i in range(1, N_SENSORS + 1)]


def _expected_columns() -> list[str]:
    return [UNIT_ID, CYCLE] + OPS_COLS + _sensor_names()


@lru_cache(maxsize=1)
def raw_dir(root: Path | str | None = None) -> Path:
    """Resolve the immutable raw-data directory.

    Prefers ``data.raw_dir`` from ``configs/project.yaml``; falls back to the
    documented default. The value is resolved relative to the project root.
    """
    root = Path(root) if root is not None else ROOT
    cfg = root / "configs" / "project.yaml"
    rel = _DEFAULT_RAW_DIR
    if cfg.exists():
        try:
            import yaml

            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            rel = (data.get("data") or {}).get("raw_dir") or _DEFAULT_RAW_DIR
        except Exception:  # pragma: no cover - config read is best-effort
            rel = _DEFAULT_RAW_DIR
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def _read_table(path: Path) -> pd.DataFrame:
    """Parse one train/test file into a schema-checked DataFrame."""
    headers = _expected_columns()
    df = pd.read_csv(
        path,
        header=None,
        names=headers,
        sep=r"\s+",
        engine="c",
        dtype={UNIT_ID: np.int64, CYCLE: np.int64},
    )
    if df.shape[1] != N_COLUMNS:
        raise ValueError(
            f"{path}: expected {N_COLUMNS} columns, got {df.shape[1]} "
            f"(names: {list(df.columns)})"
        )
    if list(df.columns) != headers:
        raise ValueError(f"{path}: column order mismatch: {list(df.columns)}")

    numeric_cols = OPS_COLS + _sensor_names()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.astype({UNIT_ID: np.int64, CYCLE: np.int64}).reset_index(drop=True)


def load_dataset(fd_id: str, split: str, root: Path | str | None = None) -> pd.DataFrame:
    """Load a train or test partition for FD00X.

    Parameters
    ----------
    fd_id: str
        One of FD001..FD004.
    split: str
        One of "train" or "test".
    root: Path | str | None
        Project root. Defaults to the repository root (parent of ``src/``).

    Returns
    -------
    pd.DataFrame
        Columns: unit_id, cycle, operational_setting_1..3, sensor_01..21.
    """
    root = Path(root) if root is not None else ROOT
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = raw_dir(root) / f"{split}_{fd_id}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_table(path)


def _parse_rul_strict(path: Path) -> list[int]:
    """Strict RUL loader: one integer per line, one per test engine in order."""
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if not re.fullmatch(r"\d+", line):
            raise ValueError(f"RUL file {path.name} has non-integer line: {line!r}")
        values.append(int(line))
    return values


def load_rul(fd_id: str, root: Path | str | None = None) -> pd.DataFrame:
    """Load ``RUL_{fd_id}.txt`` as a DataFrame aligned to test engines.

    Returns a DataFrame indexed by ``unit_id`` (1..N, in file order) with a
    single ``rul`` column holding the supplied final-cycle RUL for that test
    engine. The caller must verify alignment via ``check_test_rul_alignment``.
    """
    root = Path(root) if root is not None else ROOT
    path = raw_dir(root) / f"RUL_{fd_id}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    values = _parse_rul_strict(path)
    n = len(values)
    return pd.DataFrame(
        {"unit_id": np.arange(1, n + 1, dtype=np.int64), "rul": np.array(values, dtype=np.int64)}
    ).set_index("unit_id")


def derive_train_rul(train: pd.DataFrame) -> pd.Series:
    """Natural training RUL: max_cycle(engine) - current_cycle, per row.

    This is a *per-row* Series aligned to ``train``'s index. It uses only the
    engine's own final observed cycle (available at the end of a run-to-failure
    trajectory) and never any other engine's or the test set's information.
    For Phase-1 feature construction the engine's max cycle must NOT leak into
    historical features; this Series is the *target*, not a feature. See
    reports/LEAKAGE_AUDIT.md.
    """
    max_cycle = train.groupby(UNIT_ID)[CYCLE].transform("max")
    rul = (max_cycle - train[CYCLE]).astype(np.int64)
    rul.name = "rul"
    return rul


def final_cycle_rul(train: pd.DataFrame) -> pd.Series:
    """Per-engine total observable life = max_cycle - 1 (RUL at cycle 1)."""
    return (train.groupby(UNIT_ID)[CYCLE].max() - 1).astype(np.int64)


# readme.txt engine claims (used for provenance cross-checks in the audit).
README_ENGINE_CLAIMS = {
    "FD001": {"train": 100, "test": 100},
    "FD002": {"train": 260, "test": 259},
    "FD003": {"train": 100, "test": 100},
    "FD004": {"train": 248, "test": 249},
}

_README_SCENARIO = {
    ("FD001", "train"): dict(conditions=1, fault_modes=1),
    ("FD001", "test"): dict(conditions=1, fault_modes=1),
    ("FD002", "train"): dict(conditions=6, fault_modes=1),
    ("FD002", "test"): dict(conditions=6, fault_modes=1),
    ("FD003", "train"): dict(conditions=1, fault_modes=2),
    ("FD003", "test"): dict(conditions=1, fault_modes=2),
    ("FD004", "train"): dict(conditions=6, fault_modes=2),
    ("FD004", "test"): dict(conditions=6, fault_modes=2),
}


def get_dataset_metadata(
    fd_id: str, split: str, root: Path | str | None = None
) -> DatasetMetadata:
    """Return readme-level metadata plus the observed row/engine counts."""
    root = Path(root) if root is not None else ROOT
    info = _README_SCENARIO[(fd_id, split)]
    path = raw_dir(root) / f"{split}_{fd_id}.txt"
    df = _read_table(path)
    return DatasetMetadata(
        fd_id=fd_id,
        split=split,
        source_path=path.relative_to(root),
        n_rows=int(len(df)),
        n_engines=int(df[UNIT_ID].nunique()),
        n_columns=N_COLUMNS,
        columns=_expected_columns(),
        conditions=info["conditions"],
        fault_modes=info["fault_modes"],
        readme_engines=README_ENGINE_CLAIMS[fd_id][split],
    )
