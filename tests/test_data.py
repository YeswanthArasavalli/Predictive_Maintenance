"""Phase-0 unit tests (§18): parser, schema, RUL, alignment, integrity, leakage guards."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import (
    check_column_schema,
    check_cyclic_order,
    check_dataset_inventory,
    check_test_rul_alignment,
    check_unit_cycle_uniqueness,
    derive_train_rul,
    load_dataset,
    load_rul,
)
from src.data import integrity
from src.data.loader import CYCLE, N_COLUMNS, OPS_COLS, UNIT_ID, _expected_columns, _sensor_names


@pytest.fixture(scope="module")
def fd001():
    return load_dataset("FD001", "train"), load_dataset("FD001", "test"), load_rul("FD001")


@pytest.fixture(scope="module")
def fd004():
    return load_dataset("FD004", "train"), load_dataset("FD004", "test"), load_rul("FD004")


def test_parser_column_count(fd001):
    tr, te, _ = fd001
    assert tr.shape[1] == N_COLUMNS == 26
    assert te.shape[1] == N_COLUMNS


def test_parser_column_names(fd001):
    tr, _, _ = fd001
    assert list(tr.columns) == _expected_columns()
    assert tr.columns[0] == "unit_id" and tr.columns[1] == "cycle"
    assert list(tr.columns[2:5]) == OPS_COLS
    assert _sensor_names()[0] == "sensor_01" and _sensor_names()[-1] == "sensor_21"
    assert len(_sensor_names()) == 21


def test_no_unnamed_trailing_column(fd001):
    tr, _, _ = fd001
    assert not any(str(c).startswith("Unnamed") for c in tr.columns)
    assert not any(str(c).strip() == "" for c in tr.columns)


def test_schema_checks_pass(fd001, fd004):
    for frames in (fd001, fd004):
        tr, te, _ = frames
        for df in (tr, te):
            ok, msg = check_column_schema(df)
            assert ok, msg
            ok, msg = check_unit_cycle_uniqueness(df)
            assert ok, msg
            ok, msg = check_cyclic_order(df)
            assert ok, msg
            ok, msg = check_dataset_inventory(df)
            assert ok, msg


def test_unit_cycle_uniqueness(fd004):
    tr, _, _ = fd004
    dup = tr.duplicated(subset=[UNIT_ID, CYCLE]).sum()
    assert dup == 0


def test_no_unexpected_nulls(fd001, fd004):
    for frames in (fd001, fd004):
        tr, te, _ = frames
        for df in (tr, te):
            assert df[OPS_COLS + _sensor_names()].isna().sum().sum() == 0
            vals = df[_sensor_names()].to_numpy(dtype=float)
            assert not np.isinf(vals).any()


def test_rul_calculation(fd001):
    tr, _, _ = fd001
    rul = derive_train_rul(tr)
    # Every engine's final observed row has RUL == 0.
    ends = tr.assign(_r=rul.to_numpy()).groupby(UNIT_ID)["_r"].last()
    assert (ends == 0).all()
    # RUL at cycle 1 equals (engine_life - 1); strictly decreases by 1 per cycle.
    for _, g in tr.groupby(UNIT_ID, sort=False):
        r = rul.loc[g.index].to_numpy()
        assert r[0] == len(g) - 1
        assert np.array_equal(r, np.arange(len(g) - 1, -1, -1))


def test_rul_uses_only_own_engine_history(fd001):
    """Leakage guard: per-engine RUL must not depend on any other engine."""
    tr, _, _ = fd001
    full = derive_train_rul(tr)
    one = tr[tr[UNIT_ID] == 1].copy()
    sub = derive_train_rul(one)
    # Recomputing RUL for engine 1 in isolation must match the full-frame value.
    np.testing.assert_array_equal(sub.to_numpy(), full.loc[one.index].to_numpy())


def test_test_rul_alignment(fd001, fd004):
    for frames in (fd001, fd004):
        _, te, ru = frames
        ok, msg = check_test_rul_alignment(te, ru)
        assert ok, msg


def test_rul_values_positive_and_bounded(fd004):
    _, _, ru = fd004
    assert (ru["rul"] > 0).all()
    assert ru["rul"].max() < 1000  # sane upper bound for a single trajectory


def test_integrity_checksums_match():
    results = integrity.verify_raw()
    assert results, "no checksums recorded"
    bad = [r.name for r in results if not r.match]
    assert not bad, f"checksum mismatch for: {bad}"
