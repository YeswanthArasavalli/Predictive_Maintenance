"""Reusable Phase-0 analysis primitives.

Kept as pure functions returning dicts / DataFrames so both the audit driver
script and the unit tests exercise the same code paths (no notebook-only logic).
All functions read only through :mod:`src.data.loader` and never mutate raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import (
    CYCLE,
    OPS_COLS,
    UNIT_ID,
    _sensor_names,
    derive_train_rul,
)

FD_IDS = ["FD001", "FD002", "FD003", "FD004"]
SENSORS = _sensor_names()


def _series_stats(x: pd.Series) -> dict:
    v = x.to_numpy(dtype=float)
    return {
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "median": float(np.nanmedian(v)),
        "q01": float(np.nanquantile(v, 0.01)),
        "q25": float(np.nanquantile(v, 0.25)),
        "q75": float(np.nanquantile(v, 0.75)),
        "q99": float(np.nanquantile(v, 0.99)),
        "variance": float(np.nanvar(v, ddof=1)) if v.size > 1 else 0.0,
        "missing": int(np.isnan(v).sum()),
        "n_unique": int(pd.Series(v).nunique(dropna=True)),
    }


def inventory_entry(
    *, train: pd.DataFrame, test: pd.DataFrame, rul: pd.DataFrame,
    readme_train: int, readme_test: int,
) -> dict:
    """§7 dataset inventory for one FD partition set."""
    def cycles_per_engine(df: pd.DataFrame) -> dict:
        c = df.groupby(UNIT_ID)[CYCLE].count().to_numpy()
        return {
            "n_engines": int(len(c)),
            "n_rows": int(len(df)),
            "min_cycles": int(c.min()),
            "max_cycles": int(c.max()),
            "median_cycles": float(np.median(c)),
            "mean_cycles": float(np.mean(c)),
        }

    tr = cycles_per_engine(train)
    te = cycles_per_engine(test)
    r = rul["rul"].to_numpy()
    rul_stats = {
        "n_test_engines": int(len(r)),
        "min_rul": int(r.min()),
        "max_rul": int(r.max()),
        "mean_rul": float(np.mean(r)),
        "median_rul": float(np.median(r)),
    }
    return {
        "train": tr,
        "test": te,
        "rul": rul_stats,
        "test_matches_rul_count": te["n_engines"] == rul_stats["n_test_engines"],
        "readme_train_engines": readme_train,
        "readme_test_engines": readme_test,
        "train_matches_readme": tr["n_engines"] == readme_train,
        "test_matches_readme": te["n_engines"] == readme_test,
    }


def quality_audit(df: pd.DataFrame, label: str) -> dict:
    """§8 data-quality audit for one frame (train or test of one FD)."""
    numeric = OPS_COLS + SENSORS
    key_dupes = int(df.duplicated(subset=[UNIT_ID, CYCLE]).sum())
    full_dupes = int(df.duplicated().sum())
    all_na = df[numeric].isna().sum().sum()
    inf_mask = np.isinf(df[numeric].to_numpy(dtype=float))
    n_inf = int(inf_mask.sum())
    cycle_ok = all(
        np.array_equal(np.sort(g[CYCLE].to_numpy()), np.arange(1, len(g) + 1))
        for _, g in df.groupby(UNIT_ID, sort=False)
    )
    # constant / near-constant / low-variance sensors
    var = df[SENSORS].var(ddof=1)
    nunique = df[SENSORS].nunique()
    constant = [c for c in SENSORS if var[c] == 0]
    near_const = [c for c in SENSORS if 0 < var[c] < 1e-6 or nunique[c] <= 2]
    low_var = [c for c in SENSORS if 0 < var[c] < 1e-3]
    # negative-value screen (only meaningful where physically impossible)
    neg_counts = {
        c: int((df[c].to_numpy(dtype=float) < 0).sum())
        for c in numeric
        if (df[c].to_numpy(dtype=float) < 0).any()
    }
    return {
        "dataset": label,
        "n_rows": int(len(df)),
        "n_engines": int(df[UNIT_ID].nunique()),
        "nulls_numeric": int(all_na),
        "infinite_values": n_inf,
        "duplicate_rows_full": full_dupes,
        "duplicate_unit_cycle": key_dupes,
        "cycles_consecutive_1n": bool(cycle_ok),
        "constant_sensors": constant,
        "near_constant_sensors": near_const,
        "low_variance_sensors": low_var,
        "columns_with_negatives": neg_counts,
    }


def sensor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """§9 per-sensor descriptive statistics (one row per sensor)."""
    rows = {c: _series_stats(df[c]) for c in SENSORS}
    out = pd.DataFrame(rows).T
    out.index.name = "sensor"
    return out


def sensor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """§9 Pearson correlation matrix across the 21 sensors."""
    return df[SENSORS].corr(method="pearson")


def correlation_pairs(corr: pd.DataFrame, threshold: float = 0.95) -> list[dict]:
    """Highly-correlated sensor pairs above ``threshold`` (upper triangle)."""
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if abs(r) >= threshold:
                pairs.append({"a": cols[i], "b": cols[j], "pearson": round(r, 4)})
    return sorted(pairs, key=lambda d: -abs(d["pearson"]))


def _regime_key(df: pd.DataFrame) -> pd.Series:
    """Discretize operating regimes by rounding the 3 settings.

    FD002/FD004 vary setting1/setting2 across 6 flight profiles and setting3
    between two levels (full ~100 vs descent ~39). Rounding collapses noise.
    """
    s1 = df[OPS_COLS[0]].round(1)
    s2 = df[OPS_COLS[1]].round(2)
    s3 = df[OPS_COLS[2]].round(0)
    return s1.astype(str) + "|" + s2.astype(str) + "|" + s3.astype(str)


def operating_condition_analysis(df: pd.DataFrame) -> dict:
    """§10 operating-regime detection and per-regime sensor spread."""
    key = _regime_key(df)
    counts = key.value_counts()
    n_regimes = int(counts.shape[0])
    # variance explained of one representative sensor by regime
    sens_var_by_regime = {}
    for c in ["sensor_02", "sensor_17", "sensor_21"]:
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=float)
        total = float(np.var(v, ddof=1))
        groups = df.groupby(key.to_numpy())[c].mean()
        grp_n = df.groupby(key.to_numpy()).size()
        grand = float(np.mean(v))
        between = float(np.sum(grp_n * (groups - grand) ** 2) / max(len(v) - 1, 1))
        sens_var_by_regime[c] = {
            "total_var": round(total, 4),
            "between_regime_var": round(between, 4),
            "frac_between": round(between / total, 4) if total > 0 else 0.0,
        }
    return {
        "n_unique_regimes": n_regimes,
        "regime_row_counts": {k: int(v) for k, v in counts.items()},
        "setting_ranges": {
            OPS_COLS[i]: [float(df[OPS_COLS[i]].min()), float(df[OPS_COLS[i]].max())]
            for i in range(3)
        },
        "sensor_variance_split_by_regime": sens_var_by_regime,
    }


def degradation_and_rul(train: pd.DataFrame) -> dict:
    """§11 + §12 degradation and RUL diagnostics for a train frame."""
    rul = derive_train_rul(train)
    # per-sensor correlation with cycle (degradation trend) and with RUL
    cycle = train[CYCLE].to_numpy(dtype=float)
    trend = {}
    for c in SENSORS:
        v = train[c].to_numpy(dtype=float)
        rc = float(np.corrcoef(v, cycle)[0, 1]) if np.std(v) > 0 else 0.0
        trend[c] = {
            "corr_with_cycle": round(rc, 4),
            "abs_corr_with_cycle": round(abs(rc), 4),
        }
    # monotonicity: fraction of engines whose sensor moves consistently with cycle
    mono = {}
    for c in SENSORS:
        signs = []
        for _, g in train.groupby(UNIT_ID, sort=False):
            v = g.sort_values(CYCLE)[c].to_numpy(dtype=float)
            if len(v) > 2 and np.std(v) > 0:
                d = np.diff(v)
                frac = float(np.mean(np.sign(d) == np.sign(v[-1] - v[0])))
                signs.append(frac)
        mono[c] = round(float(np.mean(signs)), 4) if signs else 0.0
    return {
        "n_engines": int(train[UNIT_ID].nunique()),
        "rul_min": int(rul.min()),
        "rul_max": int(rul.max()),
        "rul_mean": round(float(rul.mean()), 3),
        "rul_median": float(np.median(rul.to_numpy())),
        "every_engine_ends_at_zero_rul": bool(
            all(int(derive_train_rul(g).iloc[-1]) == 0 for _, g in train.groupby(UNIT_ID, sort=False))
        ),
        "sensor_cycle_trend": trend,
        "sensor_monotonicity": mono,
    }


def target_horizon_simulation(train: pd.DataFrame, horizons: list[int]) -> dict:
    """§15 failure-risk target balance for candidate horizons (in cycles)."""
    rul = derive_train_rul(train)
    work = train[[UNIT_ID]].assign(_rul=rul.to_numpy())
    n_rows = len(rul)
    engine_lives = work.groupby(UNIT_ID).size()
    min_life = int(engine_lives.min())
    max_life = int(engine_lives.max())
    out = {}
    for h in horizons:
        pos = int((rul.to_numpy() <= h).sum())
        neg = n_rows - pos
        ppw = work[work["_rul"] <= h].groupby(UNIT_ID).size()
        ppw = ppw.reindex(work[UNIT_ID].unique(), fill_value=0)
        out[h] = {
            "positives": pos,
            "negatives": neg,
            "positive_rate": round(pos / n_rows, 4),
            "balance_ratio_1_to_n": round(neg / pos, 3) if pos else None,
            "min_pos_windows_per_engine": int(ppw.min()),
            "max_pos_windows_per_engine": int(min(ppw.max(), h + 1)),
            "engines_with_zero_positives": int((ppw == 0).sum()),
            "engine_min_life": min_life,
            "engine_max_life": max_life,
            "horizon_shorter_than_min_life": bool(h < min_life),
            "uses_truncated_label": bool(h < max_life),
        }
    return out
