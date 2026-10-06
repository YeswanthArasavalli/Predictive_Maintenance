"""Causal temporal (trajectory) feature construction for Phase 2.5.

Phase 2.5 asks a single scientific question:

    Does trajectory information beyond the current observation + engine age
    (cycle) materially improve RUL estimation and failure-risk prediction?

To answer it honestly, every temporal feature added here MUST be *causal*:
for a row at operational cycle ``t`` of an engine, the feature may depend only
on that same engine's observations at cycles ``<= t``. It may never depend on
a future cycle, on another engine, or on any validation/test information.

Design guarantees that make causality explicit and testable
-----------------------------------------------------------
1. **Engine isolation.** All rolling / difference / slope computations are
   performed inside a ``groupby(unit_id)``. A window physically cannot cross an
   engine boundary because each engine's series is processed independently.
2. **Backward-only windows.** Every ``rolling`` call uses ``min_periods`` and
   the default ``closed='right'`` orientation: the window at position ``i``
   covers rows ``[i-w+1, i]`` — the current row plus strictly-past rows only.
   ``shift`` / ``diff`` move *forward* in time (``x[t] - x[t-k]``), never
   backward.
3. **No centered / no future statistics.** There is no ``center=True``, no
   reversed series, and no whole-engine aggregate (mean/max/std over the full
   trajectory). Any statistic that would require a future observation is not
   computed here.
4. **Deterministic early-cycle policy** (spec section 13). When fewer than the
   requested window of history exists at the start of an engine:

     - ``rolling_mean`` uses ``min_periods=1`` — the shrinking expanding mean of
       whatever causal history exists.
     - ``rolling_std`` uses ``min_periods=2`` and is filled with ``0.0`` before
       two observations exist (no variability estimate is available).
     - ``rolling_min`` / ``rolling_max`` use ``min_periods=1``.
     - ``diff`` (first difference) and ``delta`` (lagged difference) are filled
       with ``0.0`` at the engine's first row(s) (no earlier value exists).
     - ``slope`` uses ``min_periods=2`` and is filled with ``0.0`` before two
       observations exist (a trend cannot be estimated from a single point).

   All fills are ``0.0`` (a neutral "no change / no spread / no slope"), never a
   forward or whole-series value.

5. **Normalization independence.** The caller supplies the frame whose sensor /
   operational columns are ALREADY normalized with a train-fitted scaler. This
   module only combines already-causal per-row values within an engine's past
   window, so it inherits both causality and the train-only leakage guarantee.

Nothing in this module reads targets (``raw_RUL`` / ``model_RUL_target`` /
``failure_risk_target_*``) or the official test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.loader import CYCLE, UNIT_ID

# ---------------------------------------------------------------------------
# Feature families (spec section 5). A Phase 2.5 experiment selects one family
# (or "combined"). These names are the audited vocabulary used across the
# registry, gate, and report.
# ---------------------------------------------------------------------------
FAMILY_LAG_DIFF = "lag_diff"      # first difference + lagged delta   (families C, E)
FAMILY_ROLLING = "rolling"        # rolling mean + rolling std        (families A, B)
FAMILY_SLOPE = "slope"            # causal recent linear slope        (family D)
FAMILY_COMBINED = "combined"      # all of the above (conservative set)
FAMILY_NONE = "none"              # current-row only (Phase 2 anchor)

VALID_FAMILIES: set[str] = {
    FAMILY_NONE, FAMILY_LAG_DIFF, FAMILY_ROLLING, FAMILY_SLOPE, FAMILY_COMBINED,
}

# Rolling min/max (family F) is folded into the ROLLING family but kept behind a
# flag so the primary experiment stays conservative (no feature explosion).
_SUFFIX = {
    "rmean": "rmean",
    "rstd": "rstd",
    "d1": "d1",
    "dk": "dk",
    "slope": "slope",
    "rmin": "rmin",
    "rmax": "rmax",
}


@dataclass
class TemporalConfig:
    """Parameters that fully determine the temporal feature set.

    Attributes
    ----------
    family : str
        One of :data:`VALID_FAMILIES`.
    windows : list[int]
        Rolling / slope window sizes. For conservative feature counts the
        primary configuration uses a single representative window.
    lags : list[int]
        Lagged-delta offsets ``k`` for ``x[t] - x[t-k]``.
    include_range : bool
        If True, add rolling min/max (family F) inside the ROLLING family.
    source_cols : list[str]
        The (already normalized) sensor / setting columns to build from.
    """

    family: str = FAMILY_NONE
    windows: tuple[int, ...] = (5,)
    lags: tuple[int, ...] = (5,)
    include_range: bool = False
    source_cols: tuple[str, ...] = ()
    # Selected degradation-monitor sensors (train-only subset of source_cols that
    # actually receive temporal features). Empty => use all source_cols.
    monitor_sensors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in VALID_FAMILIES:
            raise ValueError(f"Unknown temporal family {self.family!r}; valid: {sorted(VALID_FAMILIES)}")
        self.windows = tuple(int(w) for w in self.windows)
        self.lags = tuple(int(k) for k in self.lags)
        self.source_cols = tuple(self.source_cols)
        self.monitor_sensors = tuple(self.monitor_sensors) or self.source_cols
        for w in self.windows:
            if w < 2:
                raise ValueError(f"Temporal window must be >= 2, got {w}")
        for k in self.lags:
            if k < 1:
                raise ValueError(f"Temporal lag must be >= 1, got {k}")

    @property
    def active_sources(self) -> tuple[str, ...]:
        return self.monitor_sensors

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "windows": list(self.windows),
            "lags": list(self.lags),
            "include_range": self.include_range,
            "source_cols": list(self.source_cols),
            "monitor_sensors": list(self.monitor_sensors),
        }


def _families_for(cfg: TemporalConfig) -> dict[str, bool]:
    fam = cfg.family
    return {
        "rmean": fam in (FAMILY_ROLLING, FAMILY_COMBINED),
        "rstd": fam in (FAMILY_ROLLING, FAMILY_COMBINED),
        "rminmax": fam in (FAMILY_ROLLING, FAMILY_COMBINED) and cfg.include_range,
        "d1": fam in (FAMILY_LAG_DIFF, FAMILY_COMBINED),
        "dk": fam in (FAMILY_LAG_DIFF, FAMILY_COMBINED),
        "slope": fam in (FAMILY_SLOPE, FAMILY_COMBINED),
    }


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal least-squares slope of ``arr`` versus within-engine position.

    For each index ``t`` the slope is fitted on positions
    ``[max(0, t-window+1), t]`` against the row index (1, 2, 3, ...). The
    computation uses cumulative sums and is therefore O(n) and exactly
    reproducible. Rows with fewer than two usable observations are ``0.0``.
    """
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out
    idx = np.arange(n, dtype=np.float64)
    # cumulative sums for x, i*x, i, i^2 with a leading zero for easy ranges
    c_x = np.concatenate(([0.0], np.cumsum(arr)))
    c_ix = np.concatenate(([0.0], np.cumsum(idx * arr)))
    c_i = np.concatenate(([0.0], np.cumsum(idx)))
    c_i2 = np.concatenate(([0.0], np.cumsum(idx * idx)))

    def _rng_sum(c: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        # Inclusive sum over positions [lo, hi] using prefix array c where
        # c[0] = 0 and c[k] = sum of the first k elements.
        return c[hi + 1] - c[lo]

    lo = np.maximum(0, idx - window + 1).astype(np.int64)
    hi = np.arange(n, dtype=np.int64)
    cnt = (hi - lo + 1).astype(np.float64)
    s_x = _rng_sum(c_x, lo, hi)
    s_ix = _rng_sum(c_ix, lo, hi)
    s_i = _rng_sum(c_i, lo, hi)
    s_i2 = _rng_sum(c_i2, lo, hi)
    denom = cnt * s_i2 - s_i * s_i
    numer = cnt * s_ix - s_i * s_x
    valid = (cnt >= 2) & (np.abs(denom) > 1e-12)
    out[valid] = numer[valid] / denom[valid]
    return out


def temporal_feature_names(cfg: TemporalConfig) -> list[str]:
    """Return the deterministic ordered list of temporal columns for ``cfg``.

    Names follow ``<source>__<kind><param>`` (e.g. ``sensor_14__rmean5``), which
    the leakage gate in :mod:`src.models.features` treats as a derived sensor
    feature. This function has NO dataframe dependency so the gate/tests can
    verify naming without running the pipeline.
    """
    if cfg.family == FAMILY_NONE:
        return []
    flags = _families_for(cfg)
    names: list[str] = []
    for src in cfg.active_sources:
        if flags["d1"]:
            names.append(f"{src}__d1")
        if flags["dk"]:
            for k in cfg.lags:
                names.append(f"{src}__dk{k}")
        if flags["rmean"]:
            for w in cfg.windows:
                names.append(f"{src}__rmean{w}")
        if flags["rstd"]:
            for w in cfg.windows:
                names.append(f"{src}__rstd{w}")
        if flags["rminmax"]:
            for w in cfg.windows:
                names.append(f"{src}__rmin{w}")
                names.append(f"{src}__rmax{w}")
        if flags["slope"]:
            for w in cfg.windows:
                names.append(f"{src}__slope{w}")
    return names


def add_causal_temporal_features(
    df: pd.DataFrame,
    cfg: TemporalConfig,
) -> pd.DataFrame:
    """Return a copy of ``df`` with causal temporal columns appended.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``unit_id``, ``cycle`` and the configured ``active_sources``
        sensor/setting columns (already normalized by the caller). Rows may be
        in any order; output row order matches the input.
    cfg : TemporalConfig
        The family / window / lag configuration.

    Returns
    -------
    pd.DataFrame
        ``df`` plus the columns named by :func:`temporal_feature_names`. For
        ``family == "none"`` the frame is returned unchanged (copy).
    """
    if df[UNIT_ID].isna().any() or df[CYCLE].isna().any():
        raise ValueError("Temporal builder requires non-null unit_id and cycle.")

    out = df.copy()
    names = temporal_feature_names(cfg)
    if not names:
        return out

    flags = _families_for(cfg)
    # Establish a canonical, causal (time-ascending) order per engine.
    order = out.sort_values([UNIT_ID, CYCLE], kind="mergesort")
    uid = order[UNIT_ID]
    grp = order.groupby(UNIT_ID, sort=False)

    # Pre-allocate a per-name buffer in time-sorted (``order``) row order.
    new_cols: dict[str, np.ndarray] = {n: np.zeros(len(order), dtype=np.float64) for n in names}

    def _set(name: str, values: np.ndarray) -> None:
        new_cols[name] = np.asarray(values, dtype=np.float64)

    for src in cfg.active_sources:
        series = grp[src]
        if flags["d1"]:
            d1 = series.diff(1).to_numpy()
            _set(f"{src}__d1", np.nan_to_num(d1, nan=0.0))
        if flags["dk"]:
            for k in cfg.lags:
                dk = series.diff(int(k)).to_numpy()
                _set(f"{src}__dk{k}", np.nan_to_num(dk, nan=0.0))
        if flags["rmean"]:
            for w in cfg.windows:
                rm = grp[src].rolling(int(w), min_periods=1).mean()
                _set(f"{src}__rmean{w}", rm.to_numpy())
        if flags["rstd"]:
            for w in cfg.windows:
                rs = grp[src].rolling(int(w), min_periods=2).std()
                _set(f"{src}__rstd{w}", np.nan_to_num(rs.to_numpy(), nan=0.0))
        if flags["rminmax"]:
            for w in cfg.windows:
                rmin = grp[src].rolling(int(w), min_periods=1).min()
                rmax = grp[src].rolling(int(w), min_periods=1).max()
                _set(f"{src}__rmin{w}", rmin.to_numpy())
                _set(f"{src}__rmax{w}", rmax.to_numpy())
        if flags["slope"]:
            # Per-engine causal slope, accumulated into one buffer per window so
            # that engines never overwrite each other's positions.
            slope_buf = {w: np.zeros(len(order), dtype=np.float64) for w in cfg.windows}
            for _, gidx in grp.groups.items():
                pos = order.index.get_indexer(gidx)
                x = order.loc[gidx, src].to_numpy(dtype=np.float64)
                for w in cfg.windows:
                    slope_buf[w][pos] = _rolling_slope(x, int(w))
            for w in cfg.windows:
                _set(f"{src}__slope{w}", slope_buf[w])

    for name in names:
        # new_cols are in time-sorted order; map back to the original row order.
        restored = pd.Series(new_cols[name], index=order.index).reindex(out.index)
        out[name] = restored.to_numpy(dtype=np.float64)

    return out


@dataclass
class TemporalBuild:
    """Small helper bundling a config and its produced column names."""

    config: TemporalConfig
    feature_names: list[str] = field(default_factory=list)
