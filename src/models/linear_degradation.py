"""Causal per-engine linear degradation baseline (Phase 2 section 6).

This baseline simulates a classical prognostics pipeline:

    sensor/health trajectory -> linear degradation trend -> estimated
    failure cycle -> estimated RUL

CAUSALITY REQUIREMENT
    For each prediction row (engine u, cycle t), the trend is fit using
    only the observations of engine u at cycles 1..t. No future observation
    of engine u (or any other engine) is used at predict time. The failure
    threshold HI_f is a train-partition constant computed once at fit()
    from each training engine's HI at its own last observed cycle. This is
    a "deployment-known constant learned historically" and is NOT derived
    from any validation or test trajectory.

HEALTH INDEX (HI)
    HI_t = mean of the regime-normalized values of a small set of sensors
    that show the strongest per-engine monotonic degradation in FD004
    (default: sensor_14, sensor_17, sensor_11; see Phase 0 sensor_analysis
    and degradation_rul audits). The sensors must already be normalized in
    the Mode-B (regime-conditioned) space by the caller, because regime
    effects dominate raw sensor variance in FD004.

TREND FIT
    Ordinary least squares of HI against the per-engine time index
    (0-based cycle count within that engine's history, NOT the raw `cycle`
    column). OLS on 2 arrays: slope, intercept. Deterministic, no RNG.

FAILURE CROSSING
    Direction of degradation in HI-space is learned at fit time from the
    TRAIN partition: `direction = sign(median_HI_at_last_cycle -
    median_HI_at_first_cycle)`. HI values are then sign-flipped by
    `direction` so that degradation always makes the adjusted HI increase.
    If slope > slope_floor, extrapolate: t_star = (adjusted_HI_f - intercept) / slope
    Predicted failure time = t_star. Predicted RUL = max(0, t_star - t_now).

    If slope <= slope_floor (engine is not degrading in this HI), fall back
    to the training-set mean RUL for that prediction.

MIN_HISTORY
    At least `min_history` observations are needed to fit a trend. Earlier
    rows use the same train-mean-RUL fallback.

Complexity: O(N) in rows. Implementation avoids per-row numpy ops where
possible by using closed-form slope/intercept on cumulative sums.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.loader import CYCLE, UNIT_ID

DEFAULT_HI_SENSORS = ("sensor_14", "sensor_17", "sensor_11")


@dataclass
class LinearDegradationRUL:
    """Causal per-engine OLS extrapolation to a train-derived failure threshold.

    Fit-time constants:
        direction            +1 or -1, learned from train HI first-vs-last medians
        failure_threshold    median adjusted HI at train engines' last cycle
        fallback_rul         mean raw_RUL over train rows
        rul_cap              max raw_RUL observed in train (physical ceiling for
                             extrapolation; without it a shallow slope can
                             project a failure cycle thousands of steps away)
    """

    hi_sensors: tuple[str, ...] = DEFAULT_HI_SENSORS
    min_history: int = 15
    slope_floor: float = 1e-6
    # Fit-time constants
    direction: float = 1.0
    failure_threshold: float | None = None
    fallback_rul: float | None = None
    rul_cap: float | None = None
    # Diagnostic
    _fit_stats: dict[str, float] | None = None

    def fit(self, df_train_scaled: pd.DataFrame, *, target_col: str = "raw_RUL") -> "LinearDegradationRUL":
        """Compute HI direction, failure threshold, and fallback RUL constant.

        Parameters
        ----------
        df_train_scaled : pd.DataFrame
            Training partition, sensors already Mode-B normalized. Must
            contain the HI sensor columns, unit_id, cycle, and target_col.
        """
        missing = [c for c in self.hi_sensors if c not in df_train_scaled.columns]
        if missing:
            raise ValueError(f"LinearDegradationRUL.fit: HI sensors missing from df: {missing}")

        # Sort by engine and cycle so "first/last row" per engine is correct.
        df = df_train_scaled.sort_values([UNIT_ID, CYCLE])
        hi_all = df[list(self.hi_sensors)].to_numpy(dtype=np.float64).mean(axis=1)
        df_hi = df[[UNIT_ID, CYCLE]].copy()
        df_hi["_hi"] = hi_all

        grouped = df_hi.groupby(UNIT_ID, sort=True)
        hi_first = grouped["_hi"].first().to_numpy()
        hi_last = grouped["_hi"].last().to_numpy()

        med_first = float(np.median(hi_first))
        med_last = float(np.median(hi_last))
        # Direction that makes degradation increase the adjusted HI.
        # If HI does not shift measurably between first and last cycle we
        # default to +1 (this makes the fallback cover all engines honestly).
        self.direction = 1.0 if med_last >= med_first else -1.0
        # Failure threshold in adjusted space.
        self.failure_threshold = float(self.direction * med_last)

        # Fallback = train mean RUL
        self.fallback_rul = float(df_train_scaled[target_col].mean())
        # Physical ceiling on RUL extrapolation: the largest RUL ever observed
        # in the training partition. Prevents shallow-slope blow-up.
        self.rul_cap = float(df_train_scaled[target_col].max())

        self._fit_stats = {
            "direction": self.direction,
            "median_hi_first": med_first,
            "median_hi_last": med_last,
            "failure_threshold": self.failure_threshold,
            "fallback_rul": self.fallback_rul,
            "rul_cap": self.rul_cap,
            "n_train_engines": int(df_hi[UNIT_ID].nunique()),
        }
        return self

    def predict(self, df_val_scaled: pd.DataFrame) -> np.ndarray:
        """Predict RUL row-by-row using only each engine's history up to t."""
        if self.failure_threshold is None or self.fallback_rul is None:
            raise RuntimeError("LinearDegradationRUL.fit() must be called before predict().")

        missing = [c for c in self.hi_sensors if c not in df_val_scaled.columns]
        if missing:
            raise ValueError(f"LinearDegradationRUL.predict: HI sensors missing from df: {missing}")

        df = df_val_scaled.sort_values([UNIT_ID, CYCLE]).reset_index()
        orig_idx = df["index"].to_numpy()
        hi = df[list(self.hi_sensors)].to_numpy(dtype=np.float64).mean(axis=1)
        unit_ids = df[UNIT_ID].to_numpy()

        preds = np.full(len(df), self.fallback_rul, dtype=np.float64)

        # Iterate per engine; for each engine use cumulative sums to fit OLS
        # on the running history [0..t] for every t >= min_history - 1.
        for uid, positions in _groupby_sorted_indices(unit_ids):
            hi_e_raw = hi[positions]
            n = len(hi_e_raw)
            if n == 0:
                continue
            # Adjust HI sign so degradation always increases the adjusted HI.
            hi_e = self.direction * hi_e_raw
            # Time index within the engine's own history (0, 1, ..., n-1).
            t = np.arange(n, dtype=np.float64)

            # Cumulative sums for OLS on prefix [0..t]:
            #   sum(1)   = k+1
            #   sum(t)   = k(k+1)/2
            #   sum(t^2) = k(k+1)(2k+1)/6
            #   sum(hi)  = cumsum(hi)[k]
            #   sum(t*hi)= cumsum(t*hi)[k]
            k = t  # last index included in prefix (inclusive)
            cnt = k + 1.0
            sum_t = k * (k + 1.0) / 2.0
            sum_tt = k * (k + 1.0) * (2.0 * k + 1.0) / 6.0
            sum_h = np.cumsum(hi_e)
            sum_th = np.cumsum(t * hi_e)

            denom = cnt * sum_tt - sum_t * sum_t
            # OLS slope and intercept for each prefix
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = (cnt * sum_th - sum_t * sum_h) / denom
                intercept = (sum_h - slope * sum_t) / cnt

            # Only use prefixes with enough history and a strictly positive
            # adjusted slope (engine is degrading in adjusted HI space).
            valid = (k >= (self.min_history - 1)) & np.isfinite(slope) & (slope > self.slope_floor)
            t_star = np.full(n, np.nan)
            t_star[valid] = (self.failure_threshold - intercept[valid]) / slope[valid]

            # Predicted RUL at each row's own time t
            rul_hat = t_star - t
            rul_hat = np.where(np.isfinite(rul_hat), np.maximum(rul_hat, 0.0), self.fallback_rul)
            # Cap extrapolation at the physical ceiling seen in training.
            if self.rul_cap is not None:
                rul_hat = np.minimum(rul_hat, self.rul_cap)

            # Where valid is False, use fallback (matches the array init).
            rul_hat[~valid] = self.fallback_rul

            preds[positions] = rul_hat

        # Restore original row ordering (the caller passed an unsorted df)
        out = np.empty(len(df), dtype=np.float64)
        out[orig_idx] = preds
        return out

    def describe(self) -> dict[str, float]:
        if self._fit_stats is None:
            return {}
        return dict(self._fit_stats)


def _groupby_sorted_indices(keys: np.ndarray) -> list[tuple[object, np.ndarray]]:
    """Yield (unique_key, positions_array) sorted by key.

    Equivalent to pandas groupby(sort=True) but cheaper for our inner loop
    and preserves insertion order in the returned positions arrays.
    """
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    # Find boundaries
    unique, starts = np.unique(sorted_keys, return_index=True)
    ends = np.append(starts[1:], len(sorted_keys))
    return [
        (u, order[s:e])
        for u, s, e in zip(unique, starts, ends)
    ]
