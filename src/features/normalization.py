"""Train-only normalization: global (Mode A) and regime-conditioned (Mode B).

Leakage controls enforced here:
- Scaler fit() ONLY sees training-engine rows.
- Validation and test data are ONLY ever transformed.
- No global statistics (mean/std) are computed over the full dataset before
  the engine split.
- Regime scalers for Mode B are fit independently per regime using only
  training-engine rows that belong to that regime.

Causality note:
- The scaler parameters are "deployment-known constants learned historically"
  (from the training partition). They are NOT computed from the future
  trajectory of the same engine being scored. A validation or test engine's
  observations are transformed using the train-fitted parameters only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class GlobalScaler:
    """Mode A: single z-score scaler fit on ALL training-engine rows.

    Attributes
    ----------
    scaler : sklearn StandardScaler fitted on training data.
    feature_cols : list[str]
        The sensor (and optionally operational-setting) columns that
        were scaled.
    """

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.feature_cols: list[str] | None = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "GlobalScaler":
        """Fit the scaler using training-engine rows only.

        Parameters
        ----------
        train_df: pd.DataFrame
            Rows belonging to training engines (MUST not contain val/test engines).
        feature_cols: list[str]
            Column names to scale (typically sensors + operational settings).
        """
        self.feature_cols = list(feature_cols)
        X = train_df[self.feature_cols].to_numpy(dtype=np.float64)
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform df using the train-fitted parameters.

        Returns a DataFrame with the same index as df but scaled values.
        """
        if self.scaler is None or self.feature_cols is None:
            raise RuntimeError("GlobalScaler.fit() must be called before transform().")
        result = df.copy()
        X = df[self.feature_cols].to_numpy(dtype=np.float64)
        result[self.feature_cols] = self.scaler.transform(X)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted parameters for the manifest / artifact store."""
        if self.scaler is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "mode": "global",
            "feature_cols": self.feature_cols,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
            "n_samples_fit": int(self.scaler.n_samples_seen_),
        }


class RegimeScaler:
    """Mode B: per-regime z-score scalers, each fit on train-engine rows of that regime.

    At transform time, each row is routed to its regime's scaler. If a row's
    regime is unseen in training, it is handled according to the `fallback` policy.

    Parameters
    ----------
    fallback : str
        "error" (default) — raise on unseen regime;
        "global" — fall back to a global scaler fit on all train rows.
    """

    def __init__(self, fallback: str = "error") -> None:
        self.fallback = fallback
        self.scalers: dict[int, StandardScaler] = {}
        self.global_scaler: StandardScaler | None = None  # used if fallback="global"
        self.feature_cols: list[str] | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        regime_col: str = "operating_regime",
    ) -> "RegimeScaler":
        """Fit per-regime scalers using training-engine rows only.

        Parameters
        ----------
        train_df: pd.DataFrame
            Rows of training engines ONLY.
        feature_cols: list[str]
            Columns to scale.
        regime_col: str
            Name of the integer regime column in train_df.
        """
        self.feature_cols = list(feature_cols)

        for regime_id, grp in train_df.groupby(regime_col):
            X = grp[self.feature_cols].to_numpy(dtype=np.float64)
            sc = StandardScaler()
            sc.fit(X)
            self.scalers[int(regime_id)] = sc

        if self.fallback == "global":
            X_all = train_df[self.feature_cols].to_numpy(dtype=np.float64)
            self.global_scaler = StandardScaler()
            self.global_scaler.fit(X_all)

        return self

    def transform(self, df: pd.DataFrame, regime_col: str = "operating_regime") -> pd.DataFrame:
        """Route each row to its regime's train-fitted scaler and transform.

        Parameters
        ----------
        df: pd.DataFrame
            Any DataFrame with the regime_col and feature columns.
        regime_col: str
            Column holding integer regime IDs.
        """
        if not self.scalers or self.feature_cols is None:
            raise RuntimeError("RegimeScaler.fit() must be called before transform().")

        result = df.copy()
        X = np.empty((len(df), len(self.feature_cols)), dtype=np.float64)

        for regime_id, grp in df.groupby(regime_col):
            rid = int(regime_id)
            idx = grp.index
            if rid in self.scalers:
                X_pos = self.scalers[rid].transform(
                    grp[self.feature_cols].to_numpy(dtype=np.float64)
                )
            elif self.fallback == "global" and self.global_scaler is not None:
                X_pos = self.global_scaler.transform(
                    grp[self.feature_cols].to_numpy(dtype=np.float64)
                )
            else:
                raise ValueError(
                    f"Unseen regime {rid} at transform time and fallback={self.fallback!r}."
                )
            # We need positional indices for the X array
            pos = df.index.get_indexer(idx)
            X[pos] = X_pos

        result[self.feature_cols] = X
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fitted parameters."""
        out: dict[str, Any] = {
            "fitted": bool(self.scalers),
            "mode": "regime_conditioned",
            "feature_cols": self.feature_cols,
            "fallback": self.fallback,
            "regimes": {},
        }
        for rid, sc in sorted(self.scalers.items()):
            out["regimes"][str(rid)] = {
                "mean": sc.mean_.tolist(),
                "scale": sc.scale_.tolist(),
                "var": sc.var_.tolist(),
                "n_samples_fit": int(sc.n_samples_seen_),
            }
        return out
