"""Operating-regime clustering and within-regime sensor normalization."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import INFORMATIVE_SENSORS, OP_SETTING_COLS, RANDOM_SEED


class RegimeNormalizer:
    """Cluster operational settings into regimes and Z-score sensors within each regime.

    Fit strictly on the training split to prevent data leakage.
    """

    def __init__(
        self,
        n_regimes: int = 6,
        sensor_cols: list[str] | None = None,
        op_cols: list[str] | None = None,
        use_regime_norm: bool = True,
        random_state: int = RANDOM_SEED,
    ) -> None:
        if n_regimes < 1:
            raise ValueError(f"n_regimes must be >= 1, got {n_regimes}.")
        self.n_regimes = int(n_regimes)
        self.sensor_cols = (
            list(sensor_cols) if sensor_cols is not None else list(INFORMATIVE_SENSORS)
        )
        self.op_cols = list(op_cols) if op_cols is not None else list(OP_SETTING_COLS)
        self.use_regime_norm = bool(use_regime_norm)
        self.random_state = int(random_state)

        self.op_scaler_: StandardScaler | None = None
        self.kmeans_: KMeans | None = None
        self.global_mean_: np.ndarray | None = None
        self.global_std_: np.ndarray | None = None
        self.regime_means_: dict[int, np.ndarray] = {}
        self.regime_stds_: dict[int, np.ndarray] = {}
        self.is_fitted_: bool = False

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing_ops = [c for c in self.op_cols if c not in df.columns]
        missing_sensors = [c for c in self.sensor_cols if c not in df.columns]
        if missing_ops or missing_sensors:
            raise KeyError(
                f"Missing required columns. op_cols={missing_ops}, sensor_cols={missing_sensors}"
            )

    def fit(self, train_df: pd.DataFrame) -> "RegimeNormalizer":
        """Fit operational scaler, KMeans regime clusters, and sensor normalization stats."""
        self._validate_columns(train_df)
        if len(train_df) == 0:
            raise ValueError("Cannot fit RegimeNormalizer on an empty DataFrame.")

        op_vals = train_df[self.op_cols].to_numpy(dtype=np.float64)
        self.op_scaler_ = StandardScaler().fit(op_vals)
        scaled_ops = self.op_scaler_.transform(op_vals)

        if self.n_regimes > 1:
            self.kmeans_ = KMeans(
                n_clusters=self.n_regimes,
                n_init=20,
                random_state=self.random_state,
            ).fit(scaled_ops)
            regimes = self.kmeans_.labels_.astype(np.int64)
        else:
            self.kmeans_ = None
            regimes = np.zeros(len(train_df), dtype=np.int64)

        sensor_vals = train_df[self.sensor_cols].to_numpy(dtype=np.float64)
        self.global_mean_ = np.mean(sensor_vals, axis=0)
        global_std = np.std(sensor_vals, axis=0, ddof=0)
        self.global_std_ = np.maximum(global_std, 1e-6)

        self.regime_means_ = {}
        self.regime_stds_ = {}
        if self.use_regime_norm and self.n_regimes > 1:
            for r in range(self.n_regimes):
                mask = regimes == r
                count = int(mask.sum())
                if count > 0:
                    mu_r = np.mean(sensor_vals[mask], axis=0)
                    if count > 1:
                        sd_r = np.maximum(
                            np.std(sensor_vals[mask], axis=0, ddof=0), 1e-6
                        )
                    else:
                        sd_r = self.global_std_.copy()
                else:
                    mu_r = self.global_mean_.copy()
                    sd_r = self.global_std_.copy()
                self.regime_means_[r] = mu_r
                self.regime_stds_[r] = sd_r

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign regimes, scale op_cols, and Z-score sensor_cols without modifying input."""
        if not self.is_fitted_:
            raise RuntimeError("RegimeNormalizer must be fitted before calling transform().")
        self._validate_columns(df)

        out = df.copy()
        if len(out) == 0:
            out["regime"] = pd.Series(dtype=np.int64, index=out.index)
            return out

        op_vals = out[self.op_cols].to_numpy(dtype=np.float64)
        scaled_ops = self.op_scaler_.transform(op_vals)
        for idx, col in enumerate(self.op_cols):
            out[col] = scaled_ops[:, idx]

        if self.kmeans_ is not None:
            regimes = self.kmeans_.predict(scaled_ops).astype(np.int64)
        else:
            regimes = np.zeros(len(out), dtype=np.int64)
        out["regime"] = regimes

        sensor_vals = out[self.sensor_cols].to_numpy(dtype=np.float64)
        if self.use_regime_norm and self.n_regimes > 1:
            norm_vals = np.empty_like(sensor_vals, dtype=np.float64)
            for r in np.unique(regimes):
                mask = regimes == r
                mu = self.regime_means_.get(int(r), self.global_mean_)
                sd = self.regime_stds_.get(int(r), self.global_std_)
                norm_vals[mask] = (sensor_vals[mask] - mu) / sd
        else:
            norm_vals = (sensor_vals - self.global_mean_) / self.global_std_

        for idx, col in enumerate(self.sensor_cols):
            out[col] = norm_vals[:, idx]

        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on train_df and return its transformed copy."""
        return self.fit(train_df).transform(train_df)

    def save(self, path: str | Path) -> Path:
        """Persist fitted normalizer to disk via joblib."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "RegimeNormalizer":
        """Load a persisted RegimeNormalizer from disk."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}.")
        return obj
