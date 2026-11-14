"""Feature engineering, latent fault-mode clustering, and causal sequence windowing."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

from src.config import (
    INFORMATIVE_SENSORS,
    OP_SETTING_COLS,
    RANDOM_SEED,
    ROLLING_WINDOWS,
    WINDOW_SIZE,
)


def _group_cols_for(df: pd.DataFrame) -> list[str]:
    if "dataset_id" in df.columns:
        return ["dataset_id", "unit_id"]
    return ["unit_id"]


def _ensure_sorted_work_df(df: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, bool]:
    """Return a RangeIndex DataFrame sorted by group_cols + ['cycle'] and a flag indicating reordering."""
    sort_cols = group_cols + ["cycle"]
    work = df.copy()
    work["_orig_pos"] = np.arange(len(work), dtype=np.int64)
    # Check whether rows are already grouped contiguously and cycle-ordered
    work_sorted = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    reordered = not np.array_equal(work_sorted["_orig_pos"].to_numpy(), work["_orig_pos"].to_numpy())
    return work_sorted, reordered


def add_tabular_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    include_regime_onehot: bool = True,
    n_regimes: int = 6,
) -> tuple[pd.DataFrame, list[str]]:
    """Build causal per-engine rolling stats, first differences, op settings, and regime one-hots.

    Never places ``RUL``, ``RUL_raw``, ``unit_id``, ``cycle``, or ``dataset_id`` in
    the returned ``feature_cols`` list.
    """
    if len(df) == 0:
        return df.copy(), []

    group_cols = _group_cols_for(df)
    work, reordered = _ensure_sorted_work_df(df, group_cols)

    new_cols: dict[str, np.ndarray] = {}
    feature_cols: list[str] = list(sensor_cols)

    grouped = work.groupby(group_cols, sort=False)
    for w in windows:
        roll = grouped[sensor_cols].rolling(window=int(w), min_periods=1)
        r_mean = (
            roll.mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
            .reindex(work.index)
        )
        r_std = (
            roll.std(ddof=0)
            .reset_index(level=list(range(len(group_cols))), drop=True)
            .reindex(work.index)
            .fillna(0.0)
        )
        for col in sensor_cols:
            m_name = f"{col}_roll_mean_{w}"
            s_name = f"{col}_roll_std_{w}"
            new_cols[m_name] = r_mean[col].to_numpy(dtype=np.float32)
            new_cols[s_name] = r_std[col].to_numpy(dtype=np.float32)
            feature_cols.extend([m_name, s_name])

    diffs = grouped[sensor_cols].diff(1).fillna(0.0)
    for col in sensor_cols:
        d_name = f"{col}_diff_1"
        new_cols[d_name] = diffs[col].to_numpy(dtype=np.float32)
        feature_cols.append(d_name)

    op_present = [c for c in OP_SETTING_COLS if c in work.columns]
    feature_cols.extend(op_present)

    if include_regime_onehot and "regime" in work.columns:
        reg_vals = work["regime"].to_numpy(dtype=np.int64)
        for r in range(int(n_regimes)):
            r_name = f"regime_{r}"
            new_cols[r_name] = (reg_vals == r).astype(np.float32)
            feature_cols.append(r_name)

    feat_df = pd.DataFrame(new_cols, index=work.index)
    if reordered:
        feat_df["_orig_pos"] = work["_orig_pos"].to_numpy()
        feat_df = feat_df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
        feat_df.index = df.index
    else:
        feat_df.index = df.index

    out = pd.concat([df.copy(), feat_df], axis=1)

    forbidden = {"RUL", "RUL_raw", "unit_id", "cycle", "dataset_id", "regime"}
    feature_cols = [c for c in feature_cols if c not in forbidden]
    return out, feature_cols


class FaultModeClusterer:
    """Unsupervised latent fault-mode clusterer without target leakage at test time.

    Fits KMeans on the mean normalized sensor profile over the final ``tail_cycles``
    of each training engine (where the fault signature is strongest). At transform time,
    uses a causal rolling mean of normalized sensors per engine and computes soft
    cluster probabilities via softmax of negative Euclidean distances to centroids.
    """

    def __init__(
        self,
        n_modes: int = 2,
        sensor_cols: list[str] | None = None,
        tail_cycles: int = 25,
        random_state: int = RANDOM_SEED,
    ) -> None:
        if n_modes < 2:
            raise ValueError(f"n_modes must be >= 2, got {n_modes}.")
        if tail_cycles < 1:
            raise ValueError(f"tail_cycles must be >= 1, got {tail_cycles}.")
        self.n_modes = int(n_modes)
        self.sensor_cols = (
            list(sensor_cols) if sensor_cols is not None else list(INFORMATIVE_SENSORS)
        )
        self.tail_cycles = int(tail_cycles)
        self.random_state = int(random_state)

        self.kmeans_: KMeans | None = None
        self.centroids_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> "FaultModeClusterer":
        """Fit KMeans centroids on final tail_cycles profiles of training engines."""
        missing = [c for c in self.sensor_cols if c not in train_df.columns]
        if missing:
            raise KeyError(f"Missing sensor_cols in train_df: {missing}")

        group_cols = _group_cols_for(train_df)
        work, _ = _ensure_sorted_work_df(train_df, group_cols)
        tail_means = (
            work.groupby(group_cols, sort=False)
            .tail(self.tail_cycles)
            .groupby(group_cols, sort=False)[self.sensor_cols]
            .mean()
        )
        if len(tail_means) < self.n_modes:
            raise ValueError(
                f"Need at least {self.n_modes} training engines, found {len(tail_means)}."
            )

        self.kmeans_ = KMeans(
            n_clusters=self.n_modes,
            n_init=20,
            random_state=self.random_state,
        ).fit(tail_means.to_numpy(dtype=np.float64))
        self.centroids_ = self.kmeans_.cluster_centers_.astype(np.float64)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Attach causal soft fault-mode probabilities and hard prediction to a copy of df."""
        if not self.is_fitted_ or self.centroids_ is None:
            raise RuntimeError("FaultModeClusterer must be fitted before transform().")

        prob_cols = [f"fault_mode_prob_{k}" for k in range(self.n_modes)]
        if len(df) == 0:
            out = df.copy()
            for col in prob_cols:
                out[col] = pd.Series(dtype=np.float32, index=out.index)
            out["fault_mode_pred"] = pd.Series(dtype=np.int64, index=out.index)
            return out, prob_cols

        group_cols = _group_cols_for(df)
        work, reordered = _ensure_sorted_work_df(df, group_cols)

        causal_mean = (
            work.groupby(group_cols, sort=False)[self.sensor_cols]
            .rolling(window=self.tail_cycles, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
            .reindex(work.index)
            .to_numpy(dtype=np.float64)
        )

        # Euclidean distance from causal profile to each fault-mode centroid
        dists = np.linalg.norm(
            causal_mean[:, None, :] - self.centroids_[None, :, :],
            axis=2,
        )
        logits = -dists
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1).astype(np.int64)

        new_cols: dict[str, np.ndarray] = {
            prob_cols[k]: probs[:, k].astype(np.float32) for k in range(self.n_modes)
        }
        new_cols["fault_mode_pred"] = preds

        fault_df = pd.DataFrame(new_cols, index=work.index)
        if reordered:
            fault_df["_orig_pos"] = work["_orig_pos"].to_numpy()
            fault_df = fault_df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
        fault_df.index = df.index

        out = pd.concat([df.copy(), fault_df], axis=1)
        return out, prob_cols

    def fit_transform(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Fit on train_df and transform it."""
        return self.fit(train_df).transform(train_df)

    def save(self, path: str | Path) -> Path:
        """Save clusterer via joblib."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "FaultModeClusterer":
        """Load clusterer from disk via joblib."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}.")
        return obj


def build_sequence_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = WINDOW_SIZE,
    target_col: str | None = "RUL",
    last_only: bool = False,
    stride: int = 1,
    min_history: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build causal sliding windows [N, window_size, F] with left-edge repeat padding.

    When ``last_only=True``, returns exactly one window per engine (for official
    C-MAPSS test scoring, left-padding short units such as FD002 min=21 / FD004 min=19).
    When ``target_col`` is absent (e.g. test set), ``y`` is filled with ``np.nan``.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}.")
    if stride < 1 or min_history < 1:
        raise ValueError("stride and min_history must be >= 1.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature_cols in df: {missing}")

    has_dataset_id = "dataset_id" in df.columns
    group_cols = ["dataset_id", "unit_id"] if has_dataset_id else ["unit_id"]
    meta_cols = ["dataset_id", "unit_id", "cycle"] if has_dataset_id else ["unit_id", "cycle"]

    n_features = len(feature_cols)
    if len(df) == 0:
        empty_meta = pd.DataFrame(columns=meta_cols)
        return (
            np.empty((0, window_size, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            empty_meta,
        )

    has_target = target_col is not None and target_col in df.columns

    # Preserve input unit appearance order while sorting cycles inside each unit
    work = df.sort_values(group_cols + ["cycle"], kind="mergesort") if not df.groupby(
        group_cols, sort=False
    )["cycle"].is_monotonic_increasing.all() else df

    group_slices: list[tuple[pd.DataFrame, int]] = []
    total_windows = 0
    for _, grp in work.groupby(group_cols, sort=False):
        t_len = len(grp)
        if last_only:
            n_win = 1
        else:
            start = min_history - 1
            if start >= t_len:
                n_win = 0
            else:
                n_win = ((t_len - 1 - start) // stride) + 1
        if n_win > 0:
            group_slices.append((grp, n_win))
            total_windows += n_win

    X = np.empty((total_windows, window_size, n_features), dtype=np.float32)
    y = np.full((total_windows,), np.nan, dtype=np.float32)
    meta_records: list[pd.DataFrame] = []

    cursor = 0
    for grp, n_win in group_slices:
        arr = grp[feature_cols].to_numpy(dtype=np.float32)
        t_len = len(arr)

        if last_only:
            if t_len >= window_size:
                X[cursor] = arr[-window_size:]
            else:
                pad_len = window_size - t_len
                X[cursor, :pad_len] = arr[0]
                X[cursor, pad_len:] = arr
            if has_target:
                y[cursor] = float(grp[target_col].iloc[-1])
            meta_records.append(grp.iloc[[-1]][meta_cols])
            cursor += 1
        else:
            pad = np.repeat(arr[:1], window_size - 1, axis=0)
            padded = np.concatenate([pad, arr], axis=0)
            # Shape: (t_len, window_size, n_features)
            all_wins = sliding_window_view(
                padded, window_shape=window_size, axis=0
            ).transpose(0, 2, 1)

            idx = np.arange(min_history - 1, t_len, stride, dtype=np.int64)
            X[cursor : cursor + n_win] = all_wins[idx]
            if has_target:
                y[cursor : cursor + n_win] = grp[target_col].to_numpy(dtype=np.float32)[idx]
            meta_records.append(grp.iloc[idx][meta_cols])
            cursor += n_win

    meta_df = pd.concat(meta_records, axis=0, ignore_index=True) if meta_records else pd.DataFrame(
        columns=meta_cols
    )
    return X, y, meta_df
