"""Data loading, RUL target computation, engine-level splitting, and variance analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALL_COLS,
    DATASETS,
    DROP_SENSORS_CANONICAL,
    OP_SETTING_COLS,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RUL_CLIP,
    SENSOR_COLS,
)


def clip_rul(y: np.ndarray, cap: int = RUL_CLIP) -> np.ndarray:
    """Clip RUL values to [0, cap] (piecewise-linear RUL cap)."""
    arr = np.asarray(y, dtype=np.float64)
    if cap is None:
        return arr.copy()
    return np.clip(arr, a_min=0.0, a_max=float(cap))


def load_cmapss_dataset(
    dataset_id: str,
    raw_dir: Path | None = None,
    rul_clip: int | None = RUL_CLIP,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train, test, and raw test RUL ground truth for one C-MAPSS sub-dataset.

    Adds ``RUL_raw = max_cycle(unit) - cycle`` and ``RUL = RUL_raw.clip(upper=rul_clip)``
    to ``train_df``. Returns ``(train_df, test_df, y_test_raw)``.
    """
    ds_id = dataset_id.upper().strip()
    if ds_id not in DATASETS:
        raise ValueError(f"Unknown dataset_id '{dataset_id}'. Expected one of {DATASETS}.")

    data_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    train_path = data_dir / f"train_{ds_id}.txt"
    test_path = data_dir / f"test_{ds_id}.txt"
    rul_path = data_dir / f"RUL_{ds_id}.txt"

    for path in (train_path, test_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing C-MAPSS file: {path}")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=ALL_COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=ALL_COLS)
    y_test_raw = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(
        dtype=np.float64
    )

    for df in (train_df, test_df):
        df["unit_id"] = df["unit_id"].astype(np.int64)
        df["cycle"] = df["cycle"].astype(np.int64)

    train_df = train_df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    test_df = test_df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    n_test_units = test_df["unit_id"].nunique()
    if len(y_test_raw) != n_test_units:
        raise ValueError(
            f"{ds_id}: RUL count ({len(y_test_raw)}) != test units ({n_test_units})."
        )

    max_cycles = train_df.groupby("unit_id")["cycle"].transform("max")
    rul_raw = (max_cycles - train_df["cycle"]).astype(np.float64)
    train_df["RUL_raw"] = rul_raw
    if rul_clip is not None:
        train_df["RUL"] = rul_raw.clip(lower=0.0, upper=float(rul_clip))
    else:
        train_df["RUL"] = rul_raw.copy()

    return train_df, test_df, y_test_raw


def train_val_engine_split(
    train_df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data strictly by engine (unit_id) without cycle leakage.

    If a ``dataset_id`` column is present, splits by ``(dataset_id, unit_id)``
    stratified across each dataset.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")

    rng = np.random.default_rng(seed)

    if "dataset_id" in train_df.columns:
        val_mask = np.zeros(len(train_df), dtype=bool)
        for ds_id in sorted(train_df["dataset_id"].unique()):
            ds_mask = (train_df["dataset_id"] == ds_id).to_numpy()
            units = np.sort(train_df.loc[ds_mask, "unit_id"].unique())
            if len(units) < 2:
                raise ValueError(
                    f"Dataset '{ds_id}' has only {len(units)} unit(s); cannot split."
                )
            shuffled = rng.permutation(units)
            n_val = max(1, min(len(units) - 1, int(round(len(units) * val_ratio))))
            val_units = set(shuffled[:n_val].tolist())
            val_mask |= ds_mask & train_df["unit_id"].isin(val_units).to_numpy()
        train_split = train_df.loc[~val_mask].reset_index(drop=True)
        val_split = train_df.loc[val_mask].reset_index(drop=True)
        return train_split, val_split

    units = np.sort(train_df["unit_id"].unique())
    if len(units) < 2:
        raise ValueError(f"Need at least 2 units to split, found {len(units)}.")
    shuffled = rng.permutation(units)
    n_val = max(1, min(len(units) - 1, int(round(len(units) * val_ratio))))
    val_units = set(shuffled[:n_val].tolist())
    val_mask = train_df["unit_id"].isin(val_units)
    train_split = train_df.loc[~val_mask].reset_index(drop=True)
    val_split = train_df.loc[val_mask].reset_index(drop=True)
    return train_split, val_split


def analyze_sensor_variance(
    train_df: pd.DataFrame,
    n_regimes: int = 1,
) -> pd.DataFrame:
    """Compute global and within-regime sensor variability to verify constant sensors."""
    sensors = [c for c in SENSOR_COLS if c in train_df.columns]
    if not sensors:
        raise ValueError("No sensor columns found in train_df.")

    work = train_df.copy()
    if n_regimes > 1:
        op_cols = [c for c in OP_SETTING_COLS if c in work.columns]
        if op_cols:
            scaled_ops = StandardScaler().fit_transform(
                work[op_cols].to_numpy(dtype=np.float64)
            )
            regimes = KMeans(
                n_clusters=int(n_regimes),
                n_init=20,
                random_state=RANDOM_SEED,
            ).fit_predict(scaled_ops)
        elif "regime" in work.columns:
            regimes = work["regime"].to_numpy(dtype=np.int64)
        else:
            regimes = np.zeros(len(work), dtype=np.int64)
    else:
        regimes = np.zeros(len(work), dtype=np.int64)

    work["_regime"] = regimes
    has_rul = "RUL" in work.columns

    if has_rul:
        # Center within regime to measure degradation correlation cleanly on multi-regime data
        centered = work[sensors] - work.groupby("_regime")[sensors].transform("mean")
        rul_series = work["RUL"].astype(np.float64)
    else:
        centered = None
        rul_series = None

    grouped = work.groupby("_regime")[sensors]
    per_reg_std = grouped.std(ddof=0)
    per_reg_iqr = grouped.quantile(0.75) - grouped.quantile(0.25)

    rows = []
    for sensor in sensors:
        s_vals = work[sensor].to_numpy(dtype=np.float64)
        g_mean = float(np.mean(s_vals))
        g_std = float(np.std(s_vals, ddof=0))
        w_std = float(per_reg_std[sensor].mean())
        w_iqr = float(per_reg_iqr[sensor].mean())

        if has_rul and w_std > 1e-9:
            corr = float(centered[sensor].corr(rul_series))
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0 if has_rul else np.nan

        rows.append(
            {
                "sensor": sensor,
                "global_mean": g_mean,
                "global_std": g_std,
                "within_regime_std": w_std,
                "within_regime_iqr": w_iqr,
                "corr_with_rul": corr,
                "is_canonical_drop": sensor in DROP_SENSORS_CANONICAL,
                "is_near_constant": bool(w_std < 0.02),
            }
        )

    return pd.DataFrame(rows)
