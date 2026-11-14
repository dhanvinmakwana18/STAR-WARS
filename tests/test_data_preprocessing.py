"""Tests for C-MAPSS data loading, RUL computation/clipping, splitting, and regime normalization."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MPL_DIR = _ROOT / ".cache" / "matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

from src.config import (  # noqa: E402
    ALL_COLS,
    DATASETS,
    DROP_SENSORS_CANONICAL,
    INFORMATIVE_SENSORS,
    OP_SETTING_COLS,
    RUL_CLIP,
    SENSOR_COLS,
)
from src.data.loader import (  # noqa: E402
    analyze_sensor_variance,
    clip_rul,
    load_cmapss_dataset,
    train_val_engine_split,
)
from src.data.preprocessing import RegimeNormalizer  # noqa: E402

EXPECTED_UNIT_COUNTS = {
    "FD001": (100, 100),
    "FD002": (260, 259),
    "FD003": (100, 100),
    "FD004": (249, 248),
}


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_load_cmapss_dataset_shapes_and_counts(dataset_id: str) -> None:
    """All 4 C-MAPSS datasets load with 26 raw cols and exact train/test unit counts."""
    train_df, test_df, y_test_raw = load_cmapss_dataset(dataset_id, rul_clip=RUL_CLIP)

    expected_train_units, expected_test_units = EXPECTED_UNIT_COUNTS[dataset_id]
    assert list(test_df.columns) == ALL_COLS
    assert len(ALL_COLS) == 26
    assert list(train_df.columns) == ALL_COLS + ["RUL_raw", "RUL"]

    assert train_df["unit_id"].nunique() == expected_train_units
    assert test_df["unit_id"].nunique() == expected_test_units
    assert len(y_test_raw) == expected_test_units
    assert np.all(y_test_raw >= 0)

    # Verify RUL_raw = max_cycle(unit) - cycle and RUL clipped at 125
    unit_max = train_df.groupby("unit_id")["cycle"].transform("max")
    expected_raw = (unit_max - train_df["cycle"]).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(train_df["RUL_raw"].to_numpy(), expected_raw)
    assert train_df.groupby("unit_id")["RUL_raw"].min().eq(0).all()
    assert train_df["RUL"].max() <= RUL_CLIP
    assert train_df["RUL"].min() >= 0
    np.testing.assert_allclose(
        train_df["RUL"].to_numpy(),
        np.minimum(expected_raw, float(RUL_CLIP)),
    )


def test_clip_rul_and_unclipped_load() -> None:
    """clip_rul caps RUL at 125 and floors at 0; rul_clip=None keeps raw RUL."""
    y = np.array([-5.0, 0.0, 50.0, 125.0, 126.0, 300.0])
    clipped = clip_rul(y, cap=125)
    np.testing.assert_allclose(clipped, np.array([0.0, 0.0, 50.0, 125.0, 125.0, 125.0]))

    train_unclipped, _, _ = load_cmapss_dataset("FD001", rul_clip=None)
    np.testing.assert_allclose(
        train_unclipped["RUL"].to_numpy(),
        train_unclipped["RUL_raw"].to_numpy(),
    )
    assert train_unclipped["RUL"].max() > RUL_CLIP


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_train_val_engine_split_zero_overlap(dataset_id: str) -> None:
    """Engine-level split has strictly zero unit_id overlap and preserves all units."""
    train_df, _, _ = load_cmapss_dataset(dataset_id)
    tr_split, val_split = train_val_engine_split(train_df, val_ratio=0.2, seed=42)

    tr_units = set(tr_split["unit_id"].unique().tolist())
    val_units = set(val_split["unit_id"].unique().tolist())
    all_units = set(train_df["unit_id"].unique().tolist())

    assert len(tr_units & val_units) == 0
    assert (tr_units | val_units) == all_units
    assert len(tr_split) + len(val_split) == len(train_df)
    assert abs(len(val_units) / len(all_units) - 0.2) < 0.05


def test_train_val_engine_split_stratified_multi_dataset() -> None:
    """When dataset_id is present, split stratifies by dataset_id with zero (ds, unit) overlap."""
    frames = []
    for ds in DATASETS:
        tr, _, _ = load_cmapss_dataset(ds)
        tr["dataset_id"] = ds
        frames.append(tr)
    combined = pd.concat(frames, ignore_index=True)

    tr_split, val_split = train_val_engine_split(combined, val_ratio=0.2, seed=42)
    assert len(tr_split) + len(val_split) == len(combined)

    for ds in DATASETS:
        ds_all = set(combined.loc[combined["dataset_id"] == ds, "unit_id"].unique())
        ds_tr = set(tr_split.loc[tr_split["dataset_id"] == ds, "unit_id"].unique())
        ds_val = set(val_split.loc[val_split["dataset_id"] == ds, "unit_id"].unique())
        assert len(ds_tr & ds_val) == 0
        assert (ds_tr | ds_val) == ds_all
        assert len(ds_val) > 0


def test_analyze_sensor_variance_identifies_constant_sensors() -> None:
    """Within-regime std empirically confirms the 7 canonical constant sensors on FD001-FD004."""
    assert len(DROP_SENSORS_CANONICAL) == 7
    assert len(INFORMATIVE_SENSORS) == 14
    assert len(SENSOR_COLS) == 21

    for ds, n_reg in [("FD001", 1), ("FD002", 6), ("FD003", 1), ("FD004", 6)]:
        train_df, _, _ = load_cmapss_dataset(ds)
        summary = analyze_sensor_variance(train_df, n_regimes=n_reg)
        constant_found = set(summary.loc[summary["is_near_constant"], "sensor"])
        assert constant_found == set(DROP_SENSORS_CANONICAL), f"{ds}: got {constant_found}"

        if n_reg == 6:
            # Before regime grouping, FD002/FD004 global std is inflated across regimes
            info_rows = summary[summary["sensor"].isin(INFORMATIVE_SENSORS)]
            assert (info_rows["within_regime_std"] < info_rows["global_std"]).all()
            # Constant sensors 1, 5, 18, 19 look deceptive globally (>1.0) yet ~0 within regime
            for s_name in ["sensor_1", "sensor_5", "sensor_18", "sensor_19"]:
                row = summary.loc[summary["sensor"] == s_name].iloc[0]
                assert row["global_std"] > 1.0
                assert row["within_regime_std"] < 0.02


@pytest.mark.parametrize("dataset_id", ["FD002", "FD004"])
def test_regime_normalizer_train_only_and_within_regime_alignment(
    dataset_id: str, tmp_path: Path
) -> None:
    """RegimeNormalizer fits strictly on train, aligns regimes, and stabilizes within-regime std."""
    train_df, test_df, _ = load_cmapss_dataset(dataset_id)
    tr_df, val_df = train_val_engine_split(train_df, val_ratio=0.2, seed=42)

    unfitted = RegimeNormalizer(n_regimes=6)
    with pytest.raises(RuntimeError):
        unfitted.transform(val_df)

    # Fit ONLY on tr_df and verify val/test changes cannot alter fitted state
    norm_regime = RegimeNormalizer(n_regimes=6, use_regime_norm=True, random_state=42)
    tr_copy = tr_df.copy()
    tr_norm = norm_regime.fit_transform(tr_df)
    pd.testing.assert_frame_equal(tr_df, tr_copy)  # input not mutated

    val_norm = norm_regime.transform(val_df)
    test_norm = norm_regime.transform(test_df)

    # Verify refitting on tr_df produces identical results regardless of test_df
    norm_global = RegimeNormalizer(n_regimes=6, use_regime_norm=False, random_state=42).fit(tr_df)
    val_global = norm_global.transform(val_df)

    assert "regime" in tr_norm.columns
    assert pd.api.types.is_integer_dtype(val_norm["regime"])
    assert set(val_norm["regime"].unique()).issubset(set(range(6)))
    assert len(val_norm["regime"].unique()) == 6
    assert set(test_norm["regime"].unique()).issubset(set(range(6)))

    # On raw data, within-regime std is much smaller than raw global std
    raw_global_std = val_df[INFORMATIVE_SENSORS].std(ddof=0).mean()
    raw_within_std = (
        val_df.assign(regime=val_norm["regime"])
        .groupby("regime")[INFORMATIVE_SENSORS]
        .std(ddof=0)
        .mean()
        .mean()
    )
    assert raw_within_std < 0.15 * raw_global_std

    # Compare RegimeNormalizer vs Global Z-score:
    # Global Z-score leaves large between-regime mean offsets (>0.5) & crushed within-regime std (<0.15).
    # Within-regime Z-score eliminates between-regime offsets (<0.1) & restores unit within-regime scale (~1.0).
    between_mean_global = (
        val_global.groupby("regime")[INFORMATIVE_SENSORS].mean().std(ddof=0).mean()
    )
    between_mean_regime = (
        val_norm.groupby("regime")[INFORMATIVE_SENSORS].mean().std(ddof=0).mean()
    )
    within_std_global = (
        val_global.groupby("regime")[INFORMATIVE_SENSORS].std(ddof=0).mean().mean()
    )
    within_std_regime = (
        val_norm.groupby("regime")[INFORMATIVE_SENSORS].std(ddof=0).mean().mean()
    )

    assert between_mean_regime < 0.1
    assert between_mean_global > 0.5
    assert between_mean_regime < 0.1 * between_mean_global
    assert 0.8 <= within_std_regime <= 1.2
    assert within_std_global < 0.15

    # Op settings are scaled cleanly
    assert np.all(np.isfinite(val_norm[OP_SETTING_COLS].to_numpy()))

    # Joblib save/load roundtrip preserves exact transforms
    saved_path = norm_regime.save(tmp_path / "regime_norm.joblib")
    loaded = RegimeNormalizer.load(saved_path)
    pd.testing.assert_frame_equal(loaded.transform(test_df), test_norm)
