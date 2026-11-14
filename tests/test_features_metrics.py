"""Tests for causal feature engineering, fault-mode clustering, windowing, and metrics."""

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

from src.config import INFORMATIVE_SENSORS, ROLLING_WINDOWS, WINDOW_SIZE  # noqa: E402
from src.data.loader import load_cmapss_dataset, train_val_engine_split  # noqa: E402
from src.data.preprocessing import RegimeNormalizer  # noqa: E402
from src.evaluation.benchmarks import PUBLISHED_BENCHMARKS, compare_against_benchmarks  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    early_late_breakdown,
    evaluate_predictions,
    mae,
    mean_nasa_score,
    nasa_asymmetric_score,
    rmse,
)
from src.features.engineering import (  # noqa: E402
    FaultModeClusterer,
    add_tabular_features,
    build_sequence_windows,
)


def test_add_tabular_features_strict_causality_and_no_leakage() -> None:
    """Rolling windows (5, 10, 20) and diff1 are strictly causal and exclude targets/IDs."""
    train_df, _, _ = load_cmapss_dataset("FD001")
    norm = RegimeNormalizer(n_regimes=1, use_regime_norm=False).fit(train_df)
    subset = norm.transform(train_df[train_df["unit_id"].isin([1, 2, 3])].copy())

    feat_orig, feat_cols = add_tabular_features(
        subset,
        sensor_cols=INFORMATIVE_SENSORS,
        windows=ROLLING_WINDOWS,
        include_regime_onehot=True,
        n_regimes=1,
    )

    # No target or ID leakage in feature_cols
    for forbidden in ("RUL", "RUL_raw", "unit_id", "cycle", "dataset_id", "regime"):
        assert forbidden not in feat_cols
    for w in ROLLING_WINDOWS:
        assert f"{INFORMATIVE_SENSORS[0]}_roll_mean_{w}" in feat_cols
        assert f"{INFORMATIVE_SENSORS[0]}_roll_std_{w}" in feat_cols
    assert f"{INFORMATIVE_SENSORS[0]}_diff_1" in feat_cols

    # Mutate all future cycles (> 25) dramatically; earlier cycles (<= 25) must stay identical
    cutoff = 25
    perturbed = subset.copy()
    future_mask = perturbed["cycle"] > cutoff
    perturbed.loc[future_mask, INFORMATIVE_SENSORS] = 9999.0
    feat_pert, feat_cols_pert = add_tabular_features(
        perturbed,
        sensor_cols=INFORMATIVE_SENSORS,
        windows=ROLLING_WINDOWS,
        include_regime_onehot=True,
        n_regimes=1,
    )
    assert feat_cols_pert == feat_cols

    early_orig = feat_orig.loc[feat_orig["cycle"] <= cutoff, feat_cols].reset_index(drop=True)
    early_pert = feat_pert.loc[feat_pert["cycle"] <= cutoff, feat_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_orig, early_pert)

    # Shuffled row order should yield identical aligned features
    shuffled = subset.sample(frac=1.0, random_state=7).copy()
    feat_shuf, _ = add_tabular_features(
        shuffled,
        sensor_cols=INFORMATIVE_SENSORS,
        windows=ROLLING_WINDOWS,
        include_regime_onehot=True,
        n_regimes=1,
    )
    orig_sorted = feat_orig.sort_values(["unit_id", "cycle"]).reset_index(drop=True)[feat_cols]
    shuf_sorted = feat_shuf.sort_values(["unit_id", "cycle"]).reset_index(drop=True)[feat_cols]
    pd.testing.assert_frame_equal(orig_sorted, shuf_sorted)


def test_fault_mode_clusterer_valid_probs_causal_and_target_free(tmp_path: Path) -> None:
    """FaultModeClusterer outputs probabilities summing to 1, is causal, and uses no RUL."""
    train_df, test_df, _ = load_cmapss_dataset("FD003")
    tr_df, val_df = train_val_engine_split(train_df, val_ratio=0.2, seed=42)
    norm = RegimeNormalizer(n_regimes=1, use_regime_norm=False).fit(tr_df)
    tr_norm = norm.transform(tr_df)
    val_norm = norm.transform(val_df)
    test_norm = norm.transform(test_df)

    # Fit with RUL present vs completely stripped -> identical centroids & probabilities
    clusterer_with_rul = FaultModeClusterer(n_modes=2, sensor_cols=INFORMATIVE_SENSORS, tail_cycles=25)
    clusterer_no_rul = FaultModeClusterer(n_modes=2, sensor_cols=INFORMATIVE_SENSORS, tail_cycles=25)
    clusterer_with_rul.fit(tr_norm)
    clusterer_no_rul.fit(tr_norm.drop(columns=["RUL", "RUL_raw"]))

    np.testing.assert_allclose(clusterer_with_rul.centroids_, clusterer_no_rul.centroids_)

    val_out, prob_cols = clusterer_with_rul.transform(val_norm)
    test_out, test_prob_cols = clusterer_with_rul.transform(test_norm)
    assert prob_cols == ["fault_mode_prob_0", "fault_mode_prob_1"]
    assert test_prob_cols == prob_cols

    for out_df in (val_out, test_out):
        probs = out_df[prob_cols].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(probs))
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.array_equal(out_df["fault_mode_pred"].to_numpy(), np.argmax(probs, axis=1))

    # Verify transform is strictly causal on test cycles
    cutoff = 20
    test_pert = test_norm.copy()
    test_pert.loc[test_pert["cycle"] > cutoff, INFORMATIVE_SENSORS] = -500.0
    pert_out, _ = clusterer_with_rul.transform(test_pert)
    early_mask = test_norm["cycle"] <= cutoff
    np.testing.assert_allclose(
        test_out.loc[early_mask, prob_cols].to_numpy(),
        pert_out.loc[early_mask, prob_cols].to_numpy(),
    )

    # Save / load roundtrip
    path = clusterer_with_rul.save(tmp_path / "fault_clusterer.joblib")
    loaded = FaultModeClusterer.load(path)
    loaded_out, _ = loaded.transform(test_norm)
    pd.testing.assert_frame_equal(loaded_out[prob_cols + ["fault_mode_pred"]], test_out[prob_cols + ["fault_mode_pred"]])


@pytest.mark.parametrize("dataset_id,expected_n_test,expected_min_len", [("FD002", 259, 21), ("FD004", 248, 19)])
def test_build_sequence_windows_shapes_and_short_unit_left_padding(
    dataset_id: str, expected_n_test: int, expected_min_len: int
) -> None:
    """build_sequence_windows produces exact shapes and edge-repeats short test units (<30 cycles)."""
    train_df, test_df, _ = load_cmapss_dataset(dataset_id)
    norm = RegimeNormalizer(n_regimes=6, use_regime_norm=True).fit(train_df)
    tr_norm = norm.transform(train_df.iloc[:1500].copy())
    te_norm = norm.transform(test_df)

    # Train full sliding windows
    X_tr, y_tr, meta_tr = build_sequence_windows(
        tr_norm, feature_cols=INFORMATIVE_SENSORS, window_size=WINDOW_SIZE, target_col="RUL", last_only=False
    )
    assert X_tr.dtype == np.float32
    assert y_tr.dtype == np.float32
    assert X_tr.shape == (len(tr_norm), WINDOW_SIZE, len(INFORMATIVE_SENSORS))
    assert y_tr.shape == (len(tr_norm),)
    assert len(meta_tr) == len(tr_norm)
    assert not np.isnan(X_tr).any()
    assert not np.isnan(y_tr).any()

    # Test last_only=True across all test units including short (<30) units
    unit_lengths = te_norm.groupby("unit_id")["cycle"].count()
    assert int(unit_lengths.min()) == expected_min_len
    short_units = unit_lengths[unit_lengths < WINDOW_SIZE].index.tolist()
    assert len(short_units) > 0

    X_te, y_te, meta_te = build_sequence_windows(
        te_norm, feature_cols=INFORMATIVE_SENSORS, window_size=WINDOW_SIZE, target_col="RUL", last_only=True
    )
    assert X_te.shape == (expected_n_test, WINDOW_SIZE, len(INFORMATIVE_SENSORS))
    assert y_te.shape == (expected_n_test,)
    assert np.isnan(y_te).all()  # no RUL column in test_df
    assert len(meta_te) == expected_n_test
    assert not np.isnan(X_te).any()

    # Check exact left-edge repeat padding for short unit
    short_uid = short_units[0]
    row_idx = int(meta_te.index[meta_te["unit_id"] == short_uid][0])
    unit_raw = te_norm.loc[te_norm["unit_id"] == short_uid].sort_values("cycle")[INFORMATIVE_SENSORS].to_numpy(dtype=np.float32)
    t_len = len(unit_raw)
    pad_len = WINDOW_SIZE - t_len
    assert pad_len > 0
    np.testing.assert_allclose(X_te[row_idx, :pad_len], np.repeat(unit_raw[:1], pad_len, axis=0))
    np.testing.assert_allclose(X_te[row_idx, pad_len:], unit_raw)


def test_nasa_asymmetric_score_and_metrics_math() -> None:
    """Verify Saxena et al. (2008) Eq. 11 asymmetric penalty and evaluation helpers."""
    y_true = np.array([50.0])
    late_score = nasa_asymmetric_score(y_true, np.array([70.0]))  # d = +20 -> exp(20/10) - 1
    early_score = nasa_asymmetric_score(y_true, np.array([30.0]))  # d = -20 -> exp(20/13) - 1
    exact_score = nasa_asymmetric_score(y_true, np.array([50.0]))

    assert exact_score == pytest.approx(0.0)
    assert late_score == pytest.approx(np.exp(20.0 / 10.0) - 1.0, rel=1e-12)
    assert early_score == pytest.approx(np.exp(20.0 / 13.0) - 1.0, rel=1e-12)
    assert late_score > early_score

    yt = np.array([10.0, 20.0, 30.0, 150.0])
    yp = np.array([12.0, 15.0, 30.0, 120.0])
    assert rmse(yt, yp) == pytest.approx(np.sqrt((4 + 25 + 0 + 900) / 4.0))
    assert mae(yt, yp) == pytest.approx((2 + 5 + 0 + 30) / 4.0)
    assert mean_nasa_score(yt, yp) == pytest.approx(nasa_asymmetric_score(yt, yp) / 4.0)

    bd = early_late_breakdown(yt, yp)
    assert bd["n_early"] == 2
    assert bd["n_late"] == 1
    assert bd["n_exact"] == 1
    assert bd["early_pct"] == pytest.approx(50.0)
    assert bd["late_pct"] == pytest.approx(25.0)

    summary = evaluate_predictions(yt, yp, clip_cap=125)
    assert summary["n_samples"] == 4
    # With clip_cap=125, y_true[3] becomes 125 while y_pred[3]=120 -> d=-5
    yt_clipped = np.minimum(yt, 125.0)
    assert summary["rmse"] == pytest.approx(rmse(yt_clipped, yp))
    assert summary["nasa_score"] == pytest.approx(nasa_asymmetric_score(yt_clipped, yp))
    assert summary["late_rate"] == pytest.approx(0.25)


def test_benchmarks_comparison() -> None:
    """compare_against_benchmarks joins published literature numbers and signed % diffs."""
    assert "Zheng et al. (2017) Deep LSTM" in PUBLISHED_BENCHMARKS
    assert "Li et al. (2018) DCNN" in PUBLISHED_BENCHMARKS

    sample_df = pd.DataFrame(
        [
            {"model": "TestCNN", "dataset_id": "FD001", "rmse": 12.61, "nasa_score": 273.7},
            {"model": "TestCNN", "dataset_id": "FD004", "rmse": 28.17, "nasa_score": 5550.0},
        ]
    )
    comp = compare_against_benchmarks(sample_df)
    assert comp.loc[0, "rmse_pct_diff_vs_li_2018"] == pytest.approx(0.0)
    assert comp.loc[0, "score_pct_diff_vs_li_2018"] == pytest.approx(0.0)
    assert comp.loc[1, "rmse_pct_diff_vs_zheng_2017"] == pytest.approx(0.0)
    assert comp.loc[1, "score_pct_diff_vs_zheng_2017"] == pytest.approx(0.0)
