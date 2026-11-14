"""Unit tests for Saxena (a, b) physics feature extraction, 3-client FedAvg, and conformal intervals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import INFORMATIVE_SENSORS, OP_SETTING_COLS
from src.features.engineering import build_sequence_windows
from src.features.physics_degradation import (
    SAXENA_FEATURE_COLS,
    SaxenaPhysicsExtractor,
    fit_saxena_curve_nls,
    saxena_model_ab,
)
from src.models.federated import (
    CLIENT_IDS,
    build_client_sequence_datasets,
    calibrate_and_evaluate_conformal,
    fedavg_aggregate_state_dicts,
    partition_units_into_operator_clients,
    train_fedavg_model,
)


def _make_synthetic_multi_regime_df(n_units: int = 12, cycles_per_unit: int = 45, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic multi-regime C-MAPSS-like DataFrame with ground-truth exponential wear."""
    rng = np.random.default_rng(seed)
    regimes = [
        (42.0, 0.84, 100.0),  # High-Alt / High-Mach
        (35.0, 0.84, 100.0),  # High-Alt / High-Mach
        (25.0, 0.62, 60.0),   # Low-Throttle
        (20.0, 0.70, 100.0),  # Mid
        (10.0, 0.25, 100.0),  # Low-Alt
        (0.0, 0.00, 100.0),   # Sea-level
    ]
    rows = []
    for u in range(1, n_units + 1):
        # Give units varying regime preferences so Client A/B/C have distinct duty scores
        if u <= n_units // 3:
            probs = [0.35, 0.35, 0.05, 0.15, 0.05, 0.05]
        elif u <= 2 * (n_units // 3):
            probs = [0.05, 0.05, 0.30, 0.15, 0.25, 0.20]
        else:
            probs = [1 / 6] * 6

        a_u = float(rng.uniform(1e-5, 5e-5))
        b_u = float(rng.uniform(1.8, 2.4))
        for t in range(1, cycles_per_unit + 1):
            r_idx = int(rng.choice(6, p=probs))
            alt, mach, tra = regimes[r_idx]
            wear = float(1.0 - np.exp(-a_u * (t ** b_u)))
            rul = float(cycles_per_unit - t)
            row = {
                "unit_id": u,
                "cycle": t,
                "op_setting_1": alt,
                "op_setting_2": mach,
                "op_setting_3": tra,
                "RUL_raw": rul,
                "RUL": min(rul, 125.0),
            }
            for s_idx, col in enumerate(INFORMATIVE_SENSORS):
                sign = -1.0 if col in ("sensor_7", "sensor_12", "sensor_20", "sensor_21") else 1.0
                row[col] = float(sign * 2.5 * wear + rng.normal(0.0, 0.05 + 0.01 * s_idx))
            rows.append(row)
    return pd.DataFrame(rows)


def test_saxena_curve_nls_recovers_ground_truth() -> None:
    """Verify nonlinear least squares accurately recovers (a, b) from h(t) = 1 - exp(-a * t^b)."""
    t = np.arange(1, 181, dtype=np.float64)
    true_a = 1.8e-6
    true_b = 2.65
    d_clean = saxena_model_ab(t, true_a, true_b)

    res = fit_saxena_curve_nls(t, d_clean, use_scipy=True)
    assert res["r2"] > 0.999
    assert np.isclose(res["b"], true_b, rtol=2e-2)
    assert np.isclose(np.log10(res["a"]), np.log10(true_a), atol=0.08)


def test_saxena_extractor_strict_zero_future_leakage(tmp_path: Path) -> None:
    """Verify causal prefix fitting at cycle t never uses cycles > t and persists cleanly."""
    df = _make_synthetic_multi_regime_df(n_units=6, cycles_per_unit=50, seed=7)
    extractor = SaxenaPhysicsExtractor(
        sensor_cols=list(INFORMATIVE_SENSORS),
        min_cycles=12,
        nls_stride=2,
    ).fit(df)

    # Save and reload
    save_path = tmp_path / "saxena_extractor.joblib"
    extractor.save(save_path)
    loaded = SaxenaPhysicsExtractor.load(save_path)

    u1_full = df[df["unit_id"] == 1].copy().reset_index(drop=True)
    cutoff = 30
    u1_trunc = u1_full.iloc[:cutoff].copy().reset_index(drop=True)

    # Corrupt future rows (> cutoff) in u1_full to extreme values and verify prefix <= cutoff is unchanged
    u1_corrupted_future = u1_full.copy()
    for col in INFORMATIVE_SENSORS:
        u1_corrupted_future.loc[u1_corrupted_future["cycle"] > cutoff, col] = 999.0

    out_trunc, f_cols = loaded.transform(u1_trunc)
    out_corrupt, _ = loaded.transform(u1_corrupted_future)

    assert f_cols == SAXENA_FEATURE_COLS
    for col in ["saxena_a", *SAXENA_FEATURE_COLS]:
        np.testing.assert_allclose(
            out_trunc[col].to_numpy(),
            out_corrupt.iloc[:cutoff][col].to_numpy(),
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"Future leakage detected in column {col}",
        )


def test_operator_client_partition_disjoint_and_complete() -> None:
    """Verify 3-client split partitions engines disjointly and orders duty severity."""
    df = _make_synthetic_multi_regime_df(n_units=12, cycles_per_unit=40, seed=11)
    client_map, prof_df = partition_units_into_operator_clients(df)

    all_units = set(df["unit_id"].unique().tolist())
    a_set = set(client_map["client_a_high_alt_mach"])
    b_set = set(client_map["client_b_low_alt_throttle"])
    c_set = set(client_map["client_c_mixed"])

    assert a_set.isdisjoint(b_set)
    assert a_set.isdisjoint(c_set)
    assert b_set.isdisjoint(c_set)
    assert (a_set | b_set | c_set) == all_units

    mean_a_high = prof_df[prof_df["client_id"] == "client_a_high_alt_mach"]["frac_high_alt_mach"].mean()
    mean_b_high = prof_df[prof_df["client_id"] == "client_b_low_alt_throttle"]["frac_high_alt_mach"].mean()
    assert mean_a_high > mean_b_high


def test_fedavg_aggregate_state_dicts_exact_weighted_average() -> None:
    """Verify FedAvg computes the exact sample-weighted average of client state dicts."""
    s1 = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32), "step": torch.tensor(5, dtype=torch.int64)}
    s2 = {"w": torch.tensor([4.0, 8.0], dtype=torch.float32), "step": torch.tensor(5, dtype=torch.int64)}
    agg = fedavg_aggregate_state_dicts([s1, s2], client_weights=[100.0, 300.0])

    # Expected: 0.25 * [1, 2] + 0.75 * [4, 8] = [3.25, 6.50]
    np.testing.assert_allclose(agg["w"].numpy(), np.array([3.25, 6.50], dtype=np.float32), atol=1e-6)


def test_train_fedavg_and_conformal_end_to_end() -> None:
    """Verify end-to-end 3-client FedAvg training and split-conformal interval calibration."""
    df = _make_synthetic_multi_regime_df(n_units=9, cycles_per_unit=35, seed=21)
    client_units, _ = partition_units_into_operator_clients(df)

    feat_cols = list(INFORMATIVE_SENSORS) + list(OP_SETTING_COLS)
    X_all, y_all, meta_all = build_sequence_windows(
        df, feature_cols=feat_cols, window_size=15, stride=2
    )
    client_ds = build_client_sequence_datasets(
        df, X_all, y_all, meta_all, client_units, apply_regime_bias=True, seed=21
    )
    client_train_xy = {cid: (client_ds[cid][0], client_ds[cid][1]) for cid in CLIENT_IDS}

    model, hist = train_fedavg_model(
        arch="bilstm",
        n_features=len(feat_cols),
        seq_len=15,
        client_train_data=client_train_xy,
        client_val_data=client_train_xy,
        rounds=3,
        local_epochs=1,
        batch_size=32,
        lr=2e-3,
        model_kwargs={"hidden_size": 16, "num_layers": 1, "dropout": 0.1, "fc_hidden": 16},
        seed=21,
    )
    assert len(hist) == 3
    assert hist[-1]["val_rmse_pooled"] <= hist[0]["val_rmse_pooled"] + 5.0

    X_last, y_last, _ = build_sequence_windows(
        df, feature_cols=feat_cols, window_size=15, last_only=True
    )
    conf = calibrate_and_evaluate_conformal(
        model,
        X_cal=X_last,
        y_cal=y_last,
        X_test=X_last,
        y_test=y_last,
        alpha=0.10,
        clip_cap=125.0,
    )
    assert conf["conformal_q_cycles"] >= 0.0
    assert 0.0 <= conf["empirical_coverage"] <= 1.0
    assert np.all(conf["ci_low"] <= conf["pred_test"] + 1e-6)
    assert np.all(conf["pred_test"] <= conf["ci_high"] + 1e-6)
