"""Tests for tabular/sequence RUL models, uncertainty bands, maintenance simulator, and Streamlit app."""

from __future__ import annotations

import ast
import os
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_MPL_DIR = _ROOT / ".cache" / "matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

from src.config import INFORMATIVE_SENSORS  # noqa: E402
from src.data.loader import load_cmapss_dataset, train_val_engine_split  # noqa: E402
from src.data.preprocessing import RegimeNormalizer  # noqa: E402
from src.evaluation.maintenance import (  # noqa: E402
    CostAssumptions,
    compare_maintenance_policies,
    generate_demo_trajectories,
    load_trajectories_from_dataframe,
    load_trajectories_from_file,
    simulate_fixed_interval_policy,
    simulate_predictive_policy,
)
from src.features.engineering import add_tabular_features, build_sequence_windows  # noqa: E402
from src.models.sequence import (  # noqa: E402
    BiLSTMRegressor,
    Deep1DCNNRegressor,
    SequenceRULModel,
    load_sequence_checkpoint,
    predict_deterministic,
    predict_with_uncertainty,
    save_sequence_checkpoint,
    train_sequence_model,
)
from src.models.tabular import (  # noqa: E402
    TabularRULModel,
    load_tabular_model,
    save_tabular_model,
    train_random_forest,
    train_xgboost,
)


@pytest.fixture(scope="module")
def fd001_prepared() -> dict[str, object]:
    """Compact FD001 train/val/test slices for fast model & checkpoint unit tests."""
    train_df, test_df, y_test_raw = load_cmapss_dataset("FD001")
    # Subsample units for fast unit test execution while keeping real engine trajectories
    tr_sub = train_df[train_df["unit_id"] <= 20].copy()
    tr_df, val_df = train_val_engine_split(tr_sub, val_ratio=0.25, seed=42)
    te_sub = test_df[test_df["unit_id"] <= 10].copy()

    norm = RegimeNormalizer(n_regimes=1, use_regime_norm=False).fit(tr_df)
    tr_norm = norm.transform(tr_df)
    val_norm = norm.transform(val_df)
    te_norm = norm.transform(te_sub)

    tr_tab, tab_cols = add_tabular_features(
        tr_norm, INFORMATIVE_SENSORS, windows=(5, 10, 20), include_regime_onehot=True, n_regimes=1
    )
    val_tab, _ = add_tabular_features(
        val_norm, INFORMATIVE_SENSORS, windows=(5, 10, 20), include_regime_onehot=True, n_regimes=1
    )
    te_tab, _ = add_tabular_features(
        te_norm, INFORMATIVE_SENSORS, windows=(5, 10, 20), include_regime_onehot=True, n_regimes=1
    )
    te_last_tab = te_tab.groupby("unit_id", sort=True).tail(1).reset_index(drop=True)

    X_tr_seq, y_tr_seq, _ = build_sequence_windows(
        tr_norm, feature_cols=INFORMATIVE_SENSORS, window_size=30, target_col="RUL", last_only=False
    )
    X_val_seq, y_val_seq, _ = build_sequence_windows(
        val_norm, feature_cols=INFORMATIVE_SENSORS, window_size=30, target_col="RUL", last_only=False
    )
    X_te_seq, _, _ = build_sequence_windows(
        te_norm, feature_cols=INFORMATIVE_SENSORS, window_size=30, target_col="RUL", last_only=True
    )

    return {
        "tr_tab": tr_tab,
        "val_tab": val_tab,
        "te_last_tab": te_last_tab,
        "tab_cols": tab_cols,
        "X_tr_seq": X_tr_seq,
        "y_tr_seq": y_tr_seq,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
        "X_te_seq": X_te_seq,
        "y_te_raw": y_test_raw[:10],
    }


@pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
def test_tabular_models_predict_intervals_and_roundtrip(
    fd001_prepared: dict[str, object], tmp_path: Path, model_type: str
) -> None:
    """Tabular models produce exact shapes, valid conformal bands (ci_low <= pred <= ci_high), and roundtrip."""
    tr_tab = fd001_prepared["tr_tab"]
    val_tab = fd001_prepared["val_tab"]
    te_last = fd001_prepared["te_last_tab"]
    cols = fd001_prepared["tab_cols"]
    assert isinstance(tr_tab, pd.DataFrame)
    assert isinstance(val_tab, pd.DataFrame)
    assert isinstance(te_last, pd.DataFrame)
    assert isinstance(cols, list)

    if model_type == "random_forest":
        model = train_random_forest(
            tr_tab,
            tr_tab["RUL"],
            X_val=val_tab,
            y_val=val_tab["RUL"],
            feature_cols=cols,
            model_params={"n_estimators": 20, "max_depth": 8, "n_jobs": 1},
        )
    else:
        model = train_xgboost(
            tr_tab,
            tr_tab["RUL"],
            X_val=val_tab,
            y_val=val_tab["RUL"],
            feature_cols=cols,
            model_params={"n_estimators": 40, "max_depth": 4, "n_jobs": 1},
        )

    pred = model.predict(te_last)
    pred_u, pred_std, ci_low, ci_high = model.predict_with_uncertainty(te_last, alpha=0.10)

    n = len(te_last)
    assert pred.shape == (n,)
    assert pred_u.shape == (n,)
    assert pred_std.shape == (n,)
    assert ci_low.shape == (n,)
    assert ci_high.shape == (n,)
    assert np.all(np.isfinite(pred))
    assert np.all(pred_std >= 0.0)
    assert np.all(ci_low >= 0.0)
    assert np.all(ci_low <= pred_u + 1e-9)
    assert np.all(pred_u <= ci_high + 1e-9)
    np.testing.assert_allclose(pred, pred_u)

    imp = model.feature_importances(top_k=5)
    assert len(imp) == 5
    assert list(imp.columns) == ["feature", "importance"]

    # Joblib save / load roundtrip
    ckpt_path = save_tabular_model(model, tmp_path / f"{model_type}.joblib")
    loaded = load_tabular_model(ckpt_path)
    assert isinstance(loaded, TabularRULModel)
    pred_l, std_l, lo_l, hi_l = loaded.predict_with_uncertainty(te_last, alpha=0.10)
    np.testing.assert_allclose(pred_l, pred_u)
    np.testing.assert_allclose(std_l, pred_std)
    np.testing.assert_allclose(lo_l, ci_low)
    np.testing.assert_allclose(hi_l, ci_high)


@pytest.mark.parametrize("arch", ["1d_cnn", "bilstm"])
def test_sequence_models_forward_uncertainty_and_checkpoint_roundtrip(
    fd001_prepared: dict[str, object], tmp_path: Path, arch: str
) -> None:
    """1D-CNN and BiLSTM forward/predict shapes, MC+conformal bands, and .pt checkpoint roundtrip."""
    X_tr = fd001_prepared["X_tr_seq"]
    y_tr = fd001_prepared["y_tr_seq"]
    X_val = fd001_prepared["X_val_seq"]
    y_val = fd001_prepared["y_val_seq"]
    X_te = fd001_prepared["X_te_seq"]
    assert isinstance(X_tr, np.ndarray)
    assert isinstance(y_tr, np.ndarray)
    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    assert isinstance(X_te, np.ndarray)

    n_feat = X_tr.shape[2]
    seq_len = X_tr.shape[1]

    # Direct PyTorch module forward shape check
    if arch == "1d_cnn":
        net = Deep1DCNNRegressor(n_features=n_feat, seq_len=seq_len, channels=(16, 24), kernel_sizes=(3, 3), fc_hidden=16)
    else:
        net = BiLSTMRegressor(n_features=n_feat, seq_len=seq_len, hidden_size=16, num_layers=1, fc_hidden=16)

    with torch.no_grad():
        raw_out = net(torch.from_numpy(X_te[:4]))
    assert raw_out.shape == (4,)

    ckpt_file = tmp_path / f"{arch}.pt"
    trained_net, history = train_sequence_model(
        net,
        X_tr,
        y_tr,
        X_val=X_val,
        y_val=y_val,
        epochs=3,
        batch_size=128,
        patience=2,
        n_mc=6,
        alpha=0.10,
        checkpoint_path=ckpt_file,
        feature_cols=INFORMATIVE_SENSORS,
        seed=42,
    )
    assert ckpt_file.exists() and ckpt_file.stat().st_size > 0
    assert len(history["train_rmse"]) >= 1
    assert len(history["val_rmse"]) >= 1

    det_pred = predict_deterministic(trained_net, X_te)
    mc_mean, mc_std, ci_low, ci_high = predict_with_uncertainty(trained_net, X_te, n_mc=8, alpha=0.10)
    n_te = X_te.shape[0]
    assert det_pred.shape == (n_te,)
    assert mc_mean.shape == (n_te,)
    assert mc_std.shape == (n_te,)
    assert ci_low.shape == (n_te,)
    assert ci_high.shape == (n_te,)
    assert np.all(np.isfinite(mc_mean))
    assert np.all(mc_std >= 0.0)
    assert np.all(ci_low >= 0.0)
    assert np.all(ci_low <= mc_mean + 1e-9)
    assert np.all(mc_mean <= ci_high + 1e-9)

    # Checkpoint save / load roundtrip (weights_only=True)
    ckpt_file2 = save_sequence_checkpoint(trained_net, tmp_path / f"{arch}_copy.pt", feature_cols=INFORMATIVE_SENSORS)
    loaded_net = load_sequence_checkpoint(ckpt_file2)
    det_loaded = predict_deterministic(loaded_net, X_te)
    np.testing.assert_allclose(det_loaded, det_pred, atol=1e-6)
    assert loaded_net.feature_cols_ == INFORMATIVE_SENSORS
    assert loaded_net.window_size_ == seq_len

    _, _, lo_loaded, hi_loaded = predict_with_uncertainty(loaded_net, X_te, n_mc=8, alpha=0.10)
    assert np.all(lo_loaded <= hi_loaded + 1e-9)

    # High-level SequenceRULModel wrapper save/load smoke check
    wrapper = SequenceRULModel(arch=arch, epochs=2, batch_size=256, n_mc=4, model_kwargs={"fc_hidden": 16} if arch == "1d_cnn" else {"hidden_size": 16, "num_layers": 1, "fc_hidden": 16})
    wrapper.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, feature_cols=INFORMATIVE_SENSORS)
    w_path = wrapper.save(tmp_path / f"wrapper_{arch}.pt", feature_cols=INFORMATIVE_SENSORS)
    w_loaded = SequenceRULModel.load(w_path)
    np.testing.assert_allclose(w_loaded.predict(X_te), wrapper.predict(X_te), atol=1e-6)


def test_maintenance_simulator_invariants(tmp_path: Path) -> None:
    """Perfect predictor gets 0 unplanned failures; higher threshold cuts failures and raises early pulls."""
    # 1. Perfect predictor invariant
    perfect_trajs = []
    for uid, max_cyc in enumerate([110, 145, 190, 230, 280], start=1):
        cycles = np.arange(1, max_cyc + 1, dtype=np.float64)
        true_rul = max_cyc - cycles
        perfect_trajs.append(
            {
                "unit_id": uid,
                "cycles": cycles,
                "true_rul": true_rul,
                "pred_rul": true_rul.copy(),
            }
        )

    costs = CostAssumptions()
    perfect_res = simulate_predictive_policy(perfect_trajs, threshold_cycles=20, costs=costs, baseline_interval_cycles=150)
    assert perfect_res["unplanned_failures"] == 0
    assert perfect_res["scheduled_pulls"] == len(perfect_trajs)
    assert perfect_res["engines_pulled_early"] == 0
    assert perfect_res["total_wasted_cycles"] == pytest.approx(0.0)
    assert perfect_res["avg_true_rul_at_pull"] == pytest.approx(20.0)

    # Verify cost identity
    expected_total = (
        perfect_res["scheduled_cost"] + perfect_res["failure_cost"] + perfect_res["early_pull_cost"]
    )
    assert perfect_res["total_cost"] == pytest.approx(expected_total)

    # 2. Threshold monotonicity invariant on realistic trajectories with both late-biased & conservative/noisy predictors
    monotone_trajs = []
    for uid in range(1, 16):
        max_cyc = 180 + 5 * uid
        cycles = np.arange(1, max_cyc + 1, dtype=np.float64)
        true_rul = max_cyc - cycles
        if uid <= 5:
            # Late-biased engines (risk unplanned failure if threshold is too low)
            pred_rul = true_rul + 14.0
        else:
            # Conservative slope (risks early pull as threshold rises)
            pred_rul = 0.5 * true_rul
        monotone_trajs.append(
            {
                "unit_id": uid,
                "cycles": cycles,
                "true_rul": true_rul,
                "pred_rul": pred_rul,
            }
        )

    res_low = simulate_predictive_policy(monotone_trajs, threshold_cycles=10, costs=costs)
    res_mid = simulate_predictive_policy(monotone_trajs, threshold_cycles=25, costs=costs)
    res_high = simulate_predictive_policy(monotone_trajs, threshold_cycles=45, costs=costs)

    # Higher threshold strictly reduces unplanned failures and increases early pulls
    assert res_low["unplanned_failures"] > res_mid["unplanned_failures"] >= res_high["unplanned_failures"]
    assert res_high["unplanned_failures"] == 0
    assert res_low["engines_pulled_early"] <= res_mid["engines_pulled_early"] <= res_high["engines_pulled_early"]
    assert res_high["engines_pulled_early"] > res_low["engines_pulled_early"]
    assert res_low["total_wasted_cycles"] <= res_mid["total_wasted_cycles"] <= res_high["total_wasted_cycles"]
    assert res_low["avg_true_rul_at_pull"] < res_mid["avg_true_rul_at_pull"] < res_high["avg_true_rul_at_pull"]

    # Demo trajectories & sweep comparison smoke check
    demo_trajs = generate_demo_trajectories(n_engines=40, seed=42)
    comp = compare_maintenance_policies(demo_trajs, costs=costs)
    assert comp["n_engines"] == 40
    assert "predictive_default" in comp and "fixed_default" in comp
    assert len(comp["predictive_sweep"]) == 6
    assert len(comp["fixed_interval_sweep"]) == 6

    # Fixed interval math sanity check
    fixed_res = simulate_fixed_interval_policy(perfect_trajs, interval_cycles=150, costs=costs)
    # Units with max_cyc 110 and 145 fail before 150 -> 2 unplanned failures
    assert fixed_res["unplanned_failures"] == 2
    assert fixed_res["scheduled_pulls"] == 3

    # DataFrame & CSV trajectory loader sanity check (without and with dataset_id)
    df_traj = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "true_rul_unclipped": [2.0, 1.0, 0.0, 2.0, 1.0, 0.0],
            "pred_rul": [2.0, 1.0, 0.0, 3.0, 2.0, 1.0],
        }
    )
    loaded_trajs = load_trajectories_from_dataframe(df_traj, dataset_id="FD001")
    assert len(loaded_trajs) == 2
    assert loaded_trajs[0]["unit_id"] == 1
    assert loaded_trajs[0]["dataset_id"] == "FD001"

    csv_path = tmp_path / "val_trajectories_FD001.csv"
    df_traj.to_csv(csv_path, index=False)
    loaded_csv = load_trajectories_from_file(csv_path, dataset_id="FD001")
    assert len(loaded_csv) == 2

    df_multi = df_traj.copy()
    df_multi["dataset_id"] = ["FD001"] * 3 + ["FD002"] * 3
    loaded_multi = load_trajectories_from_dataframe(df_multi)
    assert len(loaded_multi) == 2
    assert {t["dataset_id"] for t in loaded_multi} == {"FD001", "FD002"}


def test_streamlit_app_syntax_and_apptest_smoke() -> None:
    """Verify app/streamlit_app.py parses cleanly and runs via Streamlit AppTest without errors."""
    app_path = _ROOT / "app" / "streamlit_app.py"
    assert app_path.exists()

    source = app_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(app_path))
    py_compile.compile(str(app_path), doraise=True)

    import app.streamlit_app as st_app

    assert st_app.maintenance_status(15.0)[0].startswith("CRITICAL")
    assert st_app.maintenance_status(30.0)[0] == "MONITOR"
    assert st_app.maintenance_status(80.0)[0] == "HEALTHY"

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    assert len(at.error) == 0, [e.value for e in at.error]
    assert len(at.tabs) == 5


def test_streamlit_real_artifacts_and_interactive_flows() -> None:
    """Verify real model artifact inference and interactive FD004/Unified/Mode-B Streamlit paths."""
    import app.streamlit_app as st_app
    from streamlit.testing.v1 import AppTest

    if not (st_app.MODELS_DIR / "FD001" / "best_sequence.pt").exists():
        pytest.skip("Trained artifacts not generated in models/ yet.")

    uni_bundle = st_app.load_unified_artifacts()
    assert uni_bundle.get("available") is True

    tracks = (
        "Best Per-Dataset Sequence (Auto)",
        "Per-Dataset XGBoost",
        "Unified BiLSTM (All 4 Subsets)",
    )
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        per_bundle = st_app.load_inference_artifacts(ds)
        assert per_bundle.get("available") is True, f"Missing artifacts for {ds}"
        train_df, test_df, _ = st_app.load_dataset_raw(ds)
        first_unit = int(test_df["unit_id"].min())
        unit_df = test_df[test_df["unit_id"] == first_unit].copy()

        for track in tracks:
            pred, ci_lo, ci_hi, _, _, _, active = st_app._predict_trajectory_from_selection(
                unit_df=unit_df,
                train_df=train_df,
                dataset_id=ds,
                model_track=track,
                per_bundle=per_bundle,
                uni_bundle=uni_bundle,
            )
            assert "Heuristic Fallback" not in active, f"{ds}/{track} fell back to heuristic"
            assert len(pred) == len(unit_df)
            assert not np.isnan(pred).any()
            assert np.all(ci_lo <= pred + 1e-6)
            assert np.all(pred <= ci_hi + 1e-6)

    app_path = _ROOT / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=45)
    at.run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    assert len(at.warning) == 0, [w.value for w in at.warning]

    # Switch to FD004 (multi-regime + multi-fault), Unified BiLSTM, Mode B sliders, and Tab 4 FD004 slice
    at.selectbox(key="diag_dataset").set_value("FD004")
    at.selectbox(key="diag_model_track").set_value("Unified BiLSTM (All 4 Subsets)")
    at.radio(key="diag_mode").set_value("Mode B: Interactive Sensor Sliders")
    at.selectbox(key="sim_ds").set_value("FD004")
    at.run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    assert len(at.error) == 0, [e.value for e in at.error]
    assert len(at.warning) == 0, [w.value for w in at.warning]

    # Also toggle Mode B Raw Physical Units on FD004 to verify regime-specific inverse transform
    at.radio(key="mode_b_scale_FD004").set_value("Raw Physical Units")
    at.run()
    assert len(at.exception) == 0, [e.message for e in at.exception]
    assert len(at.error) == 0, [e.value for e in at.error]
