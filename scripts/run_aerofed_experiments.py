"""End-to-end experiment runner for the STAR WARS Physics-Informed Federated Conformal upgrade.

Executes and saves artifacts for the three technical pillars in docs/aerofed_research_memo.md:
1. Causal Physics Curve-Fitting (Saxena et al. 2008 Eq. 4/5: h(t) = 1 - exp(-a * t^b)):
   - Fits (a, b) per engine causally on prefix 1..t (zero future leakage).
   - Runs a controlled ablation (XGBoost with vs. without (a, b)) across FD001-FD004,
     evaluating both overall test error and error stratified across:
       * Early Life (cycle < 35, where (a, b) is underdetermined)
       * Mid Life (cycle >= 35 and 45 <= RUL <= 110)
       * Late Life / Near-Failure (RUL < 45, where exponential curvature dominates)
   - Saves `reports/results/ablation_physics_ab.csv`, `reports/results/saxena_fitted_parameters.csv`,
     and `models/{ds}/saxena_extractor.joblib`.

2. 3-Client Simulated Federated Learning (FedAvg, McMahan et al., 2017) on FD002 & FD004:
   - Partitions engines into 3 disjoint simulated operator clients by dominant flight regime:
       * client_A: High-Altitude / High-Mach Cruise Fleet
       * client_B: Low-Altitude / Low-Throttle Sea-Level & Climb Fleet
       * client_C: Mixed / Balanced Multi-Regime Fleet
   - Trains and compares:
       * Local-Only BiLSTM (trained solely on client_A, client_B, or client_C)
       * Global Federated BiLSTM (FedAvg across E in {1, 2, 4} local epochs per round)
       * Centralized Pooled BiLSTM (upper-bound reference ceiling)
   - Saves `reports/results/federated_client_comparison.csv`,
     `reports/results/federated_convergence_rounds.csv`, and `models/{ds}/federated_bilstm.pt`.

3. Split-Conformal Uncertainty Calibration Layer (Vovk et al.):
   - Calibrates 90% distribution-free intervals [y_hat - q_0.90, y_hat + q_0.90] on held-out
     calibration splits for centralized and federated models.
   - Audits empirical coverage (target >= 90%) and interval width on unseen test engines.
   - Saves `reports/results/conformal_calibration_audit.csv`.

4. Generates diagnostic figures in `reports/figures/`:
   - `ablation_physics_ab.png`
   - `federated_convergence_and_clients.png`
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASETS as DATASET_IDS,
    FIGURES_DIR,
    INFORMATIVE_SENSORS,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    WINDOW_SIZE,
)
from src.data.loader import load_cmapss_dataset, train_val_engine_split
from src.data.preprocessing import RegimeNormalizer
from src.evaluation.metrics import evaluate_predictions
from src.features.engineering import add_tabular_features, build_sequence_windows
from src.features.physics_degradation import SAXENA_FEATURE_COLS, SaxenaPhysicsExtractor
from src.models.federated import (
    CLIENT_IDS,
    CLIENT_LABELS,
    build_client_sequence_datasets,
    calibrate_and_evaluate_conformal,
    partition_units_into_operator_clients,
    train_fedavg_model,
)
from src.models.sequence import (
    SequenceRULModel,
    build_sequence_model,
    predict_deterministic,
    save_sequence_checkpoint,
    train_sequence_model,
)
from src.models.tabular import train_xgboost

VAL_FRACTION = 0.20
POOLED_SCOPE_LABEL = "All Test Engines (Pooled Fleet)"


def prepare_dataset_with_physics(dataset_id: str) -> dict:
    """Load, regime-normalize, extract rolling + causal Saxena (a,b) features for a dataset."""
    train_df, test_df, rul_test = load_cmapss_dataset(dataset_id)
    unit_max_cyc = test_df.groupby("unit_id")["cycle"].transform("max")
    unit_end_rul = test_df["unit_id"].map(
        {uid: float(rul_test[idx]) for idx, uid in enumerate(sorted(test_df["unit_id"].unique()))}
    )
    test_df = test_df.copy()
    test_df["RUL_raw"] = (unit_max_cyc - test_df["cycle"]) + unit_end_rul
    test_df["RUL"] = np.minimum(test_df["RUL_raw"], 125.0)

    # Fit or load RegimeNormalizer on training engines only
    norm_path = MODELS_DIR / dataset_id / "normalizer.joblib"
    if norm_path.exists():
        normalizer = RegimeNormalizer.load(norm_path)
        train_norm = normalizer.transform(train_df)
        test_norm = normalizer.transform(test_df)
    else:
        normalizer = RegimeNormalizer(
            sensor_cols=list(INFORMATIVE_SENSORS),
            n_regimes=6,
            random_state=RANDOM_SEED,
        )
        train_norm = normalizer.fit_transform(train_df)
        test_norm = normalizer.transform(test_df)

    # Causal rolling / slope / EWMA features
    train_eng, base_feature_cols = add_tabular_features(train_norm, sensor_cols=list(INFORMATIVE_SENSORS))
    test_eng, _ = add_tabular_features(test_norm, sensor_cols=list(INFORMATIVE_SENSORS))

    # Causal Saxena (2008) Eq. 4/5 Physics Extractor fit on training engines only (or load if cached)
    ds_model_dir = MODELS_DIR / dataset_id
    ds_model_dir.mkdir(parents=True, exist_ok=True)
    saxena_path = ds_model_dir / "saxena_extractor.joblib"
    if saxena_path.exists():
        saxena_ext = SaxenaPhysicsExtractor.load(saxena_path)
        train_eng, _ = saxena_ext.transform(train_eng)
        test_eng, _ = saxena_ext.transform(test_eng)
    else:
        saxena_ext = SaxenaPhysicsExtractor(
            sensor_cols=list(INFORMATIVE_SENSORS),
            min_cycles=18,
            nls_stride=3,
        )
        train_eng, _ = saxena_ext.fit_transform(train_eng)
        test_eng, _ = saxena_ext.transform(test_eng)
        saxena_ext.save(saxena_path)

    physics_feature_cols = base_feature_cols + list(SAXENA_FEATURE_COLS)
    regime_cols = [f"regime_{i}" for i in range(6) if f"regime_{i}" in train_eng.columns]
    seq_feature_cols = (
        list(INFORMATIVE_SENSORS)
        + ["op_setting_1", "op_setting_2", "op_setting_3"]
        + regime_cols
        + list(SAXENA_FEATURE_COLS)
    )
    return {
        "dataset_id": dataset_id,
        "train_raw": train_df,
        "test_raw": test_df,
        "train_eng": train_eng,
        "test_eng": test_eng,
        "base_feature_cols": base_feature_cols,
        "physics_feature_cols": physics_feature_cols,
        "seq_feature_cols": seq_feature_cols,
        "saxena_extractor": saxena_ext,
    }


def run_phase1_physics_ablation() -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run controlled ablation of Saxena (a, b) physics features across FD001-FD004."""
    print("=== [1/3] Running Saxena (2008) Eq. 4/5 Physics Curve-Fitting Ablation ===", flush=True)
    ablation_rows: list[dict] = []
    fitted_param_rows: list[pd.DataFrame] = []
    prepared_cache: dict[str, dict] = {}

    for ds in DATASET_IDS:
        t0 = time.time()
        prep = prepare_dataset_with_physics(ds)
        prepared_cache[ds] = prep

        train_eng = prep["train_eng"]
        test_eng = prep["test_eng"]
        base_cols = prep["base_feature_cols"]
        phys_cols = prep["physics_feature_cols"]
        sax_ext: SaxenaPhysicsExtractor = prep["saxena_extractor"]

        summary_df = sax_ext.summarize_unit_parameters(train_eng)
        summary_df["dataset_id"] = ds
        fitted_param_rows.append(summary_df)

        tr_df, val_df = train_val_engine_split(train_eng, val_ratio=VAL_FRACTION, seed=RANDOM_SEED)
        test_last = test_eng.sort_values(["unit_id", "cycle"]).groupby("unit_id", as_index=False).last()

        model_no_ab = train_xgboost(
            tr_df[base_cols], tr_df["rul"], val_df[base_cols], val_df["rul"], feature_cols=base_cols
        )
        model_with_ab = train_xgboost(
            tr_df[phys_cols], tr_df["rul"], val_df[phys_cols], val_df["rul"], feature_cols=phys_cols
        )

        pred_test_no = model_no_ab.predict(test_last[base_cols])
        pred_test_with = model_with_ab.predict(test_last[phys_cols])
        y_test = test_last["rul"].to_numpy(dtype=np.float64)

        m_test_no = evaluate_predictions(y_test, pred_test_no)
        m_test_with = evaluate_predictions(y_test, pred_test_with)
        ablation_rows.append(
            {
                "dataset_id": ds,
                "split_segment": "Official Test Set (All Units)",
                "rmse_without_ab": round(float(m_test_no["rmse"]), 3),
                "rmse_with_ab": round(float(m_test_with["rmse"]), 3),
                "rmse_delta": round(float(m_test_with["rmse"] - m_test_no["rmse"]), 3),
                "nasa_without_ab": round(float(m_test_no["nasa_score"]), 1),
                "nasa_with_ab": round(float(m_test_with["nasa_score"]), 1),
                "nasa_delta": round(float(m_test_with["nasa_score"] - m_test_no["nasa_score"]), 1),
                "n_samples": int(len(y_test)),
            }
        )

        val_pred_no = model_no_ab.predict(val_df[base_cols])
        val_pred_with = model_with_ab.predict(val_df[phys_cols])
        val_y = val_df["rul"].to_numpy(dtype=np.float64)
        val_cyc = val_df["cycle"].to_numpy(dtype=np.float64)

        segments = [
            ("Early Life (cycle < 35, underdetermined fit)", val_cyc < 35),
            ("Mid Life (cycle >= 35 & 45 <= RUL <= 110)", (val_cyc >= 35) & (val_y >= 45) & (val_y <= 110)),
            ("Late Life / Near-Failure (RUL < 45, converged fit)", val_y < 45),
        ]
        for seg_name, mask in segments:
            if int(np.sum(mask)) == 0:
                continue
            m_no = evaluate_predictions(val_y[mask], val_pred_no[mask])
            m_with = evaluate_predictions(val_y[mask], val_pred_with[mask])
            ablation_rows.append(
                {
                    "dataset_id": ds,
                    "split_segment": seg_name,
                    "rmse_without_ab": round(float(m_no["rmse"]), 3),
                    "rmse_with_ab": round(float(m_with["rmse"]), 3),
                    "rmse_delta": round(float(m_with["rmse"] - m_no["rmse"]), 3),
                    "nasa_without_ab": round(float(m_no["nasa_score"]), 1),
                    "nasa_with_ab": round(float(m_with["nasa_score"]), 1),
                    "nasa_delta": round(float(m_with["nasa_score"] - m_no["nasa_score"]), 1),
                    "n_samples": int(np.sum(mask)),
                }
            )

        print(
            f"  [{ds}] Done in {time.time() - t0:.1f}s | Test RMSE: "
            f"{m_test_no['rmse']:.2f} -> {m_test_with['rmse']:.2f} | "
            f"NASA: {m_test_no['nasa_score']:.1f} -> {m_test_with['nasa_score']:.1f}",
            flush=True,
        )

    ablation_df = pd.DataFrame(ablation_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(RESULTS_DIR / "ablation_physics_ab.csv", index=False)

    fitted_params_df = pd.concat(fitted_param_rows, ignore_index=True)
    fitted_params_df.to_csv(RESULTS_DIR / "saxena_fitted_parameters.csv", index=False)
    return ablation_df, prepared_cache


def run_phase2_federated_and_conformal(
    prepared_cache: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run 3-Client Simulated Federated Learning (FedAvg) and Conformal Audit on FD002 & FD004."""
    print(
        "\n=== [2/3] Running 3-Client Federated Learning (FedAvg) & Conformal Audit (FD002 & FD004) ===",
        flush=True,
    )
    comp_rows: list[dict] = []
    conv_rows: list[dict] = []
    conf_rows: list[dict] = []

    for ds in ("FD002", "FD004"):
        t0 = time.time()
        prep = prepared_cache[ds]
        train_eng = prep["train_eng"]
        test_eng = prep["test_eng"]
        feature_cols = prep["seq_feature_cols"]

        # 1. Engine-level train/val split before client partitioning
        tr_df, val_df = train_val_engine_split(train_eng, val_ratio=VAL_FRACTION, seed=RANDOM_SEED)

        # Partition train, val, and test engines into 3 disjoint simulated operator clients
        tr_client_units, _ = partition_units_into_operator_clients(tr_df)
        val_client_units, _ = partition_units_into_operator_clients(val_df)
        test_client_units, _ = partition_units_into_operator_clients(test_eng)

        # 2. Build sequence windows for train, val, and test (last_only=True for official test)
        X_tr_all, y_tr_all, meta_tr = build_sequence_windows(
            tr_df, feature_cols=feature_cols, window_size=WINDOW_SIZE, last_only=False
        )
        X_val_all, y_val_all, meta_val = build_sequence_windows(
            val_df, feature_cols=feature_cols, window_size=WINDOW_SIZE, last_only=False
        )
        X_test_all, y_test_all, meta_test = build_sequence_windows(
            test_eng, feature_cols=feature_cols, window_size=WINDOW_SIZE, last_only=True
        )

        # Build per-client datasets (with realistic regime-skewed sampling on training clients)
        tr_clients_raw = build_client_sequence_datasets(
            tr_df,
            X_tr_all,
            y_tr_all,
            meta_tr,
            tr_client_units,
            apply_regime_bias=True,
            dominant_keep_prob=1.0,
            non_dominant_keep_prob=0.25,
            seed=RANDOM_SEED,
        )
        val_clients_raw = build_client_sequence_datasets(
            val_df,
            X_val_all,
            y_val_all,
            meta_val,
            val_client_units,
            apply_regime_bias=False,
            seed=RANDOM_SEED,
        )
        test_clients_raw = build_client_sequence_datasets(
            test_eng,
            X_test_all,
            y_test_all,
            meta_test,
            test_client_units,
            apply_regime_bias=False,
            seed=RANDOM_SEED,
        )

        # Subsample overlapping 30-cycle sliding windows (keep every 3rd step + all RUL <= 35)
        client_train_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        client_val_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        client_test_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for cid in CLIENT_IDS:
            X_c, y_c, _ = tr_clients_raw[cid]
            keep_tr = np.where((np.arange(len(y_c)) % 3 == 0) | (y_c <= 35.0))[0]
            client_train_data[cid] = (X_c[keep_tr], y_c[keep_tr])

            X_v, y_v, _ = val_clients_raw[cid]
            keep_val = np.where((np.arange(len(y_v)) % 2 == 0) | (y_v <= 35.0))[0]
            client_val_data[cid] = (X_v[keep_val], y_v[keep_val])

            X_t, y_t, _ = test_clients_raw[cid]
            client_test_data[cid] = (X_t, y_t)

        X_train_pooled = np.concatenate([client_train_data[c][0] for c in CLIENT_IDS], axis=0)
        y_train_pooled = np.concatenate([client_train_data[c][1] for c in CLIENT_IDS], axis=0)
        X_val_pooled = np.concatenate([client_val_data[c][0] for c in CLIENT_IDS], axis=0)
        y_val_pooled = np.concatenate([client_val_data[c][1] for c in CLIENT_IDS], axis=0)

        eval_scopes: dict[str, tuple[np.ndarray, np.ndarray]] = {
            POOLED_SCOPE_LABEL: (X_test_all, y_test_all),
            **{f"{cid} ({CLIENT_LABELS[cid]})": client_test_data[cid] for cid in CLIENT_IDS},
        }

        # 3. Train Local-Only BiLSTM models per client (no parameter sharing)
        for cid in CLIENT_IDS:
            X_c, y_c = client_train_data[cid]
            X_v, y_v = client_val_data[cid]
            loc_net = build_sequence_model(
                arch="bilstm",
                n_features=len(feature_cols),
                seq_len=WINDOW_SIZE,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
            )
            loc_net, _ = train_sequence_model(
                model=loc_net,
                X_train=X_c,
                y_train=y_c,
                X_val=X_v,
                y_val=y_v,
                epochs=12,
                batch_size=512,
                lr=1.5e-3,
                patience=4,
                seed=RANDOM_SEED,
                verbose=False,
            )
            print(f"    [{ds}] Trained Local-{cid} ({len(y_c)} windows)", flush=True)

            for scope_name, (X_ev, y_ev) in eval_scopes.items():
                preds = predict_deterministic(loc_net, X_ev, clip_lower=0.0, clip_upper=125.0)
                m = evaluate_predictions(y_ev, preds, clip_cap=125, clip_pred=True)
                comp_rows.append(
                    {
                        "dataset_id": ds,
                        "strategy": f"Local-Only ({cid})",
                        "local_epochs": 12,
                        "eval_scope": scope_name,
                        "rmse": round(float(m["rmse"]), 3),
                        "mae": round(float(m["mae"]), 3),
                        "nasa_score": round(float(m["nasa_score"]), 1),
                        "n_test_units": int(len(y_ev)),
                    }
                )

        # 4. Train Federated (FedAvg) models across E in {1, 2, 4} local epochs per round
        best_fed_model = None
        for E in (1, 2, 4):
            fed_model, history = train_fedavg_model(
                arch="bilstm",
                n_features=len(feature_cols),
                seq_len=WINDOW_SIZE,
                client_train_data=client_train_data,
                client_val_data=client_val_data,
                rounds=12,
                local_epochs=E,
                batch_size=512,
                lr=1.5e-3,
                model_kwargs={"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
                seed=RANDOM_SEED,
            )
            if E == 2:
                best_fed_model = fed_model

            for h_row in history:
                conv_rows.append({"dataset_id": ds, **h_row})

            for scope_name, (X_ev, y_ev) in eval_scopes.items():
                preds = predict_deterministic(fed_model, X_ev, clip_lower=0.0, clip_upper=125.0)
                m = evaluate_predictions(y_ev, preds, clip_cap=125, clip_pred=True)
                comp_rows.append(
                    {
                        "dataset_id": ds,
                        "strategy": f"Global Federated (FedAvg, E={E})",
                        "local_epochs": E,
                        "eval_scope": scope_name,
                        "rmse": round(float(m["rmse"]), 3),
                        "mae": round(float(m["mae"]), 3),
                        "nasa_score": round(float(m["nasa_score"]), 1),
                        "n_test_units": int(len(y_ev)),
                    }
                )
            pooled_m = [
                r
                for r in comp_rows
                if r["dataset_id"] == ds
                and r["strategy"] == f"Global Federated (FedAvg, E={E})"
                and r["eval_scope"] == POOLED_SCOPE_LABEL
            ][-1]
            print(
                f"    [{ds}] Trained FedAvg (E={E}) | Pooled Test RMSE={pooled_m['rmse']:.2f} | NASA={pooled_m['nasa_score']:.1f}",
                flush=True,
            )

        # 5. Train Centralized Pooled BiLSTM (Upper-bound reference ceiling)
        cent_net = build_sequence_model(
            arch="bilstm",
            n_features=len(feature_cols),
            seq_len=WINDOW_SIZE,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        )
        cent_net, _ = train_sequence_model(
            model=cent_net,
            X_train=X_train_pooled,
            y_train=y_train_pooled,
            X_val=X_val_pooled,
            y_val=y_val_pooled,
            epochs=14,
            batch_size=512,
            lr=1.5e-3,
            patience=4,
            seed=RANDOM_SEED,
            verbose=False,
        )
        print(f"    [{ds}] Trained Centralized Pooled Reference", flush=True)

        for scope_name, (X_ev, y_ev) in eval_scopes.items():
            preds = predict_deterministic(cent_net, X_ev, clip_lower=0.0, clip_upper=125.0)
            m = evaluate_predictions(y_ev, preds, clip_cap=125, clip_pred=True)
            comp_rows.append(
                {
                    "dataset_id": ds,
                    "strategy": "Centralized Pooled (Upper-Bound Ceiling)",
                    "local_epochs": 14,
                    "eval_scope": scope_name,
                    "rmse": round(float(m["rmse"]), 3),
                    "mae": round(float(m["mae"]), 3),
                    "nasa_score": round(float(m["nasa_score"]), 1),
                    "n_test_units": int(len(y_ev)),
                }
            )

        # 6. Split-Conformal Calibration Audit (Target: 90% Coverage, alpha=0.10)
        # Note: calibrate on full validation trajectories (covering RUL in [0, 125]),
        # NOT last_only=True where every training engine ends at RUL=0.
        assert best_fed_model is not None

        fed_conf_pooled = calibrate_and_evaluate_conformal(
            model=best_fed_model,
            X_cal=X_val_all,
            y_cal=y_val_all,
            X_test=X_test_all,
            y_test=y_test_all,
            alpha=0.10,
        )
        for k in ("pred_test", "ci_low", "ci_high"):
            fed_conf_pooled.pop(k, None)

        # Save federated checkpoint (nn.Module) with conformal quantile for Streamlit compatibility
        best_fed_model.conformal_q_ = float(fed_conf_pooled["conformal_q_cycles"])
        best_fed_model.window_size_ = WINDOW_SIZE
        save_sequence_checkpoint(
            best_fed_model,
            MODELS_DIR / ds / "federated_bilstm.pt",
            feature_cols=feature_cols,
        )

        conf_rows.append(
            {
                "dataset_id": ds,
                "model": "Global Federated BiLSTM (FedAvg E=2)",
                "calibration_scope": POOLED_SCOPE_LABEL,
                **fed_conf_pooled,
            }
        )
        for cid in CLIENT_IDS:
            X_vc, y_vc = client_val_data[cid]
            X_tc, y_tc = client_test_data[cid]
            c_audit = calibrate_and_evaluate_conformal(
                model=best_fed_model,
                X_cal=X_vc if len(X_vc) >= 20 else X_val_all,
                y_cal=y_vc if len(y_vc) >= 20 else y_val_all,
                X_test=X_tc,
                y_test=y_tc,
                alpha=0.10,
            )
            for k in ("pred_test", "ci_low", "ci_high"):
                c_audit.pop(k, None)
            conf_rows.append(
                {
                    "dataset_id": ds,
                    "model": "Global Federated BiLSTM (FedAvg E=2)",
                    "calibration_scope": f"{cid} ({CLIENT_LABELS[cid]})",
                    **c_audit,
                }
            )

        cent_conf = calibrate_and_evaluate_conformal(
            model=cent_net,
            X_cal=X_val_all,
            y_cal=y_val_all,
            X_test=X_test_all,
            y_test=y_test_all,
            alpha=0.10,
        )
        for k in ("pred_test", "ci_low", "ci_high"):
            cent_conf.pop(k, None)
        conf_rows.append(
            {
                "dataset_id": ds,
                "model": "Centralized Pooled BiLSTM (Ceiling)",
                "calibration_scope": POOLED_SCOPE_LABEL,
                **cent_conf,
            }
        )

        print(f"  [{ds}] Federated & Conformal benchmark completed in {time.time() - t0:.1f}s", flush=True)

    comp_df = pd.DataFrame(comp_rows)
    conv_df = pd.DataFrame(conv_rows)
    conf_df = pd.DataFrame(conf_rows)

    comp_df.to_csv(RESULTS_DIR / "federated_client_comparison.csv", index=False)
    conv_df.to_csv(RESULTS_DIR / "federated_convergence_rounds.csv", index=False)
    conf_df.to_csv(RESULTS_DIR / "conformal_calibration_audit.csv", index=False)
    return comp_df, conv_df, conf_df


def generate_aerofed_figures(
    ablation_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    conv_df: pd.DataFrame,
    conf_df: pd.DataFrame,
) -> None:
    """Generate clean, factual diagnostic figures for the physics & federated ablations."""
    print("\n=== [3/3] Generating Diagnostic Figures ===", flush=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Physics (a, b) Ablation — Official Test Set + Life-Stage Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=160)
    test_sub = ablation_df[ablation_df["split_segment"] == "Official Test Set (All Units)"]
    x = np.arange(len(test_sub))
    w = 0.36
    axes[0].bar(x - w / 2, test_sub["rmse_without_ab"], width=w, label="Without Saxena (a, b)", color="#64748b")
    axes[0].bar(x + w / 2, test_sub["rmse_with_ab"], width=w, label="With Causal Saxena (a, b)", color="#0284c7")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(test_sub["dataset_id"])
    axes[0].set_ylabel("Test RMSE (cycles, Clipped@125)")
    axes[0].set_title("Official Test Set RMSE: Without vs. With Causal (a, b)")
    axes[0].legend(frameon=True)

    seg_sub = ablation_df[ablation_df["split_segment"] != "Official Test Set (All Units)"]
    pivot = seg_sub.pivot(index="dataset_id", columns="split_segment", values="rmse_delta")
    cols_order = [
        "Early Life (cycle < 35, underdetermined fit)",
        "Mid Life (cycle >= 35 & 45 <= RUL <= 110)",
        "Late Life / Near-Failure (RUL < 45, converged fit)",
    ]
    cols_order = [c for c in cols_order if c in pivot.columns]
    pivot = pivot[cols_order]
    x2 = np.arange(len(pivot.index))
    w2 = 0.25
    colors = ["#f59e0b", "#3b82f6", "#10b981"]
    short_labels = ["Early Life (t < 35)", "Mid Life (45<=RUL<=110)", "Late Life (RUL < 45)"]
    for idx, col in enumerate(cols_order):
        axes[1].bar(x2 + (idx - 1) * w2, pivot[col], width=w2, label=short_labels[idx], color=colors[idx])
    axes[1].axhline(0.0, color="#1e293b", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(pivot.index)
    axes[1].set_ylabel("RMSE Change (With (a,b) - Without (a,b)) [cycles]")
    axes[1].set_title("Validation RMSE Delta by Engine Life Stage (Negative = Better)")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ablation_physics_ab.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Federated Convergence Curves & Local vs Federated vs Centralized Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=160)
    for ds, ls in [("FD002", "-"), ("FD004", "--")]:
        for E, col in [(1, "#6366f1"), (2, "#0284c7"), (4, "#10b981")]:
            sub = conv_df[(conv_df["dataset_id"] == ds) & (conv_df["local_epochs"] == E)]
            if not sub.empty:
                axes[0].plot(
                    sub["round"],
                    sub["val_rmse_pooled"],
                    linestyle=ls,
                    marker="o",
                    markersize=3.5,
                    color=col,
                    label=f"{ds} FedAvg (E={E})",
                )
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Pooled Validation Fleet RMSE (cycles)")
    axes[0].set_title("FedAvg Convergence Across Communication Rounds (FD002 & FD004)")
    axes[0].legend(fontsize=8, ncol=2, frameon=True)

    pooled_sub = comp_df[comp_df["eval_scope"] == POOLED_SCOPE_LABEL].copy()
    keep_strats = [
        *[f"Local-Only ({cid})" for cid in CLIENT_IDS],
        "Global Federated (FedAvg, E=2)",
        "Centralized Pooled (Upper-Bound Ceiling)",
    ]
    pooled_sub = pooled_sub[pooled_sub["strategy"].isin(keep_strats)]
    piv_fed = pooled_sub.pivot(index="dataset_id", columns="strategy", values="rmse")[keep_strats]
    x3 = np.arange(len(piv_fed.index))
    w3 = 0.15
    strat_colors = ["#94a3b8", "#64748b", "#475569", "#0284c7", "#0f766e"]
    for idx, strat in enumerate(keep_strats):
        axes[1].bar(
            x3 + (idx - 2) * w3,
            piv_fed[strat],
            width=w3,
            label=strat.replace(" (Upper-Bound Ceiling)", ""),
            color=strat_colors[idx],
        )
    axes[1].set_xticks(x3)
    axes[1].set_xticklabels(piv_fed.index)
    axes[1].set_ylabel("Pooled Test Fleet RMSE (cycles)")
    axes[1].set_title("Simulated 3-Client Operator Split: Local vs. FedAvg vs. Centralized")
    axes[1].legend(fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "federated_convergence_and_clients.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved `ablation_physics_ab.png` and `federated_convergence_and_clients.png`.", flush=True)


def main() -> None:
    ablation_csv = RESULTS_DIR / "ablation_physics_ab.csv"
    saxena_csv = RESULTS_DIR / "saxena_fitted_parameters.csv"
    all_saxena_saved = all((MODELS_DIR / ds / "saxena_extractor.joblib").exists() for ds in DATASET_IDS)

    if ablation_csv.exists() and saxena_csv.exists() and all_saxena_saved:
        print("=== [1/3] Loading cached Phase 1 Saxena Physics Ablation & preparing FD002/FD004 ===", flush=True)
        ablation_df = pd.read_csv(ablation_csv)
        prepared_cache = {ds: prepare_dataset_with_physics(ds) for ds in ("FD002", "FD004")}
    else:
        ablation_df, prepared_cache = run_phase1_physics_ablation()

    comp_df, conv_df, conf_df = run_phase2_federated_and_conformal(prepared_cache)
    generate_aerofed_figures(ablation_df, comp_df, conv_df, conf_df)

    summary = {
        "physics_ablation_test": ablation_df[
            ablation_df["split_segment"] == "Official Test Set (All Units)"
        ].to_dict(orient="records"),
        "federated_pooled_summary": comp_df[comp_df["eval_scope"] == POOLED_SCOPE_LABEL].to_dict(orient="records"),
        "conformal_audit": conf_df.to_dict(orient="records"),
    }
    (RESULTS_DIR / "aerofed_upgrade_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== All STAR WARS Physics-Informed Federated Conformal Experiments Complete ===", flush=True)


if __name__ == "__main__":
    main()
