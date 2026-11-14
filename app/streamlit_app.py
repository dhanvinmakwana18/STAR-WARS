"""Interactive Streamlit Dashboard for NASA C-MAPSS Turbofan RUL & Maintenance Intelligence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MPL_CACHE = _PROJECT_ROOT / ".cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

try:
    from src.config import (
        ALL_COLS,
        DATASET_META,
        DATASETS,
        DROP_SENSORS_CANONICAL,
        FIGURES_DIR,
        INFORMATIVE_SENSORS,
        MODELS_DIR,
        OP_SETTING_COLS,
        RAW_DATA_DIR,
        RESULTS_DIR,
        ROLLING_WINDOWS,
        RUL_CLIP,
        SENSOR_COLS,
        WINDOW_SIZE,
    )
except ImportError:
    RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
    MODELS_DIR = _PROJECT_ROOT / "models"
    FIGURES_DIR = _PROJECT_ROOT / "reports" / "figures"
    RESULTS_DIR = _PROJECT_ROOT / "reports" / "results"
    DATASETS = ("FD001", "FD002", "FD003", "FD004")
    DATASET_META = {
        "FD001": {"op_conditions": 1, "fault_modes": 1, "description": "Sea Level | HPC Degradation"},
        "FD002": {"op_conditions": 6, "fault_modes": 1, "description": "6 Op Conditions | HPC Degradation"},
        "FD003": {"op_conditions": 1, "fault_modes": 2, "description": "Sea Level | HPC + Fan Degradation"},
        "FD004": {"op_conditions": 6, "fault_modes": 2, "description": "6 Op Conditions | HPC + Fan Degradation"},
    }
    OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
    ALL_COLS = ["unit_id", "cycle"] + OP_SETTING_COLS + SENSOR_COLS
    DROP_SENSORS_CANONICAL = [
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ]
    INFORMATIVE_SENSORS = [c for c in SENSOR_COLS if c not in DROP_SENSORS_CANONICAL]
    RUL_CLIP = 125
    WINDOW_SIZE = 30
    ROLLING_WINDOWS = (5, 10, 20)

SENSOR_DESCRIPTIONS: dict[str, str] = {
    "sensor_2": "T24 — LPC outlet total temperature (°R)",
    "sensor_3": "T30 — HPC outlet total temperature (°R)",
    "sensor_4": "T50 — LPT outlet total temperature (°R)",
    "sensor_7": "P30 — HPC outlet total pressure (psia)",
    "sensor_8": "Nf — Physical fan speed (rpm)",
    "sensor_9": "Nc — Physical core speed (rpm)",
    "sensor_11": "Ps30 — HPC outlet static pressure (psia)",
    "sensor_12": "phi — Fuel flow to Ps30 ratio (pps/psi)",
    "sensor_13": "NRf — Corrected fan speed (rpm)",
    "sensor_14": "NRc — Corrected core speed (rpm)",
    "sensor_15": "BPR — Bypass ratio",
    "sensor_17": "htBleed — Bleed enthalpy",
    "sensor_20": "W31 — HPT coolant bleed (lbm/s)",
    "sensor_21": "W32 — LPT coolant bleed (lbm/s)",
}

MODEL_TRACK_OPTIONS: dict[str, tuple[str, str]] = {
    "Best Per-Dataset Sequence (Auto)": ("per_dataset", "best_sequence.pt"),
    "Per-Dataset BiLSTM": ("per_dataset", "sequence_bilstm.pt"),
    "Per-Dataset Deep1DCNN": ("per_dataset", "sequence_1dcnn.pt"),
    "Per-Dataset XGBoost": ("per_dataset", "tabular_xgb.joblib"),
    "Per-Dataset RandomForest": ("per_dataset", "tabular_rf.joblib"),
    "Unified BiLSTM (All 4 Subsets)": ("unified", "sequence_bilstm.pt"),
    "Unified 1D-CNN (All 4 Subsets)": ("unified", "sequence_1dcnn.pt"),
    "Unified XGBoost (All 4 Subsets)": ("unified", "tabular_xgb.joblib"),
}


def maintenance_status(rul: float) -> tuple[str, str, str]:
    """Return (status_label, hex_color, action_guidance) from predicted RUL."""
    if rul <= 20.0:
        return "CRITICAL — SCHEDULE MAINTENANCE", "#dc2626", "Immediate hangar slot / engine swap recommended."
    if rul <= 40.0:
        return "MONITOR", "#d97706", "Increase borescope/telemetry watch and prepare parts."
    return "HEALTHY", "#16a34a", "Normal dispatch; no near-term removal required."


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_dataset_raw(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load raw train, test, and test ground-truth RUL arrays for a C-MAPSS subset."""
    try:
        from src.data.loader import load_cmapss_dataset

        return load_cmapss_dataset(dataset_id, raw_dir=RAW_DATA_DIR, rul_clip=RUL_CLIP)
    except Exception:
        train_df = pd.read_csv(RAW_DATA_DIR / f"train_{dataset_id}.txt", sep=r"\s+", header=None, names=ALL_COLS)
        test_df = pd.read_csv(RAW_DATA_DIR / f"test_{dataset_id}.txt", sep=r"\s+", header=None, names=ALL_COLS)
        y_test = np.loadtxt(RAW_DATA_DIR / f"RUL_{dataset_id}.txt", dtype=float)
        max_cycles = train_df.groupby("unit_id")["cycle"].transform("max")
        train_df["RUL_raw"] = max_cycles - train_df["cycle"]
        train_df["RUL"] = train_df["RUL_raw"].clip(upper=RUL_CLIP)
        return train_df, test_df, y_test


@st.cache_resource(show_spinner=False)
def load_inference_artifacts(dataset_id: str) -> dict[str, Any]:
    """Discover and load saved normalizer, fault clusterer, models, and calibration metadata."""
    bundle: dict[str, Any] = {
        "available": False,
        "dataset_id": dataset_id,
        "checkpoints": {},
        "tabular_models": {},
    }
    ds_dir = MODELS_DIR / dataset_id
    if not ds_dir.exists():
        return bundle

    try:
        import joblib

        meta_path = ds_dir / "metadata.json"
        if meta_path.exists():
            bundle["metadata"] = _read_json(meta_path)

        norm_path = ds_dir / "normalizer.joblib"
        if norm_path.exists():
            bundle["normalizer"] = joblib.load(norm_path)

        fault_path = ds_dir / "fault_clusterer.joblib"
        if fault_path.exists():
            bundle["fault_clusterer"] = joblib.load(fault_path)

        for fname in ("tabular_xgb.joblib", "tabular_rf.joblib"):
            p = ds_dir / fname
            if p.exists():
                bundle["tabular_models"][fname] = joblib.load(p)

        for pt_name in ("best_sequence.pt", "sequence_bilstm.pt", "sequence_1dcnn.pt"):
            pt_path = ds_dir / pt_name
            if pt_path.exists():
                bundle["checkpoints"][pt_name] = pt_path

        bundle["available"] = bool(bundle["checkpoints"] or bundle["tabular_models"])
    except Exception as exc:
        bundle["error"] = str(exc)

    return bundle


@st.cache_resource(show_spinner=False)
def load_unified_artifacts() -> dict[str, Any]:
    """Load unified multi-dataset artifacts from models/unified/ if present."""
    bundle: dict[str, Any] = {
        "available": False,
        "dataset_id": "unified",
        "checkpoints": {},
        "tabular_models": {},
    }
    uni_dir = MODELS_DIR / "unified"
    if not uni_dir.exists():
        return bundle

    try:
        import joblib

        meta_path = uni_dir / "metadata.json"
        if meta_path.exists():
            bundle["metadata"] = _read_json(meta_path)

        norm_path = uni_dir / "normalizer.joblib"
        if norm_path.exists():
            bundle["normalizer"] = joblib.load(norm_path)

        xgb_path = uni_dir / "tabular_xgb.joblib"
        if xgb_path.exists():
            bundle["tabular_models"]["tabular_xgb.joblib"] = joblib.load(xgb_path)

        for pt_name in ("sequence_bilstm.pt", "sequence_1dcnn.pt"):
            pt_path = uni_dir / pt_name
            if pt_path.exists():
                bundle["checkpoints"][pt_name] = pt_path

        bundle["available"] = bool(bundle["checkpoints"] or bundle["tabular_models"])
    except Exception as exc:
        bundle["error"] = str(exc)

    return bundle


@st.cache_resource(show_spinner=False)
def _load_sequence_model_cached(pt_path_str: str) -> Any:
    """Instantiate and cache a PyTorch sequence model checkpoint."""
    try:
        from src.models.sequence import load_sequence_checkpoint

        return load_sequence_checkpoint(pt_path_str, map_location="cpu")
    except Exception:
        return None


def _compute_heuristic_rul(df_norm: pd.DataFrame, sensor_cols: list[str]) -> np.ndarray:
    """Fallback degradation proxy when trained model artifacts are not yet saved."""
    z_mag = np.sqrt((df_norm[sensor_cols].to_numpy(dtype=float) ** 2).mean(axis=1))
    return np.clip(125.0 - 38.0 * np.maximum(z_mag - 0.5, 0.0), 0.0, float(RUL_CLIP))


def _attach_dataset_onehot(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    out = df.copy()
    for ds in DATASETS:
        out[f"dataset_{ds}"] = 1.0 if ds == dataset_id else 0.0
    return out


def _attach_regime_onehot(df: pd.DataFrame, n_regimes: int) -> pd.DataFrame:
    out = df.copy()
    if n_regimes > 1:
        regimes = out["regime"].to_numpy(dtype=int) if "regime" in out.columns else np.zeros(len(out), dtype=int)
        for r in range(n_regimes):
            out[f"regime_{r}"] = (regimes == r).astype(np.float32)
    return out


def _prepare_engineered_frame(
    raw_df: pd.DataFrame,
    train_df: pd.DataFrame,
    dataset_id: str,
    bundle: dict[str, Any],
    fault_bundle: dict[str, Any],
    is_unified: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame | None, list[str]]:
    """Normalize raw cycles, attach regime & dataset one-hots, fault probabilities, and rolling features."""
    normalizer = bundle.get("normalizer")
    if normalizer is not None:
        df_norm = normalizer.transform(raw_df)
    else:
        try:
            from src.data.preprocessing import RegimeNormalizer

            n_reg = 6 if is_unified else DATASET_META[dataset_id]["op_conditions"]
            temp_norm = RegimeNormalizer(
                n_regimes=n_reg, sensor_cols=INFORMATIVE_SENSORS, use_regime_norm=(n_reg > 1)
            )
            temp_norm.fit(train_df)
            df_norm = temp_norm.transform(raw_df)
        except Exception:
            df_norm = raw_df.copy()
            for col in INFORMATIVE_SENSORS:
                mean = float(train_df[col].mean())
                std = max(float(train_df[col].std()), 1e-6)
                df_norm[col] = (df_norm[col] - mean) / std
            df_norm["regime"] = 0

    regimes = df_norm["regime"].to_numpy(dtype=int) if "regime" in df_norm.columns else np.zeros(len(df_norm), dtype=int)

    fault_probs: pd.DataFrame | None = None
    fault_cols: list[str] = []
    fault_clusterer = fault_bundle.get("fault_clusterer")
    if fault_clusterer is not None:
        # FaultModeClusterer was fit in the per-dataset normalized space
        per_norm = fault_bundle.get("normalizer")
        df_for_fault = per_norm.transform(raw_df) if (is_unified and per_norm is not None) else df_norm
        df_fault, fault_cols = fault_clusterer.transform(df_for_fault)
        fault_probs = df_fault[fault_cols].copy()
        for fc in fault_cols:
            df_norm[fc] = df_fault[fc].to_numpy()

    n_reg_onehot = 6 if is_unified else DATASET_META[dataset_id]["op_conditions"]
    include_reg = n_reg_onehot > 1
    try:
        from src.features.engineering import add_tabular_features

        df_feat, _ = add_tabular_features(
            df_norm,
            sensor_cols=list(INFORMATIVE_SENSORS),
            windows=ROLLING_WINDOWS,
            include_regime_onehot=include_reg,
            n_regimes=n_reg_onehot,
        )
    except Exception:
        df_feat = _attach_regime_onehot(df_norm, n_reg_onehot)

    if is_unified:
        df_feat = _attach_dataset_onehot(df_feat, dataset_id)

    return df_feat, regimes, fault_probs, fault_cols


def _predict_trajectory_from_selection(
    unit_df: pd.DataFrame,
    train_df: pd.DataFrame,
    dataset_id: str,
    model_track: str,
    per_bundle: dict[str, Any],
    uni_bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame | None, pd.DataFrame, str]:
    """Run causal cycle-by-cycle RUL inference for the chosen model track."""
    scope, artifact_name = MODEL_TRACK_OPTIONS.get(
        model_track, ("per_dataset", "best_sequence.pt")
    )
    is_unified = scope == "unified" and uni_bundle.get("available", False)
    active_bundle = uni_bundle if is_unified else per_bundle

    df_feat, regimes, fault_probs, _ = _prepare_engineered_frame(
        unit_df, train_df, dataset_id, active_bundle, per_bundle, is_unified
    )
    meta = active_bundle.get("metadata") or {}
    conformal_q = float(
        (meta.get("conformal_calibration") or {}).get("conformal_q", 18.0)
    )

    if artifact_name.endswith(".pt"):
        pt_path = active_bundle.get("checkpoints", {}).get(artifact_name)
        if pt_path is None and active_bundle.get("checkpoints"):
            pt_path = next(iter(active_bundle["checkpoints"].values()))
        if pt_path is not None:
            seq_model = _load_sequence_model_cached(str(pt_path))
            if seq_model is not None:
                try:
                    from src.features.engineering import build_sequence_windows
                    from src.models.sequence import predict_with_uncertainty

                    ckpt_cols = getattr(seq_model, "feature_cols_", None)
                    seq_features = list(ckpt_cols or meta.get("seq_feature_cols") or INFORMATIVE_SENSORS)
                    for col in seq_features:
                        if col not in df_feat.columns:
                            df_feat[col] = 0.0
                    win_size = int(getattr(seq_model, "window_size_", meta.get("window_size", WINDOW_SIZE)))
                    X_seq, _, _ = build_sequence_windows(
                        df_feat,
                        feature_cols=seq_features,
                        window_size=win_size,
                        target_col="RUL",
                        last_only=False,
                    )
                    pred_mean, _, ci_low, ci_high = predict_with_uncertainty(
                        seq_model,
                        X_seq,
                        n_mc=20,
                        alpha=0.10,
                        clip_lower=0.0,
                        clip_upper=float(RUL_CLIP),
                    )
                    return pred_mean, ci_low, ci_high, regimes, fault_probs, df_feat, f"{seq_model.__class__.__name__} ({pt_path.name})"
                except Exception:
                    pass

    tab_models = active_bundle.get("tabular_models", {})
    tab_model = tab_models.get(artifact_name) or (
        next(iter(tab_models.values())) if tab_models else None
    )
    if tab_model is not None:
        try:
            model_features = list(
                getattr(tab_model, "feature_cols_", None)
                or meta.get("tab_feature_cols")
                or []
            )
            for col in model_features:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            pred_mean, _, ci_low, ci_high = tab_model.predict_with_uncertainty(
                df_feat[model_features], alpha=0.10
            )
            pred_mean = np.clip(pred_mean, 0.0, float(RUL_CLIP))
            ci_low = np.clip(ci_low, 0.0, float(RUL_CLIP))
            ci_high = np.clip(ci_high, 0.0, float(RUL_CLIP))
            return pred_mean, ci_low, ci_high, regimes, fault_probs, df_feat, f"Tabular ({tab_model.model_type})"
        except Exception:
            pass

    pred = _compute_heuristic_rul(df_feat, INFORMATIVE_SENSORS)
    ci_low = np.clip(pred - conformal_q, 0.0, float(RUL_CLIP))
    ci_high = np.clip(pred + conformal_q, 0.0, float(RUL_CLIP))
    return pred, ci_low, ci_high, regimes, fault_probs, df_feat, "Heuristic Fallback"


def _render_status_badge(rul: float) -> None:
    label, color, guidance = maintenance_status(rul)
    st.markdown(
        f"""
        <div style="padding:12px 16px;border-radius:8px;border-left:6px solid {color};
                    background-color:rgba(128,128,128,0.08);margin-bottom:10px;">
            <div style="font-size:0.8rem;opacity:0.8;text-transform:uppercase;letter-spacing:0.05em;">
                Maintenance Status
            </div>
            <div style="font-size:1.15rem;font-weight:700;color:{color};">
                {label}
            </div>
            <div style="font-size:0.88rem;opacity:0.9;">{guidance}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_mode_a(
    dataset_id: str,
    model_track: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    per_bundle: dict[str, Any],
    uni_bundle: dict[str, Any],
) -> None:
    unit_ids = sorted(test_df["unit_id"].unique().tolist())
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_unit = st.selectbox(
            "Select Test Engine (unit_id)",
            unit_ids,
            index=0,
            key=f"mode_a_unit_{dataset_id}",
        )
    unit_df = test_df[test_df["unit_id"] == selected_unit].sort_values("cycle").reset_index(drop=True)
    max_cycle = int(unit_df["cycle"].max())
    with col_sel2:
        current_cycle = st.slider(
            "Operational Cycle Scrubber (Replay Engine Timeline)",
            min_value=1,
            max_value=max_cycle,
            value=max_cycle,
            key=f"mode_a_cycle_{dataset_id}_{selected_unit}",
        )

    pred_all, ci_low_all, ci_high_all, regimes, fault_probs, df_feat, active_label = (
        _predict_trajectory_from_selection(
            unit_df, train_df, dataset_id, model_track, per_bundle, uni_bundle
        )
    )
    idx = min(current_cycle - 1, len(pred_all) - 1)
    pred_now = float(pred_all[idx])
    low_now = float(ci_low_all[idx])
    high_now = float(ci_high_all[idx])
    regime_now = int(regimes[idx])

    true_final_raw = float(y_test[unit_ids.index(selected_unit)])
    true_final_clipped = min(true_final_raw, float(RUL_CLIP))

    _render_status_badge(pred_now)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted RUL", f"{pred_now:.1f} cycles", delta=active_label)
    m2.metric("90% Confidence Band", f"[{low_now:.1f}, {high_now:.1f}]")
    m3.metric("Operating Regime (KMeans)", f"Regime {regime_now}")
    if current_cycle == max_cycle:
        m4.metric("True Test RUL (Final Cycle)", f"{true_final_raw:.0f} raw ({true_final_clipped:.0f} clipped)")
    else:
        m4.metric("Observed Progress", f"Cycle {current_cycle} / {max_cycle}")

    if fault_probs is not None and not fault_probs.empty:
        st.caption("Latent Fault-Mode Soft Probabilities (Causal Trailing Profile via FaultModeClusterer)")
        f_cols = st.columns(fault_probs.shape[1])
        for i, _ in enumerate(fault_probs.columns):
            prob_val = float(fault_probs.iloc[idx, i])
            f_cols[i].metric(f"Latent Fault Mode {i}", f"{prob_val * 100:.1f}%")

    cycles = unit_df["cycle"].to_numpy()[: idx + 1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    ax1.plot(cycles, pred_all[: idx + 1], color="#1d4ed8", linewidth=2.2, label="Predicted RUL")
    ax1.fill_between(
        cycles,
        ci_low_all[: idx + 1],
        ci_high_all[: idx + 1],
        color="#60a5fa",
        alpha=0.25,
        label="90% Confidence Band",
    )
    ax1.axhline(20, color="#dc2626", linestyle="--", linewidth=1.2, label="Critical Threshold (20 cyc)")
    ax1.axhline(40, color="#d97706", linestyle=":", linewidth=1.2, label="Monitor Threshold (40 cyc)")
    if current_cycle == max_cycle:
        ax1.scatter([max_cycle], [true_final_clipped], color="#111827", zorder=5, s=60, label="True Clipped RUL")
    ax1.set_ylabel("RUL (cycles)")
    ax1.set_ylim(-2, RUL_CLIP + 15)
    ax1.set_title(f"{dataset_id} — Engine #{selected_unit} Live RUL Trajectory (Up to Cycle {current_cycle})")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(alpha=0.25)

    trace_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_14", "sensor_15"]
    for s_col in trace_sensors:
        if s_col in df_feat.columns:
            ax2.plot(cycles, df_feat[s_col].to_numpy()[: idx + 1], linewidth=1.3, alpha=0.85, label=s_col)
    ax2.set_xlabel("Operational Cycle")
    ax2.set_ylabel("Regime-Normalized Z-Score")
    ax2.set_title("Regime-Normalized Key Sensor Degradation Traces")
    ax2.legend(loc="upper left", fontsize=8, ncol=4)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_mode_b(
    dataset_id: str,
    model_track: str,
    train_df: pd.DataFrame,
    per_bundle: dict[str, Any],
    uni_bundle: dict[str, Any],
) -> None:
    st.markdown(
        "Adjust operating conditions and key sensor readings (in either **Normalized Z-score** or **Raw Physical Units**) "
        "to inspect real-time RUL prediction, 90% confidence interval, and top contributing sensors."
    )
    c_top1, c_top2 = st.columns([1, 2])
    with c_top1:
        input_scale = st.radio(
            "Sensor Input Scale",
            ["Normalized Z-score", "Raw Physical Units"],
            horizontal=True,
            key=f"mode_b_scale_{dataset_id}",
        )
    with c_top2:
        preset = st.radio(
            "Degradation Scenario Preset",
            ["Healthy Baseline", "Moderate Wear", "Critical Degradation"],
            horizontal=True,
            key=f"mode_b_preset_{dataset_id}",
        )

    preset_shifts = {"Healthy Baseline": 0.0, "Moderate Wear": 1.1, "Critical Degradation": 2.3}
    base_shift = preset_shifts[preset]

    key_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_14", "sensor_15"]
    scope, artifact_name = MODEL_TRACK_OPTIONS.get(model_track, ("per_dataset", "best_sequence.pt"))
    is_unified = scope == "unified" and uni_bundle.get("available", False)
    active_bundle = uni_bundle if is_unified else per_bundle
    normalizer = active_bundle.get("normalizer")
    per_norm = per_bundle.get("normalizer") or normalizer

    n_reg = DATASET_META[dataset_id]["op_conditions"]
    col_op, col_sens = st.columns([1, 2])

    with col_op:
        st.subheader("Operating Settings")
        selected_regime = st.selectbox(
            "Operating Regime",
            list(range(n_reg)),
            index=0,
            key=f"mode_b_regime_{dataset_id}",
        )
        op1 = st.slider("op_setting_1 (Altitude proxy)", -0.01, 42.05, float(train_df["op_setting_1"].median()), key=f"op1_{dataset_id}")
        op2 = st.slider("op_setting_2 (Mach proxy)", -0.01, 0.85, float(train_df["op_setting_2"].median()), key=f"op2_{dataset_id}")
        op3 = st.slider("op_setting_3 (TRA proxy)", 58.0, 102.0, float(train_df["op_setting_3"].median()), key=f"op3_{dataset_id}")

    slider_z: dict[str, float] = {}
    with col_sens:
        st.subheader(f"Sensor Inputs ({input_scale})")
        scols = st.columns(2)
        for i, s_col in enumerate(key_sensors):
            sign = -1.0 if s_col in ("sensor_7", "sensor_12") else 1.0
            default_z = float(np.clip(sign * base_shift, -3.5, 3.5))
            if input_scale == "Normalized Z-score":
                with scols[i % 2]:
                    slider_z[s_col] = st.slider(
                        s_col,
                        min_value=-3.5,
                        max_value=3.5,
                        value=default_z,
                        step=0.1,
                        help=SENSOR_DESCRIPTIONS.get(s_col, s_col),
                        key=f"slider_z_{dataset_id}_{preset}_{s_col}",
                    )
            else:
                s_idx = INFORMATIVE_SENSORS.index(s_col)
                if per_norm is not None and getattr(per_norm, "regime_means_", None) and selected_regime in per_norm.regime_means_:
                    mu_r = float(per_norm.regime_means_[selected_regime][s_idx])
                    sd_r = float(per_norm.regime_stds_[selected_regime][s_idx])
                else:
                    mu_r = float(train_df[s_col].mean())
                    sd_r = max(float(train_df[s_col].std()), 1e-4)
                low_raw = round(mu_r - 3.5 * sd_r, 3)
                high_raw = round(mu_r + 3.5 * sd_r, 3)
                default_raw = float(np.clip(mu_r + default_z * sd_r, low_raw, high_raw))
                step_raw = max(round(sd_r * 0.1, 4), 1e-4)
                with scols[i % 2]:
                    raw_val = st.slider(
                        s_col,
                        min_value=float(low_raw),
                        max_value=float(high_raw),
                        value=default_raw,
                        step=float(step_raw),
                        help=SENSOR_DESCRIPTIONS.get(s_col, s_col),
                        key=f"slider_raw_{dataset_id}_{selected_regime}_{preset}_{s_col}",
                    )
                    slider_z[s_col] = (raw_val - mu_r) / max(sd_r, 1e-6)

    # Construct a 30-cycle normalized trajectory reflecting the selected operating regime & sensors
    ramp = np.linspace(0.65, 1.0, WINDOW_SIZE)
    if normalizer is not None and getattr(normalizer, "op_scaler_", None) is not None:
        scaled_op = normalizer.op_scaler_.transform(np.array([[op1, op2, op3]], dtype=np.float64))[0]
    else:
        scaled_op = np.array(
            [
                (op1 - float(train_df["op_setting_1"].mean())) / max(float(train_df["op_setting_1"].std()), 1e-6),
                (op2 - float(train_df["op_setting_2"].mean())) / max(float(train_df["op_setting_2"].std()), 1e-6),
                (op3 - float(train_df["op_setting_3"].mean())) / max(float(train_df["op_setting_3"].std()), 1e-6),
            ],
            dtype=np.float64,
        )
    synth_norm = pd.DataFrame(
        {
            "unit_id": np.ones(WINDOW_SIZE, dtype=int),
            "cycle": np.arange(1, WINDOW_SIZE + 1, dtype=int),
            "op_setting_1": np.full(WINDOW_SIZE, float(scaled_op[0])),
            "op_setting_2": np.full(WINDOW_SIZE, float(scaled_op[1])),
            "op_setting_3": np.full(WINDOW_SIZE, float(scaled_op[2])),
            "regime": np.full(WINDOW_SIZE, selected_regime, dtype=int),
        }
    )
    for s_col in INFORMATIVE_SENSORS:
        synth_norm[s_col] = slider_z.get(s_col, 0.0) * ramp

    n_reg_onehot = 6 if is_unified else n_reg
    try:
        from src.features.engineering import add_tabular_features

        synth_feat, _ = add_tabular_features(
            synth_norm,
            sensor_cols=list(INFORMATIVE_SENSORS),
            windows=ROLLING_WINDOWS,
            include_regime_onehot=(n_reg_onehot > 1),
            n_regimes=n_reg_onehot,
        )
    except Exception:
        synth_feat = _attach_regime_onehot(synth_norm, n_reg_onehot)

    if is_unified:
        synth_feat = _attach_dataset_onehot(synth_feat, dataset_id)

    fault_clusterer = per_bundle.get("fault_clusterer")
    fault_probs_row: pd.Series | None = None
    if fault_clusterer is not None:
        synth_f, f_cols = fault_clusterer.transform(synth_norm)
        fault_probs_row = synth_f[f_cols].iloc[-1]
        for fc in f_cols:
            synth_feat[fc] = synth_f[fc].to_numpy()

    meta = active_bundle.get("metadata") or {}
    conformal_q = float((meta.get("conformal_calibration") or {}).get("conformal_q", 18.0))
    pred_val: float
    ci_low_val: float
    ci_high_val: float
    contributions: dict[str, float] = {}
    used_label = "Heuristic Fallback"

    pt_path = None
    if artifact_name.endswith(".pt"):
        pt_path = active_bundle.get("checkpoints", {}).get(artifact_name)
        if pt_path is None and active_bundle.get("checkpoints"):
            pt_path = next(iter(active_bundle["checkpoints"].values()))

    seq_model = _load_sequence_model_cached(str(pt_path)) if pt_path is not None else None

    if seq_model is not None:
        from src.features.engineering import build_sequence_windows
        from src.models.sequence import predict_deterministic, predict_with_uncertainty

        ckpt_cols = getattr(seq_model, "feature_cols_", None)
        seq_features = list(ckpt_cols or meta.get("seq_feature_cols") or INFORMATIVE_SENSORS)
        for col in seq_features:
            if col not in synth_feat.columns:
                synth_feat[col] = 0.0
        win_size = int(getattr(seq_model, "window_size_", meta.get("window_size", WINDOW_SIZE)))
        X_base, _, _ = build_sequence_windows(
            synth_feat, feature_cols=seq_features, window_size=win_size, last_only=True
        )
        p_mean, _, p_lo, p_hi = predict_with_uncertainty(
            seq_model, X_base, n_mc=20, alpha=0.10, clip_lower=0.0, clip_upper=float(RUL_CLIP)
        )
        pred_val, ci_low_val, ci_high_val = float(p_mean[0]), float(p_lo[0]), float(p_hi[0])
        used_label = f"{seq_model.__class__.__name__} ({pt_path.name})"

        base_det = float(predict_deterministic(seq_model, X_base, clip_lower=0.0, clip_upper=float(RUL_CLIP))[0])
        zero_df = synth_feat.copy()
        for s_col in key_sensors:
            zero_df[s_col] = 0.0
            X_zero, _, _ = build_sequence_windows(
                zero_df, feature_cols=seq_features, window_size=win_size, last_only=True
            )
            p_alt = float(predict_deterministic(seq_model, X_zero, clip_lower=0.0, clip_upper=float(RUL_CLIP))[0])
            contributions[s_col] = abs(p_alt - base_det)
            zero_df[s_col] = synth_feat[s_col]
    elif active_bundle.get("tabular_models"):
        tab_models = active_bundle["tabular_models"]
        tab_model = tab_models.get(artifact_name) or next(iter(tab_models.values()))
        model_features = list(getattr(tab_model, "feature_cols_", None) or meta.get("tab_feature_cols") or [])
        for col in model_features:
            if col not in synth_feat.columns:
                synth_feat[col] = 0.0
        last_row = synth_feat[model_features].iloc[[-1]]
        p_mean, _, p_lo, p_hi = tab_model.predict_with_uncertainty(last_row, alpha=0.10)
        pred_val = float(np.clip(p_mean[0], 0.0, float(RUL_CLIP)))
        ci_low_val = float(np.clip(p_lo[0], 0.0, float(RUL_CLIP)))
        ci_high_val = float(np.clip(p_hi[0], 0.0, float(RUL_CLIP)))
        used_label = f"Tabular ({tab_model.model_type})"
        for s_col in key_sensors:
            alt_df = synth_norm.copy()
            alt_df[s_col] = 0.0
            alt_feat, _ = add_tabular_features(
                alt_df,
                sensor_cols=list(INFORMATIVE_SENSORS),
                windows=ROLLING_WINDOWS,
                include_regime_onehot=(n_reg_onehot > 1),
                n_regimes=n_reg_onehot,
            )
            if is_unified:
                alt_feat = _attach_dataset_onehot(alt_feat, dataset_id)
            for col in model_features:
                if col not in alt_feat.columns:
                    alt_feat[col] = 0.0
            p_alt = float(np.clip(tab_model.predict(alt_feat[model_features].iloc[[-1]])[0], 0.0, float(RUL_CLIP)))
            contributions[s_col] = abs(p_alt - pred_val)
    else:
        pred_arr = _compute_heuristic_rul(synth_norm, INFORMATIVE_SENSORS)
        pred_val = float(pred_arr[-1])
        ci_low_val = float(max(0.0, pred_val - conformal_q))
        ci_high_val = float(min(float(RUL_CLIP), pred_val + conformal_q))
        for s_col in key_sensors:
            contributions[s_col] = abs(slider_z.get(s_col, 0.0)) * 12.0

    _render_status_badge(pred_val)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted RUL", f"{pred_val:.1f} cycles", delta=used_label)
    c2.metric("90% Confidence Band", f"[{ci_low_val:.1f}, {ci_high_val:.1f}]")
    c3.metric("Selected Regime", f"Regime {selected_regime}")

    if fault_probs_row is not None:
        st.caption("Latent Fault-Mode Probabilities for Current Snapshot")
        fc1, fc2 = st.columns(2)
        fc1.metric("Latent Fault Mode 0", f"{float(fault_probs_row.iloc[0]) * 100:.1f}%")
        if len(fault_probs_row) > 1:
            fc2.metric("Latent Fault Mode 1", f"{float(fault_probs_row.iloc[1]) * 100:.1f}%")

    contrib_df = (
        pd.DataFrame(
            [{"Sensor": k, "Impact (|Δ RUL| cycles)": v, "Description": SENSOR_DESCRIPTIONS.get(k, k)} for k, v in contributions.items()]
        )
        .sort_values("Impact (|Δ RUL| cycles)", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.barh(contrib_df["Sensor"], contrib_df["Impact (|Δ RUL| cycles)"], color="#2563eb")
    ax.set_xlabel("Estimated RUL Sensitivity (|Δ RUL| cycles when zeroed)")
    ax.set_title("Top Contributing Sensors for Current Operating Snapshot")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_tab_diagnostics() -> None:
    """Tab 1: Live Engine Diagnostics (Mode A replay + Mode B sliders)."""
    st.header("Live Engine Diagnostics & RUL Inference")
    col_ds, col_mod = st.columns(2)
    with col_ds:
        dataset_id = st.selectbox("Select C-MAPSS Subset", list(DATASETS), index=0, key="diag_dataset")
    with col_mod:
        model_track = st.selectbox(
            "Inference Model Track",
            list(MODEL_TRACK_OPTIONS.keys()),
            index=0,
            key="diag_model_track",
        )

    per_bundle = load_inference_artifacts(dataset_id)
    uni_bundle = load_unified_artifacts()

    if not per_bundle.get("available"):
        st.warning(
            f"Trained model artifacts for `{dataset_id}` were not found in `{MODELS_DIR / dataset_id}`. "
            "Showing normalized degradation proxy until `scripts/run_experiments.py` finishes."
        )

    train_df, test_df, y_test = load_dataset_raw(dataset_id)
    mode = st.radio(
        "Diagnostic Mode",
        ["Mode A: Real Test Engine Replay", "Mode B: Interactive Sensor Sliders"],
        horizontal=True,
        key="diag_mode",
    )
    if mode.startswith("Mode A"):
        _render_mode_a(dataset_id, model_track, train_df, test_df, y_test, per_bundle, uni_bundle)
    else:
        _render_mode_b(dataset_id, model_track, train_df, per_bundle, uni_bundle)


def render_tab_eda() -> None:
    """Tab 2: Dataset Progression, Constant Sensors Finding, and Operating Regimes."""
    st.header("Dataset & Exploratory Data Analysis (EDA)")

    progression_df = pd.DataFrame(
        [
            {"Dataset": "FD001", "Train Engines": 100, "Test Engines": 100, "Operating Conditions": 1, "Fault Modes": "1 (HPC)", "Difficulty": "Baseline"},
            {"Dataset": "FD003", "Train Engines": 100, "Test Engines": 100, "Operating Conditions": 1, "Fault Modes": "2 (HPC + Fan)", "Difficulty": "Multi-Fault"},
            {"Dataset": "FD002", "Train Engines": 260, "Test Engines": 259, "Operating Conditions": 6, "Fault Modes": "1 (HPC)", "Difficulty": "Multi-Regime"},
            {"Dataset": "FD004", "Train Engines": 249, "Test Engines": 248, "Operating Conditions": 6, "Fault Modes": "2 (HPC + Fan)", "Difficulty": "Hardest (Multi-Regime + Multi-Fault)"},
        ]
    )
    st.subheader("1. Four-Dataset Difficulty Progression (FD001 < FD003 < FD002 < FD004)")
    st.dataframe(progression_df, width="stretch", hide_index=True)
    st.caption("Note: Actual raw files contain train_FD004=249 engines and test_FD004=248 engines (correcting a historical typo in NASA's readme.txt).")

    st.subheader("2. Constant Sensors Discovery & Why FD002/FD004 Mask Them")
    st.info(
        "**Key EDA Finding:** Sensors **1, 5, 6, 10, 16, 18, 19** carry zero or near-zero degradation signal and are "
        "dropped, leaving **14 informative sensors**. On FD002/FD004, raw global standard deviation makes several of these "
        "sensors appear active only because the aircraft switches across 6 flight conditions (altitude, Mach, TRA) — "
        "within-regime Z-score normalization exposes their flat within-regime variance."
    )

    eda_summary = _read_json(RESULTS_DIR / "eda_sensor_summary.json")
    if isinstance(eda_summary, dict) and "datasets" in eda_summary:
        fd2_stats = eda_summary["datasets"].get("FD002", {}).get("sensor_stats", {})
        fd4_stats = eda_summary["datasets"].get("FD004", {}).get("sensor_stats", {})
        if fd2_stats and fd4_stats:
            const_rows = []
            for s_col in DROP_SENSORS_CANONICAL:
                s2 = fd2_stats.get(s_col, {})
                s4 = fd4_stats.get(s_col, {})
                const_rows.append(
                    {
                        "Sensor": s_col,
                        "FD002 Global Std": round(float(s2.get("global_std", 0.0)), 4),
                        "FD002 Within-Regime Std": round(float(s2.get("within_regime_std", 0.0)), 4),
                        "FD004 Global Std": round(float(s4.get("global_std", 0.0)), 4),
                        "FD004 Within-Regime Std": round(float(s4.get("within_regime_std", 0.0)), 4),
                        "Dropped": "Yes (Constant)",
                    }
                )
            st.caption("Empirical Proof: Constant Sensors Masked by 6 Flight Regimes in FD002 & FD004")
            st.dataframe(pd.DataFrame(const_rows), width="stretch", hide_index=True)

        with st.expander("View Full Sensor Variance Summary (eda_sensor_summary.json)"):
            st.json(eda_summary)

    st.subheader("3. EDA Figures")
    fig_var = FIGURES_DIR / "eda_sensor_variance.png"
    fig_reg = FIGURES_DIR / "eda_regimes_FD002_FD004.png"
    col1, col2 = st.columns(2)
    with col1:
        if fig_var.exists():
            st.image(str(fig_var), caption="Global vs Within-Regime Sensor Variance (Constant vs Informative)", width="stretch")
        else:
            st.warning("`eda_sensor_variance.png` not generated yet.")
    with col2:
        if fig_reg.exists():
            st.image(str(fig_reg), caption="FD002/FD004 6-Regime Operating Clusters & Normalization Effect", width="stretch")
        else:
            st.warning("`eda_regimes_FD002_FD004.png` not generated yet.")

    ds_fig_choice = st.selectbox("Inspect Engine Sensor Traces for Subset", list(DATASETS), key="eda_fig_ds")
    ds_fig = FIGURES_DIR / f"eda_sensors_{ds_fig_choice}.png"
    if ds_fig.exists():
        st.image(str(ds_fig), caption=f"Sample Engine Degradation Traces — {ds_fig_choice}", width="stretch")
    else:
        st.warning(f"`{ds_fig.name}` not generated yet.")


def render_tab_ablations_benchmarks() -> None:
    """Tab 3: Engineering Ablations & Published Literature Benchmarks."""
    st.header("Engineering Ablations & Literature Benchmarks")

    eval_view = st.radio(
        "Evaluation Target Scale",
        ["Clipped at 125 (Literature Standard)", "Unclipped Raw RUL"],
        horizontal=True,
        key="bench_scale_toggle",
    )
    is_clipped = eval_view.startswith("Clipped")

    st.subheader("1. Before vs After Regime Normalization (FD002 & FD004)")
    reg_df = _read_csv(RESULTS_DIR / "ablation_regime_norm.csv")
    if reg_df is not None:
        cols = [
            "dataset_id",
            "model",
            "condition",
            "rmse_clipped" if is_clipped else "rmse_unclipped",
            "mae_clipped" if is_clipped else "mae_unclipped",
            "nasa_score_clipped" if is_clipped else "nasa_score_unclipped",
            "late_rate_clipped" if is_clipped else "late_rate_unclipped",
        ]
        show_cols = [c for c in cols if c in reg_df.columns]
        st.dataframe(reg_df[show_cols].round(3), width="stretch", hide_index=True)
    else:
        st.info("Waiting for `reports/results/ablation_regime_norm.csv`.")
    fig_reg = FIGURES_DIR / "ablation_regime_norm.png"
    if fig_reg.exists():
        st.image(str(fig_reg), caption="Global Z-Score vs KMeans (k=6) Within-Regime Z-Score Impact", width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("2. Fault-Mode Clustering vs Implicit (FD003 & FD004)")
        fm_df = _read_csv(RESULTS_DIR / "ablation_fault_modes.csv")
        if fm_df is not None:
            cols = [
                "dataset_id",
                "model",
                "condition",
                "rmse_clipped" if is_clipped else "rmse_unclipped",
                "nasa_score_clipped" if is_clipped else "nasa_score_unclipped",
            ]
            show_cols = [c for c in cols if c in fm_df.columns]
            st.dataframe(fm_df[show_cols].round(3), width="stretch", hide_index=True)
        else:
            st.info("Waiting for `reports/results/ablation_fault_modes.csv`.")

    with col_b:
        st.subheader("3. Per-Dataset vs Unified Multi-Dataset Models")
        uni_df = _read_csv(RESULTS_DIR / "ablation_unified_vs_per_dataset.csv")
        if uni_df is not None:
            cols = [
                "dataset_id",
                "architecture",
                "per_dataset_rmse_clipped" if is_clipped else "per_dataset_rmse_unclipped",
                "unified_rmse_clipped" if is_clipped else "unified_rmse_unclipped",
                "per_dataset_nasa_clipped",
                "unified_nasa_clipped",
                "rmse_delta_unified_minus_per_ds",
            ]
            show_cols = [c for c in cols if c in uni_df.columns]
            st.dataframe(uni_df[show_cols].round(3), width="stretch", hide_index=True)
        else:
            st.info("Waiting for `reports/results/ablation_unified_vs_per_dataset.csv`.")

    st.subheader("4. Literature Benchmark Comparison (RMSE & NASA Asymmetric Score)")
    st.caption(
        "NASA Asymmetric Score (Saxena et al., 2008 Eq. 11) penalizes late overestimation "
        "(exp(d/10) - 1) exponentially harder than early underestimation (exp(-d/13) - 1). "
        "Negative % difference = lower error than published literature baseline."
    )
    bench_df = _read_csv(RESULTS_DIR / "benchmark_comparison.csv")
    if bench_df is not None:
        cols = [
            "dataset_id",
            "model",
            "scope",
            "rmse_clipped" if is_clipped else "rmse_unclipped",
            "nasa_score_clipped" if is_clipped else "nasa_score_unclipped",
            "babu_2016_rmse",
            "zheng_2017_rmse",
            "li_2018_rmse",
            "rmse_pct_diff_vs_zheng_2017",
            "rmse_pct_diff_vs_li_2018",
            "zheng_2017_nasa_score",
            "li_2018_nasa_score",
            "score_pct_diff_vs_zheng_2017",
            "score_pct_diff_vs_li_2018",
        ]
        show_cols = [c for c in cols if c in bench_df.columns]
        st.dataframe(bench_df[show_cols].round(2), width="stretch", hide_index=True)
    else:
        st.info("Waiting for `reports/results/benchmark_comparison.csv`.")

    fig_bench = FIGURES_DIR / "model_benchmark_comparison.png"
    fig_pred = FIGURES_DIR / "model_rul_predictions_FD001_FD004.png"
    if fig_bench.exists():
        st.image(str(fig_bench), caption="Model vs Published Literature Benchmarks (Babu 2016, Zheng 2017, Li 2018)", width="stretch")
    if fig_pred.exists():
        st.image(str(fig_pred), caption="Test Set Predicted vs True RUL Across FD001–FD004", width="stretch")


def _load_all_val_trajectories(choice: str) -> list[dict[str, Any]]:
    datasets = list(DATASETS) if choice == "All Combined (FD001–FD004)" else [choice]
    trajectories: list[dict[str, Any]] = []
    for ds in datasets:
        path = RESULTS_DIR / f"val_trajectories_{ds}.csv"
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        true_col = "true_rul_unclipped" if "true_rul_unclipped" in df.columns else "true_rul"
        for uid, grp in df.groupby("unit_id", sort=True):
            grp = grp.sort_values("cycle")
            trajectories.append(
                {
                    "unit_id": f"{ds}_{int(uid)}",
                    "cycles": grp["cycle"].to_numpy(dtype=float),
                    "true_rul": grp[true_col].to_numpy(dtype=float),
                    "pred_rul": grp["pred_rul"].to_numpy(dtype=float),
                }
            )
    return trajectories


def render_tab_maintenance_simulator() -> None:
    """Tab 4: Interactive 300-Engine Fleet Maintenance Cost Simulator."""
    st.header("300-Engine Fleet Maintenance Cost Simulator")
    st.markdown(
        "Simulate shifting from fixed-interval overhauls to RUL-driven predictive maintenance "
        "on held-out engine run-to-failure validation trajectories."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        unsched_cost = st.slider("Unscheduled Removal Cost ($)", 500_000, 1_200_000, 750_000, step=50_000, key="sim_unsched")
        aog_cost_day = st.slider("AOG Cost per Day ($)", 50_000, 300_000, 150_000, step=25_000, key="sim_aog")
    with c2:
        aog_days = st.slider("AOG Days per Failure", 1.0, 5.0, 2.0, step=0.5, key="sim_aog_days")
        sched_cost = st.slider("Scheduled Overhaul Cost ($)", 50_000, 300_000, 120_000, step=10_000, key="sim_sched")
    with c3:
        pred_thresh = st.slider("Predictive Pull Threshold (cycles)", 5, 45, 20, step=1, key="sim_thresh")
        fixed_interval = st.slider("Fixed-Interval Baseline (cycles)", 100, 220, 150, step=5, key="sim_fixed")

    ds_choice = st.selectbox(
        "Validation Fleet Trajectories",
        ["All Combined (FD001–FD004)", *DATASETS],
        index=0,
        key="sim_ds",
    )
    trajectories = _load_all_val_trajectories(ds_choice)
    if not trajectories:
        st.warning(
            "Validation trajectory files (`reports/results/val_trajectories_FD00X.csv`) were not found yet. "
            "Run `scripts/run_experiments.py` to populate live maintenance simulation."
        )
        sim_json = _read_json(RESULTS_DIR / "maintenance_simulation.json")
        if isinstance(sim_json, dict):
            st.json(sim_json)
        return

    from src.evaluation.maintenance import (
        CostAssumptions,
        simulate_fixed_interval_policy,
        simulate_predictive_policy,
    )

    costs = CostAssumptions(
        scheduled_maintenance_cost=float(sched_cost),
        unscheduled_removal_cost=float(unsched_cost),
        aog_cost_per_day=float(aog_cost_day),
        aog_days_per_failure=float(aog_days),
        fleet_size=300,
        cycles_per_engine_year=800.0,
    )
    pred_res = simulate_predictive_policy(
        trajectories,
        threshold_cycles=pred_thresh,
        costs=costs,
        baseline_interval_cycles=fixed_interval,
    )
    fixed_res = simulate_fixed_interval_policy(
        trajectories,
        interval_cycles=fixed_interval,
        costs=costs,
        reference_threshold=pred_thresh,
    )

    fixed_annual = float(fixed_res["annual_cost_300_fleet"])
    pred_annual = float(pred_res["annual_cost_300_fleet"])
    annual_savings = float(pred_res["annual_savings_300_fleet"])
    savings_pct = float(pred_res["savings_pct"])
    failures_avoided = int(pred_res["unplanned_failures_avoided_vs_fixed"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual Fleet Savings (300 Engines)", f"${annual_savings / 1e6:.2f}M", f"{savings_pct:.1f}% reduction")
    k2.metric("Unplanned Failures Avoided", f"{failures_avoided}", f"Pred {pred_res['unplanned_failures']} vs Fixed {fixed_res['unplanned_failures']}")
    k3.metric("Engines Pulled Early", f"{pred_res['engines_pulled_early']}", f"Avg wasted: {pred_res['avg_wasted_cycles']:.1f} cyc")
    k4.metric("Annual Cost (Pred vs Fixed)", f"${pred_annual / 1e6:.2f}M", f"Fixed: ${fixed_annual / 1e6:.2f}M")

    thresholds = [10, 15, 20, 25, 30, 35, 40]
    sweep_rows = []
    for th in thresholds:
        r = simulate_predictive_policy(
            trajectories,
            threshold_cycles=th,
            costs=costs,
            baseline_interval_cycles=fixed_interval,
        )
        sweep_rows.append(
            {
                "Threshold (cyc)": th,
                "Annual Cost ($M)": round(float(r["annual_cost_300_fleet"]) / 1e6, 2),
                "Annual Savings ($M)": round(float(r["annual_savings_300_fleet"]) / 1e6, 2),
                "Savings (%)": round(float(r["savings_pct"]), 1),
                "Unplanned Failures": int(r["unplanned_failures"]),
                "Early Pulls": int(r["engines_pulled_early"]),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax1.bar(
        ["Fixed-Interval", "Predictive RUL"],
        [fixed_annual / 1e6, pred_annual / 1e6],
        color=["#dc2626", "#16a34a"],
        width=0.5,
    )
    ax1.set_ylabel("Annual 300-Engine Fleet Cost ($M)")
    ax1.set_title("Annual Fleet Cost Comparison")
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(sweep_df["Threshold (cyc)"], sweep_df["Annual Cost ($M)"], marker="o", color="#1d4ed8", label="Predictive Annual Cost ($M)")
    ax2.axhline(fixed_annual / 1e6, color="#dc2626", linestyle="--", label=f"Fixed {fixed_interval}-cyc ($M)")
    ax2.set_xlabel("Predictive Pull Threshold (cycles)")
    ax2.set_ylabel("Annual Fleet Cost ($M)")
    ax2.set_title("Threshold Sensitivity Tradeoff")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader("Threshold Sensitivity Breakdown")
    st.dataframe(sweep_df, width="stretch", hide_index=True)

    fig_maint = FIGURES_DIR / "maintenance_cost_comparison.png"
    if fig_maint.exists():
        with st.expander("View Saved Maintenance Simulation Report Figure"):
            st.image(str(fig_maint), caption="Baseline Simulation Report Output", width="stretch")


def main() -> None:
    st.set_page_config(
        page_title="Turbofan Predictive Maintenance & RUL",
        page_icon="✈️",
        layout="wide",
    )
    st.title("Turbofan Engine RUL Prediction & Predictive Maintenance")
    st.markdown(
        "**NASA C-MAPSS (FD001–FD004)** | Mitigating **$500K–$1M+** unscheduled engine removals and "
        "**$150K/day** AOG penalties across a **300-engine fleet**."
    )

    with st.sidebar:
        st.header("System Artifacts")
        for ds in DATASETS:
            ready = (MODELS_DIR / ds / "best_sequence.pt").exists()
            st.write(f"{'✅' if ready else '⏳'} `{ds}` model bundle")
        uni_ready = (MODELS_DIR / "unified" / "sequence_bilstm.pt").exists()
        bench_ready = (RESULTS_DIR / "benchmark_comparison.csv").exists()
        sim_ready = (RESULTS_DIR / "maintenance_simulation.json").exists()
        st.write(f"{'✅' if uni_ready else '⏳'} Unified multi-dataset bundle")
        st.write(f"{'✅' if bench_ready else '⏳'} Benchmark CSV")
        st.write(f"{'✅' if sim_ready else '⏳'} Maintenance simulation JSON")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Live Engine Diagnostics",
            "2. Dataset & EDA Insights",
            "3. Ablations & Benchmarks",
            "4. 300-Engine Fleet Simulator",
        ]
    )
    with tab1:
        render_tab_diagnostics()
    with tab2:
        render_tab_eda()
    with tab3:
        render_tab_ablations_benchmarks()
    with tab4:
        render_tab_maintenance_simulator()


if __name__ == "__main__":
    main()
