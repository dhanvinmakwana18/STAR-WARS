"""End-to-end NASA C-MAPSS experiment suite (Phases 2, 3, 4, 5).

Runs:
- Phase 2 Ablation (FD002 & FD004): Global Z-score vs KMeans(k=6) Within-Regime Normalization
- Phase 3 Ablation (FD003 & FD004): Implicit vs Explicit Latent Fault-Mode Clustering (k=2)
- Phase 4 Modeling (FD001-FD004): Random Forest, XGBoost, BiLSTM, Deep 1D-CNN + Unified Models
- Phase 5 Evaluation: Clipped-at-125 & unclipped metrics, NASA Score, Zheng (2017) & Li (2018)
  benchmark comparisons, validation run-to-failure trajectories, checkpoints, and figures.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MPL_CACHE = _PROJECT_ROOT / ".cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DATASET_META,
    DATASETS,
    FIGURES_DIR,
    INFORMATIVE_SENSORS,
    MODELS_DIR,
    OP_SETTING_COLS,
    RANDOM_SEED,
    RESULTS_DIR,
    ROLLING_WINDOWS,
    RUL_CLIP,
    WINDOW_SIZE,
    ensure_project_dirs,
)
from src.data.loader import clip_rul, load_cmapss_dataset, train_val_engine_split
from src.data.preprocessing import RegimeNormalizer
from src.evaluation.benchmarks import PUBLISHED_BENCHMARKS, compare_against_benchmarks
from src.evaluation.metrics import evaluate_predictions
from src.features.engineering import (
    FaultModeClusterer,
    add_tabular_features,
    build_sequence_windows,
)
from src.models.sequence import SequenceRULModel, predict_with_uncertainty
from src.models.tabular import TabularRULModel


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sanitize_json(obj: Any) -> Any:
    """Recursively convert NaN/Inf floats and numpy scalars to strict JSON values."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    return obj


def _add_dataset_onehot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cols: list[str] = []
    for ds in DATASETS:
        col = f"dataset_{ds}"
        out[col] = (out["dataset_id"] == ds).astype(np.float32)
        cols.append(col)
    return out, cols


def _extract_test_last_rows(test_feat_df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one final-cycle row per test engine sorted by unit_id ascending."""
    last_rows = (
        test_feat_df.sort_values(["unit_id", "cycle"], kind="mergesort")
        .groupby("unit_id", sort=True)
        .tail(1)
        .sort_values("unit_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return last_rows


def _eval_both_scales(
    y_test_raw: np.ndarray,
    y_pred: np.ndarray,
    ci_low: np.ndarray | None = None,
    ci_high: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate predictions on both clipped-at-125 and unclipped raw RUL scales."""
    y_pred_nonneg = np.maximum(0.0, np.asarray(y_pred, dtype=np.float64))
    clipped = evaluate_predictions(y_test_raw, y_pred_nonneg, clip_cap=RUL_CLIP, clip_pred=True)
    unclipped = evaluate_predictions(y_test_raw, y_pred_nonneg, clip_cap=None, clip_pred=False)

    out: dict[str, Any] = {
        "rmse_clipped": clipped["rmse"],
        "mae_clipped": clipped["mae"],
        "nasa_score_clipped": clipped["nasa_score"],
        "mean_nasa_score_clipped": clipped["mean_nasa_score"],
        "early_pct_clipped": clipped["early_pct"],
        "late_pct_clipped": clipped["late_pct"],
        "late_rate_clipped": clipped["late_rate"],
        "mean_bias_clipped": clipped["mean_bias"],
        "rmse_unclipped": unclipped["rmse"],
        "mae_unclipped": unclipped["mae"],
        "nasa_score_unclipped": unclipped["nasa_score"],
        "mean_nasa_score_unclipped": unclipped["mean_nasa_score"],
        "early_pct_unclipped": unclipped["early_pct"],
        "late_pct_unclipped": unclipped["late_pct"],
        "late_rate_unclipped": unclipped["late_rate"],
        "mean_bias_unclipped": unclipped["mean_bias"],
    }

    if ci_low is not None and ci_high is not None:
        y_clip = clip_rul(y_test_raw, cap=RUL_CLIP)
        lo = np.asarray(ci_low, dtype=np.float64)
        hi = np.asarray(ci_high, dtype=np.float64)
        out["ci90_test_coverage_clipped"] = float(np.mean((y_clip >= lo) & (y_clip <= hi)))
        out["ci90_test_coverage_unclipped"] = float(
            np.mean((y_test_raw >= lo) & (y_test_raw <= hi))
        )
        out["ci90_mean_width"] = float(np.mean(hi - lo))
    else:
        out["ci90_test_coverage_clipped"] = np.nan
        out["ci90_test_coverage_unclipped"] = np.nan
        out["ci90_mean_width"] = np.nan

    return out


def _prepare_dataset_bundle(
    dataset_id: str,
    tr_raw: pd.DataFrame,
    va_raw: pd.DataFrame,
    te_raw: pd.DataFrame,
    n_regimes: int,
    use_regime_norm: bool,
    use_fault_clusterer: bool = False,
) -> dict[str, Any]:
    """Normalize and engineer tabular + sequence features cleanly without leakage."""
    normalizer = RegimeNormalizer(
        n_regimes=n_regimes,
        sensor_cols=list(INFORMATIVE_SENSORS),
        op_cols=list(OP_SETTING_COLS),
        use_regime_norm=use_regime_norm,
        random_state=RANDOM_SEED,
    ).fit(tr_raw)

    tr_n = normalizer.transform(tr_raw)
    va_n = normalizer.transform(va_raw)
    te_n = normalizer.transform(te_raw)

    fault_clusterer: FaultModeClusterer | None = None
    fault_cols: list[str] = []
    if use_fault_clusterer:
        fault_clusterer = FaultModeClusterer(
            n_modes=2,
            sensor_cols=list(INFORMATIVE_SENSORS),
            tail_cycles=25,
            random_state=RANDOM_SEED,
        ).fit(tr_n)
        tr_n, fault_cols = fault_clusterer.transform(tr_n)
        va_n, _ = fault_clusterer.transform(va_n)
        te_n, _ = fault_clusterer.transform(te_n)

    include_regime_onehot = bool(use_regime_norm and n_regimes > 1)
    tr_f, base_tab_cols = add_tabular_features(
        tr_n,
        sensor_cols=list(INFORMATIVE_SENSORS),
        windows=ROLLING_WINDOWS,
        include_regime_onehot=include_regime_onehot,
        n_regimes=n_regimes,
    )
    va_f, _ = add_tabular_features(
        va_n,
        sensor_cols=list(INFORMATIVE_SENSORS),
        windows=ROLLING_WINDOWS,
        include_regime_onehot=include_regime_onehot,
        n_regimes=n_regimes,
    )
    te_f, _ = add_tabular_features(
        te_n,
        sensor_cols=list(INFORMATIVE_SENSORS),
        windows=ROLLING_WINDOWS,
        include_regime_onehot=include_regime_onehot,
        n_regimes=n_regimes,
    )

    tab_cols = base_tab_cols + fault_cols
    regime_onehot_cols = (
        [f"regime_{r}" for r in range(n_regimes)] if include_regime_onehot else []
    )
    if n_regimes > 1:
        seq_cols = list(INFORMATIVE_SENSORS) + list(OP_SETTING_COLS) + regime_onehot_cols + fault_cols
    else:
        # Single operating condition (FD001/FD003): avoid noisy jitter in op_settings
        seq_cols = list(INFORMATIVE_SENSORS) + fault_cols

    return {
        "dataset_id": dataset_id,
        "normalizer": normalizer,
        "fault_clusterer": fault_clusterer,
        "tr_feat": tr_f,
        "va_feat": va_f,
        "te_feat": te_f,
        "te_last": _extract_test_last_rows(te_f),
        "tab_cols": tab_cols,
        "seq_cols": seq_cols,
    }


def _train_eval_xgb(
    bundle: dict[str, Any],
    y_test_raw: np.ndarray,
) -> tuple[TabularRULModel, dict[str, Any], np.ndarray]:
    tr_f = bundle["tr_feat"]
    va_f = bundle["va_feat"]
    te_last = bundle["te_last"]
    tab_cols = bundle["tab_cols"]

    model = TabularRULModel(model_type="xgboost", feature_cols=tab_cols, random_state=RANDOM_SEED)
    model.fit(tr_f, tr_f["RUL"], va_f, va_f["RUL"], feature_cols=tab_cols)

    val_pred = model.predict(va_f)
    val_rmse = float(np.sqrt(np.mean((va_f["RUL"].to_numpy(dtype=np.float64) - val_pred) ** 2)))
    pred, _, ci_lo, ci_hi = model.predict_with_uncertainty(te_last)
    metrics = _eval_both_scales(y_test_raw, pred, ci_lo, ci_hi)
    metrics["val_rmse"] = val_rmse
    return model, metrics, pred


def _train_eval_rf(
    bundle: dict[str, Any],
    y_test_raw: np.ndarray,
) -> tuple[TabularRULModel, dict[str, Any], np.ndarray]:
    tr_f = bundle["tr_feat"]
    va_f = bundle["va_feat"]
    te_last = bundle["te_last"]
    tab_cols = bundle["tab_cols"]

    model = TabularRULModel(model_type="random_forest", feature_cols=tab_cols, random_state=RANDOM_SEED)
    model.fit(tr_f, tr_f["RUL"], va_f, va_f["RUL"], feature_cols=tab_cols)

    val_pred = model.predict(va_f)
    val_rmse = float(np.sqrt(np.mean((va_f["RUL"].to_numpy(dtype=np.float64) - val_pred) ** 2)))
    pred, _, ci_lo, ci_hi = model.predict_with_uncertainty(te_last)
    metrics = _eval_both_scales(y_test_raw, pred, ci_lo, ci_hi)
    metrics["val_rmse"] = val_rmse
    return model, metrics, pred


def _train_eval_seq(
    arch: str,
    bundle: dict[str, Any],
    y_test_raw: np.ndarray,
    epochs: int = 30,
    patience: int = 7,
    lr: float = 1.2e-3,
    checkpoint_path: Path | None = None,
) -> tuple[SequenceRULModel, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    tr_f = bundle["tr_feat"]
    va_f = bundle["va_feat"]
    te_f = bundle["te_feat"]
    seq_cols = bundle["seq_cols"]

    X_tr, y_tr, _ = build_sequence_windows(
        tr_f, seq_cols, window_size=WINDOW_SIZE, target_col="RUL", last_only=False
    )
    X_va, y_va, _ = build_sequence_windows(
        va_f, seq_cols, window_size=WINDOW_SIZE, target_col="RUL", last_only=False
    )
    X_te, _, meta_te = build_sequence_windows(
        te_f, seq_cols, window_size=WINDOW_SIZE, target_col="RUL", last_only=True
    )

    expected_units = np.arange(1, len(y_test_raw) + 1, dtype=np.int64)
    if not np.array_equal(meta_te["unit_id"].to_numpy(dtype=np.int64), expected_units):
        order = np.argsort(meta_te["unit_id"].to_numpy(dtype=np.int64))
        X_te = X_te[order]

    batch_size = 512 if X_tr.shape[0] > 30_000 else 256
    model = SequenceRULModel(
        arch=arch,
        n_features=len(seq_cols),
        seq_len=WINDOW_SIZE,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-3,
        patience=patience,
        n_mc=30,
        alpha=0.10,
        seed=RANDOM_SEED,
    )
    model.fit(
        X_tr,
        y_tr,
        X_va,
        y_va,
        feature_cols=seq_cols,
        checkpoint_path=checkpoint_path,
        verbose=False,
    )

    pred, _, ci_lo, ci_hi = model.predict_with_uncertainty(X_te, n_mc=30, alpha=0.10)
    metrics = _eval_both_scales(y_test_raw, pred, ci_lo, ci_hi)
    metrics["val_rmse"] = float(model.history_["best_val_rmse"])
    metrics["best_epoch"] = int(model.history_["best_epoch"])
    return model, metrics, pred, ci_lo, ci_hi


def _plot_ablation_regime_norm(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=160)
    labels = [f"{r.dataset_id}\n{r.model}" for r in df.itertuples() if r.condition == "Global Z-score (Before)"]
    before = df[df["condition"] == "Global Z-score (Before)"].reset_index(drop=True)
    after = df[df["condition"] == "Regime KMeans(k=6) Z-score (After)"].reset_index(drop=True)

    x = np.arange(len(labels))
    w = 0.36

    ax = axes[0]
    ax.bar(x - w / 2, before["rmse_clipped"], width=w, label="Before: Global Z-score", color="#d95f02")
    ax.bar(x + w / 2, after["rmse_clipped"], width=w, label="After: Within-Regime Z-score (k=6)", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test RMSE (Clipped @ 125)")
    ax.set_title("Operating Regime Normalization Impact: RMSE")
    ax.set_ylim(0.0, float(before["rmse_clipped"].max()) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    for i in range(len(x)):
        pct = 100.0 * (after.loc[i, "rmse_clipped"] - before.loc[i, "rmse_clipped"]) / before.loc[i, "rmse_clipped"]
        ax.text(x[i] - w / 2, before.loc[i, "rmse_clipped"] + 0.45, f"{before.loc[i, 'rmse_clipped']:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, after.loc[i, "rmse_clipped"] + 0.45, f"{after.loc[i, 'rmse_clipped']:.1f}\n({pct:+.1f}%)", ha="center", fontsize=8, fontweight="bold")

    ax = axes[1]
    ax.bar(x - w / 2, before["nasa_score_clipped"], width=w, label="Before: Global Z-score", color="#d95f02")
    ax.bar(x + w / 2, after["nasa_score_clipped"], width=w, label="After: Within-Regime Z-score (k=6)", color="#1b9e77")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("NASA Asymmetric Score (log scale)")
    ax.set_title("Operating Regime Normalization Impact: NASA Score")
    ax.set_ylim(float(after["nasa_score_clipped"].min()) * 0.72, float(before["nasa_score_clipped"].max()) * 1.65)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="upper left")
    for i in range(len(x)):
        pct = 100.0 * (after.loc[i, "nasa_score_clipped"] - before.loc[i, "nasa_score_clipped"]) / before.loc[i, "nasa_score_clipped"]
        ax.text(x[i] - w / 2, before.loc[i, "nasa_score_clipped"] * 1.10, f"{before.loc[i, 'nasa_score_clipped']:.0f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, after.loc[i, "nasa_score_clipped"] * 1.10, f"{after.loc[i, 'nasa_score_clipped']:.0f}\n({pct:+.1f}%)", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Phase 2 Engineering Breakthrough: Regime-Conditioned Z-Score Normalization (FD002 & FD004)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_benchmark_comparison(bench_df: pd.DataFrame, out_path: Path) -> None:
    models_to_plot = [
        ("RandomForest", "#8da0cb"),
        ("XGBoost", "#fc8d62"),
        ("BiLSTM", "#66c2a5"),
        ("Deep1DCNN", "#1f78b4"),
        ("Unified 1D-CNN", "#6a3d9a"),
        ("Unified BiLSTM", "#33a02c"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=160)
    x = np.arange(len(DATASETS))
    n_bars = len(models_to_plot) + 2  # + Zheng 2017 & Li 2018
    w = 0.84 / n_bars

    for panel_idx, (metric_col, ref_key, ylabel, title, log_y) in enumerate(
        [
            ("rmse_clipped", "rmse", "Test RMSE (cycles, Clipped @ 125)", "Test RMSE vs Published Benchmarks", False),
            ("nasa_score_clipped", "nasa_score", "NASA Asymmetric Score (log scale)", "NASA Asymmetric Score vs Published Benchmarks", True),
        ]
    ):
        ax = axes[panel_idx]
        offset = -(n_bars - 1) / 2.0
        for m_name, color in models_to_plot:
            sub = bench_df[bench_df["model"] == m_name].set_index("dataset_id").reindex(DATASETS)
            vals = sub[metric_col].to_numpy(dtype=float)
            ax.bar(x + offset * w, vals, width=w, label=m_name, color=color)
            offset += 1.0

        zheng_vals = [PUBLISHED_BENCHMARKS["Zheng et al. (2017) Deep LSTM"][ref_key][ds] for ds in DATASETS]
        li_vals = [PUBLISHED_BENCHMARKS["Li et al. (2018) DCNN"][ref_key][ds] for ds in DATASETS]
        ax.bar(x + offset * w, zheng_vals, width=w, label="Zheng et al. (2017) LSTM", color="#555555", hatch="//", edgecolor="white")
        offset += 1.0
        ax.bar(x + offset * w, li_vals, width=w, label="Li et al. (2018) DCNN", color="#111111", hatch="..", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(180.0, max(li_vals) * 3.4)
            ax.grid(axis="y", alpha=0.3, which="both")
        else:
            ax.set_ylim(0.0, max(zheng_vals) * 1.24)
            ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="upper left")

    fig.suptitle("C-MAPSS Benchmark Comparison Across All Four Sub-Datasets (FD001–FD004)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rul_predictions(pred_store: dict[str, dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=160)
    axes_flat = axes.ravel()

    for idx, ds in enumerate(DATASETS):
        ax = axes_flat[idx]
        rec = pred_store[ds]
        y_true = clip_rul(rec["y_test_raw"], cap=RUL_CLIP)
        pred = np.clip(rec["pred"], 0.0, float(RUL_CLIP))
        lo = np.clip(rec["ci_low"], 0.0, float(RUL_CLIP))
        hi = np.clip(rec["ci_high"], 0.0, float(RUL_CLIP) + 20.0)

        order = np.argsort(y_true)
        x = np.arange(len(order))

        ax.fill_between(x, lo[order], hi[order], color="#a6cee3", alpha=0.45, label="90% Conformal MC Band")
        ax.plot(x, y_true[order], color="#111111", linewidth=1.8, label="True RUL (Clipped @ 125)")
        ax.plot(x, pred[order], color="#1f78b4", linewidth=1.3, alpha=0.9, label=f"Predicted ({rec['best_arch']})")

        ax.set_title(
            f"{ds} ({DATASET_META[ds]['op_conditions']} op, {DATASET_META[ds]['fault_modes']} fault) — "
            f"RMSE={rec['rmse_clipped']:.2f} | NASA={rec['nasa_score_clipped']:.1f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("Test Engine (Sorted by True RUL)")
        ax.set_ylabel("RUL (Cycles)")
        ax.set_ylim(-3, 145)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Test-Set RUL Predictions with 90% MC-Dropout + Conformal Prediction Intervals", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if "--plots-only" in sys.argv[1:]:
        ensure_project_dirs()
        ablation_regime_df = pd.read_csv(RESULTS_DIR / "ablation_regime_norm.csv")
        bench_df = pd.read_csv(RESULTS_DIR / "benchmark_comparison.csv")
        _plot_ablation_regime_norm(ablation_regime_df, FIGURES_DIR / "ablation_regime_norm.png")
        _plot_benchmark_comparison(bench_df, FIGURES_DIR / "model_benchmark_comparison.png")
        _log("Regenerated ablation_regime_norm.png and model_benchmark_comparison.png from saved CSVs.")
        return

    t_start = time.perf_counter()
    ensure_project_dirs()
    np.random.seed(RANDOM_SEED)

    _log("=== Loading C-MAPSS FD001..FD004 and building leak-free engine splits ===")
    raw_splits: dict[str, dict[str, Any]] = {}
    for ds in DATASETS:
        tr_full, te_df, y_test_raw = load_cmapss_dataset(ds, rul_clip=RUL_CLIP)
        tr_s, va_s = train_val_engine_split(tr_full, val_ratio=0.2, seed=RANDOM_SEED)
        assert set(tr_s["unit_id"]).isdisjoint(set(va_s["unit_id"]))
        raw_splits[ds] = {
            "tr": tr_s,
            "va": va_s,
            "te": te_df,
            "y_test_raw": y_test_raw,
        }
        _log(
            f"  {ds}: train_engines={tr_s['unit_id'].nunique()}, "
            f"val_engines={va_s['unit_id'].nunique()}, test_engines={len(y_test_raw)}"
        )

    # ------------------------------------------------------------------
    # Prepare canonical per-dataset bundles (Regime Norm ON)
    # ------------------------------------------------------------------
    bundles_norm: dict[str, dict[str, Any]] = {}
    for ds in DATASETS:
        n_regimes = int(DATASET_META[ds]["op_conditions"])
        bundles_norm[ds] = _prepare_dataset_bundle(
            ds,
            raw_splits[ds]["tr"],
            raw_splits[ds]["va"],
            raw_splits[ds]["te"],
            n_regimes=n_regimes,
            use_regime_norm=True,
            use_fault_clusterer=False,
        )

    # ------------------------------------------------------------------
    # PHASE 2 ABLATION (FD002 & FD004): Global vs Within-Regime Z-score
    # ------------------------------------------------------------------
    _log("\n=== Phase 2 Ablation: Global Z-score vs Within-Regime KMeans(k=6) ===")
    ablation_regime_rows: list[dict[str, Any]] = []
    # Cache Phase 4 After-Regime-Norm XGBoost & 1D-CNN for FD002/FD004 to reuse cleanly
    cached_models: dict[tuple[str, str], dict[str, Any]] = {}

    for ds in ("FD002", "FD004"):
        y_te = raw_splits[ds]["y_test_raw"]
        bundle_global = _prepare_dataset_bundle(
            ds,
            raw_splits[ds]["tr"],
            raw_splits[ds]["va"],
            raw_splits[ds]["te"],
            n_regimes=6,
            use_regime_norm=False,
            use_fault_clusterer=False,
        )
        # Global XGBoost
        _, m_xgb_glob, _ = _train_eval_xgb(bundle_global, y_te)
        ablation_regime_rows.append(
            {"dataset_id": ds, "model": "XGBoost", "condition": "Global Z-score (Before)", **m_xgb_glob}
        )
        # Global 1D-CNN
        _, m_cnn_glob, _, _, _ = _train_eval_seq(
            "1d_cnn", bundle_global, y_te, epochs=30, patience=6, lr=1.2e-3
        )
        ablation_regime_rows.append(
            {"dataset_id": ds, "model": "Deep1DCNN", "condition": "Global Z-score (Before)", **m_cnn_glob}
        )

        # After: Regime KMeans(k=6) Z-score (also saved as Phase 4 models)
        ds_dir = MODELS_DIR / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        xgb_after, m_xgb_after, p_xgb_after = _train_eval_xgb(bundles_norm[ds], y_te)
        xgb_after.save(ds_dir / "tabular_xgb.joblib")
        cached_models[(ds, "XGBoost")] = {"model": xgb_after, "metrics": m_xgb_after, "pred": p_xgb_after}
        ablation_regime_rows.append(
            {"dataset_id": ds, "model": "XGBoost", "condition": "Regime KMeans(k=6) Z-score (After)", **m_xgb_after}
        )

        cnn_after, m_cnn_after, p_cnn_after, lo_cnn_after, hi_cnn_after = _train_eval_seq(
            "1d_cnn",
            bundles_norm[ds],
            y_te,
            epochs=32,
            patience=7,
            lr=1.2e-3,
            checkpoint_path=ds_dir / "sequence_1dcnn.pt",
        )
        cached_models[(ds, "Deep1DCNN")] = {
            "model": cnn_after,
            "metrics": m_cnn_after,
            "pred": p_cnn_after,
            "ci_low": lo_cnn_after,
            "ci_high": hi_cnn_after,
        }
        ablation_regime_rows.append(
            {"dataset_id": ds, "model": "Deep1DCNN", "condition": "Regime KMeans(k=6) Z-score (After)", **m_cnn_after}
        )
        _log(
            f"  {ds} Regime Norm | XGB RMSE: {m_xgb_glob['rmse_clipped']:.2f} -> {m_xgb_after['rmse_clipped']:.2f} "
            f"| 1D-CNN RMSE: {m_cnn_glob['rmse_clipped']:.2f} -> {m_cnn_after['rmse_clipped']:.2f} "
            f"(Score: {m_cnn_glob['nasa_score_clipped']:.1f} -> {m_cnn_after['nasa_score_clipped']:.1f})"
        )

    ablation_regime_df = pd.DataFrame(ablation_regime_rows)
    ablation_regime_df.to_csv(RESULTS_DIR / "ablation_regime_norm.csv", index=False)
    _plot_ablation_regime_norm(ablation_regime_df, FIGURES_DIR / "ablation_regime_norm.png")

    # ------------------------------------------------------------------
    # PHASE 4 PER-DATASET REMAINING MODELS (FD001..FD004)
    # ------------------------------------------------------------------
    _log("\n=== Phase 4 Per-Dataset Modeling (RF, XGBoost, BiLSTM, Deep 1D-CNN) ===")
    for ds in DATASETS:
        y_te = raw_splits[ds]["y_test_raw"]
        ds_dir = MODELS_DIR / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        bundle = bundles_norm[ds]
        bundle["normalizer"].save(ds_dir / "normalizer.joblib")

        # Random Forest
        rf_model, m_rf, p_rf = _train_eval_rf(bundle, y_te)
        rf_model.save(ds_dir / "tabular_rf.joblib")
        cached_models[(ds, "RandomForest")] = {"model": rf_model, "metrics": m_rf, "pred": p_rf}

        # XGBoost (if not already run in Phase 2)
        if (ds, "XGBoost") not in cached_models:
            xgb_model, m_xgb, p_xgb = _train_eval_xgb(bundle, y_te)
            xgb_model.save(ds_dir / "tabular_xgb.joblib")
            cached_models[(ds, "XGBoost")] = {"model": xgb_model, "metrics": m_xgb, "pred": p_xgb}

        # Deep 1D-CNN (if not already run in Phase 2)
        if (ds, "Deep1DCNN") not in cached_models:
            cnn_model, m_cnn, p_cnn, lo_cnn, hi_cnn = _train_eval_seq(
                "1d_cnn",
                bundle,
                y_te,
                epochs=32,
                patience=7,
                lr=1.2e-3,
                checkpoint_path=ds_dir / "sequence_1dcnn.pt",
            )
            cached_models[(ds, "Deep1DCNN")] = {
                "model": cnn_model,
                "metrics": m_cnn,
                "pred": p_cnn,
                "ci_low": lo_cnn,
                "ci_high": hi_cnn,
            }

        # BiLSTM + Attention
        lstm_model, m_lstm, p_lstm, lo_lstm, hi_lstm = _train_eval_seq(
            "bilstm",
            bundle,
            y_te,
            epochs=32,
            patience=7,
            lr=1.2e-3,
            checkpoint_path=ds_dir / "sequence_bilstm.pt",
        )
        cached_models[(ds, "BiLSTM")] = {
            "model": lstm_model,
            "metrics": m_lstm,
            "pred": p_lstm,
            "ci_low": lo_lstm,
            "ci_high": hi_lstm,
        }

        _log(
            f"  {ds} Summary (clipped RMSE / NASA): "
            f"RF={cached_models[(ds, 'RandomForest')]['metrics']['rmse_clipped']:.2f}/{cached_models[(ds, 'RandomForest')]['metrics']['nasa_score_clipped']:.0f}, "
            f"XGB={cached_models[(ds, 'XGBoost')]['metrics']['rmse_clipped']:.2f}/{cached_models[(ds, 'XGBoost')]['metrics']['nasa_score_clipped']:.0f}, "
            f"BiLSTM={m_lstm['rmse_clipped']:.2f}/{m_lstm['nasa_score_clipped']:.0f}, "
            f"1D-CNN={cached_models[(ds, 'Deep1DCNN')]['metrics']['rmse_clipped']:.2f}/{cached_models[(ds, 'Deep1DCNN')]['metrics']['nasa_score_clipped']:.0f}"
        )

    # ------------------------------------------------------------------
    # PHASE 3 ABLATION (FD003 & FD004): Implicit vs Explicit Fault Mode Clusterer
    # ------------------------------------------------------------------
    _log("\n=== Phase 3 Ablation: Implicit vs Explicit Latent Fault Mode Clustering (FD003 & FD004) ===")
    ablation_fault_rows: list[dict[str, Any]] = []
    for ds in ("FD003", "FD004"):
        y_te = raw_splits[ds]["y_test_raw"]
        ds_dir = MODELS_DIR / ds
        n_regimes = int(DATASET_META[ds]["op_conditions"])

        # Implicit baseline rows (reuse Phase 4 XGBoost & Deep1DCNN)
        ablation_fault_rows.append(
            {"dataset_id": ds, "model": "XGBoost", "condition": "Implicit (No Fault Cluster)", **cached_models[(ds, "XGBoost")]["metrics"]}
        )
        ablation_fault_rows.append(
            {"dataset_id": ds, "model": "Deep1DCNN", "condition": "Implicit (No Fault Cluster)", **cached_models[(ds, "Deep1DCNN")]["metrics"]}
        )

        # Explicit FaultModeClusterer(k=2)
        bundle_fault = _prepare_dataset_bundle(
            ds,
            raw_splits[ds]["tr"],
            raw_splits[ds]["va"],
            raw_splits[ds]["te"],
            n_regimes=n_regimes,
            use_regime_norm=True,
            use_fault_clusterer=True,
        )
        bundle_fault["fault_clusterer"].save(ds_dir / "fault_clusterer.joblib")

        _, m_xgb_fm, _ = _train_eval_xgb(bundle_fault, y_te)
        ablation_fault_rows.append(
            {"dataset_id": ds, "model": "XGBoost", "condition": "Explicit FaultModeClusterer(k=2)", **m_xgb_fm}
        )

        cnn_fm, m_cnn_fm, p_cnn_fm, lo_fm, hi_fm = _train_eval_seq(
            "1d_cnn", bundle_fault, y_te, epochs=32, patience=7, lr=1.2e-3
        )
        ablation_fault_rows.append(
            {"dataset_id": ds, "model": "Deep1DCNN", "condition": "Explicit FaultModeClusterer(k=2)", **m_cnn_fm}
        )
        _log(
            f"  {ds} FaultMode | XGB RMSE: {cached_models[(ds, 'XGBoost')]['metrics']['rmse_clipped']:.2f} -> {m_xgb_fm['rmse_clipped']:.2f} "
            f"| 1D-CNN RMSE: {cached_models[(ds, 'Deep1DCNN')]['metrics']['rmse_clipped']:.2f} -> {m_cnn_fm['rmse_clipped']:.2f}"
        )

    ablation_fault_df = pd.DataFrame(ablation_fault_rows)
    ablation_fault_df.to_csv(RESULTS_DIR / "ablation_fault_modes.csv", index=False)

    # ------------------------------------------------------------------
    # PHASE 4 UNIFIED MULTI-DATASET MODELS (FD001 + FD002 + FD003 + FD004)
    # ------------------------------------------------------------------
    _log("\n=== Phase 4 Unified Multi-Dataset Modeling (FD001-FD004 combined) ===")
    uni_dir = MODELS_DIR / "unified"
    uni_dir.mkdir(parents=True, exist_ok=True)

    tr_uni_raw = pd.concat(
        [raw_splits[ds]["tr"].assign(dataset_id=ds) for ds in DATASETS], ignore_index=True
    )
    va_uni_raw = pd.concat(
        [raw_splits[ds]["va"].assign(dataset_id=ds) for ds in DATASETS], ignore_index=True
    )
    te_uni_raw = pd.concat(
        [raw_splits[ds]["te"].assign(dataset_id=ds) for ds in DATASETS], ignore_index=True
    )

    uni_norm = RegimeNormalizer(
        n_regimes=6,
        sensor_cols=list(INFORMATIVE_SENSORS),
        op_cols=list(OP_SETTING_COLS),
        use_regime_norm=True,
        random_state=RANDOM_SEED,
    ).fit(tr_uni_raw)
    uni_norm.save(uni_dir / "normalizer.joblib")

    tr_uni_n = uni_norm.transform(tr_uni_raw)
    va_uni_n = uni_norm.transform(va_uni_raw)
    te_uni_n = uni_norm.transform(te_uni_raw)

    tr_uni_f, base_uni_tab_cols = add_tabular_features(
        tr_uni_n, sensor_cols=list(INFORMATIVE_SENSORS), windows=ROLLING_WINDOWS,
        include_regime_onehot=True, n_regimes=6
    )
    va_uni_f, _ = add_tabular_features(
        va_uni_n, sensor_cols=list(INFORMATIVE_SENSORS), windows=ROLLING_WINDOWS,
        include_regime_onehot=True, n_regimes=6
    )
    te_uni_f, _ = add_tabular_features(
        te_uni_n, sensor_cols=list(INFORMATIVE_SENSORS), windows=ROLLING_WINDOWS,
        include_regime_onehot=True, n_regimes=6
    )

    tr_uni_f, ds_onehot_cols = _add_dataset_onehot(tr_uni_f)
    va_uni_f, _ = _add_dataset_onehot(va_uni_f)
    te_uni_f, _ = _add_dataset_onehot(te_uni_f)

    uni_tab_cols = base_uni_tab_cols + ds_onehot_cols
    uni_seq_cols = (
        list(INFORMATIVE_SENSORS)
        + list(OP_SETTING_COLS)
        + [f"regime_{r}" for r in range(6)]
        + ds_onehot_cols
    )

    # Unified XGBoost
    uni_xgb = TabularRULModel(model_type="xgboost", feature_cols=uni_tab_cols, random_state=RANDOM_SEED)
    uni_xgb.fit(tr_uni_f, tr_uni_f["RUL"], va_uni_f, va_uni_f["RUL"], feature_cols=uni_tab_cols)
    uni_xgb.save(uni_dir / "tabular_xgb.joblib")
    uni_xgb_val_rmse = float(
        np.sqrt(np.mean((va_uni_f["RUL"].to_numpy(dtype=np.float64) - uni_xgb.predict(va_uni_f)) ** 2))
    )

    # Unified 1D-CNN & BiLSTM
    X_uni_tr, y_uni_tr, _ = build_sequence_windows(
        tr_uni_f, uni_seq_cols, window_size=WINDOW_SIZE, target_col="RUL", last_only=False
    )
    X_uni_va, y_uni_va, _ = build_sequence_windows(
        va_uni_f, uni_seq_cols, window_size=WINDOW_SIZE, target_col="RUL", last_only=False
    )

    uni_cnn = SequenceRULModel(
        arch="1d_cnn",
        n_features=len(uni_seq_cols),
        seq_len=WINDOW_SIZE,
        epochs=30,
        batch_size=512,
        lr=1.2e-3,
        weight_decay=1e-3,
        patience=7,
        n_mc=30,
        alpha=0.10,
        seed=RANDOM_SEED,
    )
    uni_cnn.fit(
        X_uni_tr,
        y_uni_tr,
        X_uni_va,
        y_uni_va,
        feature_cols=uni_seq_cols,
        checkpoint_path=uni_dir / "sequence_1dcnn.pt",
        verbose=False,
    )

    uni_lstm = SequenceRULModel(
        arch="bilstm",
        n_features=len(uni_seq_cols),
        seq_len=WINDOW_SIZE,
        epochs=30,
        batch_size=512,
        lr=1.2e-3,
        weight_decay=1e-3,
        patience=7,
        n_mc=30,
        alpha=0.10,
        seed=RANDOM_SEED,
    )
    uni_lstm.fit(
        X_uni_tr,
        y_uni_tr,
        X_uni_va,
        y_uni_va,
        feature_cols=uni_seq_cols,
        checkpoint_path=uni_dir / "sequence_bilstm.pt",
        verbose=False,
    )

    # Evaluate Unified models separately on FD001..FD004 test sets
    unified_eval_rows: list[dict[str, Any]] = []
    unified_preds_by_ds: dict[str, dict[str, np.ndarray]] = {}
    for ds in DATASETS:
        y_te = raw_splits[ds]["y_test_raw"]
        te_sub = te_uni_f[te_uni_f["dataset_id"] == ds].copy()
        te_sub_last = _extract_test_last_rows(te_sub)

        p_uxgb, _, lo_uxgb, hi_uxgb = uni_xgb.predict_with_uncertainty(te_sub_last)
        m_uxgb = _eval_both_scales(y_te, p_uxgb, lo_uxgb, hi_uxgb)
        m_uxgb["val_rmse"] = uni_xgb_val_rmse
        cached_models[(ds, "Unified XGBoost")] = {"metrics": m_uxgb, "pred": p_uxgb}

        X_te_sub, _, meta_sub = build_sequence_windows(
            te_sub, uni_seq_cols, window_size=WINDOW_SIZE, target_col="RUL", last_only=True
        )
        expected_units = np.arange(1, len(y_te) + 1, dtype=np.int64)
        if not np.array_equal(meta_sub["unit_id"].to_numpy(dtype=np.int64), expected_units):
            X_te_sub = X_te_sub[np.argsort(meta_sub["unit_id"].to_numpy(dtype=np.int64))]

        p_ucnn, _, lo_ucnn, hi_ucnn = uni_cnn.predict_with_uncertainty(X_te_sub, n_mc=30, alpha=0.10)
        m_ucnn = _eval_both_scales(y_te, p_ucnn, lo_ucnn, hi_ucnn)
        m_ucnn["val_rmse"] = float(uni_cnn.history_["best_val_rmse"])
        cached_models[(ds, "Unified 1D-CNN")] = {"metrics": m_ucnn, "pred": p_ucnn}

        p_ulstm, _, lo_ulstm, hi_ulstm = uni_lstm.predict_with_uncertainty(X_te_sub, n_mc=30, alpha=0.10)
        m_ulstm = _eval_both_scales(y_te, p_ulstm, lo_ulstm, hi_ulstm)
        m_ulstm["val_rmse"] = float(uni_lstm.history_["best_val_rmse"])
        cached_models[(ds, "Unified BiLSTM")] = {"metrics": m_ulstm, "pred": p_ulstm}

        unified_preds_by_ds[ds] = {"xgb": p_uxgb, "cnn": p_ucnn, "bilstm": p_ulstm}

        for model_label, per_ds_key, uni_key in [
            ("XGBoost", "XGBoost", "Unified XGBoost"),
            ("Deep1DCNN", "Deep1DCNN", "Unified 1D-CNN"),
            ("BiLSTM", "BiLSTM", "Unified BiLSTM"),
        ]:
            unified_eval_rows.append(
                {
                    "dataset_id": ds,
                    "architecture": model_label,
                    "per_dataset_rmse_clipped": cached_models[(ds, per_ds_key)]["metrics"]["rmse_clipped"],
                    "unified_rmse_clipped": cached_models[(ds, uni_key)]["metrics"]["rmse_clipped"],
                    "per_dataset_nasa_clipped": cached_models[(ds, per_ds_key)]["metrics"]["nasa_score_clipped"],
                    "unified_nasa_clipped": cached_models[(ds, uni_key)]["metrics"]["nasa_score_clipped"],
                    "per_dataset_rmse_unclipped": cached_models[(ds, per_ds_key)]["metrics"]["rmse_unclipped"],
                    "unified_rmse_unclipped": cached_models[(ds, uni_key)]["metrics"]["rmse_unclipped"],
                }
            )
        _log(
            f"  Unified on {ds} (clipped RMSE): XGB={m_uxgb['rmse_clipped']:.2f} | "
            f"1D-CNN={m_ucnn['rmse_clipped']:.2f} | BiLSTM={m_ulstm['rmse_clipped']:.2f}"
        )

    uni_vs_per_df = pd.DataFrame(unified_eval_rows)
    uni_vs_per_df["rmse_delta_unified_minus_per_ds"] = (
        uni_vs_per_df["unified_rmse_clipped"] - uni_vs_per_df["per_dataset_rmse_clipped"]
    )
    uni_vs_per_df.to_csv(RESULTS_DIR / "ablation_unified_vs_per_dataset.csv", index=False)

    with open(uni_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_json(
                {
                    "tab_feature_cols": uni_tab_cols,
                    "seq_feature_cols": uni_seq_cols,
                    "window_size": WINDOW_SIZE,
                    "n_regimes": 6,
                    "xgb_val_rmse": uni_xgb_val_rmse,
                    "cnn_val_rmse": float(uni_cnn.history_["best_val_rmse"]),
                    "bilstm_val_rmse": float(uni_lstm.history_["best_val_rmse"]),
                }
            ),
            f,
            indent=2,
            allow_nan=False,
        )

    # ------------------------------------------------------------------
    # PHASE 5: BEST MODEL SELECTION, VAL TRAJECTORIES, BENCHMARK TABLES & FIGURES
    # ------------------------------------------------------------------
    _log("\n=== Phase 5: Generating validation trajectories, benchmark tables & figures ===")
    pred_plot_store: dict[str, dict[str, Any]] = {}
    best_per_dataset: dict[str, Any] = {}

    for ds in DATASETS:
        ds_dir = MODELS_DIR / ds
        bundle = bundles_norm[ds]
        y_te = raw_splits[ds]["y_test_raw"]

        # Select best sequence model strictly by validation RMSE (no test leakage)
        cnn_entry = cached_models[(ds, "Deep1DCNN")]
        lstm_entry = cached_models[(ds, "BiLSTM")]
        if cnn_entry["metrics"]["val_rmse"] <= lstm_entry["metrics"]["val_rmse"]:
            best_arch = "Deep1DCNN"
            best_seq_entry = cnn_entry
        else:
            best_arch = "BiLSTM"
            best_seq_entry = lstm_entry

        best_model: SequenceRULModel = best_seq_entry["model"]
        best_model.save(ds_dir / "best_sequence.pt", feature_cols=bundle["seq_cols"])

        # Save validation run-to-failure trajectories for @eval_biz
        X_va_all, y_va_unclipped, meta_va = build_sequence_windows(
            bundle["va_feat"],
            bundle["seq_cols"],
            window_size=WINDOW_SIZE,
            target_col="RUL_raw",
            last_only=False,
        )
        va_pred, va_std, va_lo, va_hi = best_model.predict_with_uncertainty(
            X_va_all, n_mc=30, alpha=0.10, batch_size=1024
        )
        assert np.all(va_lo <= va_pred) and np.all(va_pred <= va_hi), f"CI ordering failed on {ds}"
        val_traj_df = pd.DataFrame(
            {
                "unit_id": meta_va["unit_id"].to_numpy(dtype=np.int64),
                "cycle": meta_va["cycle"].to_numpy(dtype=np.int64),
                "true_rul_unclipped": y_va_unclipped.astype(np.float64),
                "pred_rul": va_pred,
                "ci_low": va_lo,
                "ci_high": va_hi,
                "pred_std": va_std,
            }
        ).sort_values(["unit_id", "cycle"], kind="mergesort").reset_index(drop=True)
        traj_path = RESULTS_DIR / f"val_trajectories_{ds}.csv"
        val_traj_df.to_csv(traj_path, index=False)

        # Save per-dataset test predictions CSV
        test_preds_df = pd.DataFrame(
            {
                "unit_id": np.arange(1, len(y_te) + 1, dtype=np.int64),
                "y_test_raw": y_te,
                "y_test_clipped": clip_rul(y_te, cap=RUL_CLIP),
                "rf_pred": cached_models[(ds, "RandomForest")]["pred"],
                "xgb_pred": cached_models[(ds, "XGBoost")]["pred"],
                "bilstm_pred": lstm_entry["pred"],
                "cnn_pred": cnn_entry["pred"],
                "best_seq_pred": best_seq_entry["pred"],
                "best_seq_ci_low": best_seq_entry["ci_low"],
                "best_seq_ci_high": best_seq_entry["ci_high"],
                "unified_xgb_pred": unified_preds_by_ds[ds]["xgb"],
                "unified_cnn_pred": unified_preds_by_ds[ds]["cnn"],
                "unified_bilstm_pred": unified_preds_by_ds[ds]["bilstm"],
            }
        )
        test_preds_df.to_csv(ds_dir / "test_predictions.csv", index=False)

        # Save dataset metadata.json for Streamlit & reproducibility
        meta_json = {
            "dataset_id": ds,
            "n_regimes": int(DATASET_META[ds]["op_conditions"]),
            "fault_modes": int(DATASET_META[ds]["fault_modes"]),
            "window_size": WINDOW_SIZE,
            "best_sequence_arch": best_arch,
            "best_sequence_file": "best_sequence.pt",
            "seq_feature_cols": bundle["seq_cols"],
            "tab_feature_cols": bundle["tab_cols"],
            "use_fault_clusterer_in_best": False,
            "has_fault_clusterer": (ds_dir / "fault_clusterer.joblib").exists(),
            "conformal_calibration": best_model.history_.get("calibration", {}),
            "best_sequence_metrics": best_seq_entry["metrics"],
        }
        with open(ds_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(_sanitize_json(meta_json), f, indent=2, allow_nan=False)

        pred_plot_store[ds] = {
            "y_test_raw": y_te,
            "pred": best_seq_entry["pred"],
            "ci_low": best_seq_entry["ci_low"],
            "ci_high": best_seq_entry["ci_high"],
            "best_arch": best_arch,
            **best_seq_entry["metrics"],
        }
        best_per_dataset[ds] = {
            "best_arch": best_arch,
            "val_traj_path": str(traj_path),
            "n_val_engines": int(val_traj_df["unit_id"].nunique()),
            "n_val_cycles": int(len(val_traj_df)),
            **best_seq_entry["metrics"],
        }
        _log(
            f"  {ds}: Selected {best_arch} (val_rmse={best_seq_entry['metrics']['val_rmse']:.2f}, "
            f"test_rmse={best_seq_entry['metrics']['rmse_clipped']:.2f}, "
            f"nasa={best_seq_entry['metrics']['nasa_score_clipped']:.1f}) -> {traj_path.name}"
        )

    # Assemble full benchmark_comparison.csv
    bench_rows: list[dict[str, Any]] = []
    model_catalog = [
        ("RandomForest", "Tabular", "per_dataset"),
        ("XGBoost", "Tabular", "per_dataset"),
        ("BiLSTM", "Sequence", "per_dataset"),
        ("Deep1DCNN", "Sequence", "per_dataset"),
        ("Unified XGBoost", "Tabular", "unified"),
        ("Unified 1D-CNN", "Sequence", "unified"),
        ("Unified BiLSTM", "Sequence", "unified"),
    ]
    for ds in DATASETS:
        for m_name, m_fam, scope in model_catalog:
            m = cached_models[(ds, m_name)]["metrics"]
            bench_rows.append(
                {
                    "dataset_id": ds,
                    "model": m_name,
                    "model_family": m_fam,
                    "scope": scope,
                    **m,
                }
            )

    bench_df = pd.DataFrame(bench_rows)
    bench_df = compare_against_benchmarks(bench_df)
    bench_df.to_csv(RESULTS_DIR / "benchmark_comparison.csv", index=False)

    # Generate figures
    _plot_benchmark_comparison(bench_df, FIGURES_DIR / "model_benchmark_comparison.png")
    _plot_rul_predictions(pred_plot_store, FIGURES_DIR / "model_rul_predictions_FD001_FD004.png")

    elapsed = time.perf_counter() - t_start
    summary_payload = {
        "config": {
            "seed": RANDOM_SEED,
            "rul_clip": RUL_CLIP,
            "window_size": WINDOW_SIZE,
            "rolling_windows": list(ROLLING_WINDOWS),
            "datasets": list(DATASETS),
        },
        "elapsed_seconds": round(elapsed, 2),
        "best_per_dataset_models": best_per_dataset,
        "ablation_regime_norm": ablation_regime_df.to_dict(orient="records"),
        "ablation_fault_modes": ablation_fault_df.to_dict(orient="records"),
        "ablation_unified_vs_per_dataset": uni_vs_per_df.to_dict(orient="records"),
        "benchmark_comparison": bench_df.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary_payload), f, indent=2, allow_nan=False)

    _log(f"\n=== ALL EXPERIMENTS COMPLETE in {elapsed:.1f}s ({elapsed/60.0:.2f} min) ===")


if __name__ == "__main__":
    main()
