"""Matplotlib visualization helpers for Phase 1 C-MAPSS EDA."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import DROP_SENSORS_CANONICAL, FIGURES_DIR, PROJECT_ROOT

_MPL_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_EDA_SENSORS: list[str] = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_11",
    "sensor_12",
    "sensor_15",
]

_SENSOR_LABELS: dict[str, str] = {
    "sensor_2": "Sensor 2 (LPC Outlet Temp T24)",
    "sensor_3": "Sensor 3 (HPC Outlet Temp T30)",
    "sensor_4": "Sensor 4 (LPT Outlet Temp T50)",
    "sensor_11": "Sensor 11 (HPC Static Press Ps30)",
    "sensor_12": "Sensor 12 (Fuel Flow Ratio phi)",
    "sensor_15": "Sensor 15 (Bypass Ratio BPR)",
}


def _select_representative_units(df: pd.DataFrame, n_units: int = 4) -> list[int]:
    lengths = df.groupby("unit_id")["cycle"].max().sort_values()
    if len(lengths) <= n_units:
        return [int(u) for u in lengths.index.tolist()]
    quantiles = np.linspace(0.1, 0.9, n_units)
    indices = [int(round(q * (len(lengths) - 1))) for q in quantiles]
    # Deduplicate while preserving order
    seen: set[int] = set()
    chosen: list[int] = []
    for idx in indices:
        uid = int(lengths.index[idx])
        if uid not in seen:
            seen.add(uid)
            chosen.append(uid)
    return chosen


def plot_sensor_degradation(
    train_df: pd.DataFrame,
    dataset_id: str,
    sensors: list[str] | None = None,
    unit_ids: list[int] | None = None,
    normalized: bool = False,
    save_path: Path | None = None,
) -> Path:
    """Plot sensor readings vs cycle across 4 representative engines for a dataset."""
    plot_sensors = list(sensors) if sensors is not None else list(DEFAULT_EDA_SENSORS)
    chosen_units = (
        [int(u) for u in unit_ids]
        if unit_ids is not None
        else _select_representative_units(train_df, n_units=4)
    )

    dest = Path(save_path) if save_path is not None else FIGURES_DIR / f"eda_sensors_{dataset_id}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)

    n_cols = 3
    n_rows = int(np.ceil(len(plot_sensors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), sharex=False)
    axes_flat = np.atleast_1d(axes).ravel()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax_idx, sensor in enumerate(plot_sensors):
        ax = axes_flat[ax_idx]
        for u_idx, uid in enumerate(chosen_units):
            sub = train_df[train_df["unit_id"] == uid].sort_values("cycle")
            color = palette[u_idx % len(palette)]
            cycles = sub["cycle"].to_numpy()
            vals = sub[sensor].to_numpy(dtype=np.float64)
            smoothed = (
                pd.Series(vals)
                .rolling(window=7, min_periods=1)
                .mean()
                .to_numpy()
            )
            ax.plot(cycles, vals, color=color, alpha=0.22, linewidth=0.9)
            ax.plot(
                cycles,
                smoothed,
                color=color,
                linewidth=1.8,
                label=f"Unit {uid} (max={int(cycles[-1])})" if ax_idx == 0 else None,
            )
        ax.set_title(_SENSOR_LABELS.get(sensor, sensor), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=9)
        ax.set_ylabel("Z-score (within regime)" if normalized else "Raw Reading", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    for unused in axes_flat[len(plot_sensors) :]:
        unused.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=len(chosen_units),
            frameon=True,
            fontsize=9.5,
        )

    mode_note = "Within-Regime Normalized Sensors" if normalized else "Raw Sensor Trajectories"
    fig.suptitle(
        f"{dataset_id} — {mode_note} vs. Cycle (4 Engines, 7-Cycle Moving Avg)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_sensor_variance_comparison(
    variance_by_dataset: dict[str, pd.DataFrame],
    save_path: Path | None = None,
) -> Path:
    """Plot global vs within-regime sensor std across FD001-FD004 highlighting constant sensors."""
    dest = Path(save_path) if save_path is not None else FIGURES_DIR / "eda_sensor_variance.png"
    dest.parent.mkdir(parents=True, exist_ok=True)

    datasets = list(variance_by_dataset.keys())
    n_ds = len(datasets)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes_flat = axes.ravel()

    floor = 1e-4
    for idx, ds_id in enumerate(datasets):
        ax = axes_flat[idx]
        df_var = variance_by_dataset[ds_id].copy()
        y_pos = np.arange(len(df_var))
        sensors = df_var["sensor"].tolist()

        g_vals = np.maximum(df_var["global_std"].to_numpy(dtype=np.float64), floor)
        w_vals = np.maximum(df_var["within_regime_std"].to_numpy(dtype=np.float64), floor)
        is_drop = df_var["is_canonical_drop"].to_numpy(dtype=bool)

        bar_h = 0.38
        ax.barh(
            y_pos - bar_h / 2,
            g_vals,
            height=bar_h,
            color="#94a3b8",
            alpha=0.65,
            label="Global Std",
        )
        w_colors = ["#dc2626" if d else "#0f766e" for d in is_drop]
        ax.barh(
            y_pos + bar_h / 2,
            w_vals,
            height=bar_h,
            color=w_colors,
            alpha=0.92,
            label="Within-Regime Std (Informative)" if idx == 0 else None,
        )

        ax.axvline(0.02, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlim(floor * 0.8, max(float(g_vals.max()), 1.0) * 2.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sensors, fontsize=8.5)
        for tick_label, dropped in zip(ax.get_yticklabels(), is_drop):
            if dropped:
                tick_label.set_color("#b91c1c")
                tick_label.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlabel("Standard Deviation (log scale; floor=1e-4)", fontsize=9)
        ax.set_title(
            f"{ds_id} — Global vs Within-Regime Std (red = canonical drop)",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    for unused in axes_flat[n_ds:]:
        unused.set_visible(False)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_items = [
        Patch(facecolor="#94a3b8", alpha=0.65, label="Global Std (confounded on FD002/FD004)"),
        Patch(facecolor="#0f766e", alpha=0.92, label="Within-Regime Std: 14 Informative Sensors"),
        Patch(facecolor="#dc2626", alpha=0.92, label="Within-Regime Std: 7 Constant/Dropped Sensors"),
        Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=1.2, label="Constant Cutoff (0.02)"),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=True,
        fontsize=9.5,
    )
    fig.suptitle(
        "C-MAPSS Sensor Variability: Global vs. Within-Regime Standard Deviation",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_operating_regimes_and_norm(
    datasets_raw_and_norm: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    sensor_col: str = "sensor_11",
    save_path: Path | None = None,
) -> Path:
    """Plot 6 KMeans operating regimes and before/after sensor normalization for FD002 & FD004."""
    dest = (
        Path(save_path)
        if save_path is not None
        else FIGURES_DIR / "eda_regimes_FD002_FD004.png"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)

    ds_names = list(datasets_raw_and_norm.keys())
    n_rows = len(ds_names)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16.5, 4.8 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    regime_colors = plt.get_cmap("tab10")

    for row_idx, ds_id in enumerate(ds_names):
        raw_df, norm_df = datasets_raw_and_norm[ds_id]
        regimes = norm_df["regime"].to_numpy(dtype=np.int64)
        unique_reg = sorted(np.unique(regimes).tolist())

        # Col 0: Operating condition clusters (Altitude vs Mach)
        ax0 = axes[row_idx, 0]
        for r in unique_reg:
            m = regimes == r
            op1 = raw_df.loc[m, "op_setting_1"].to_numpy()
            op2 = raw_df.loc[m, "op_setting_2"].to_numpy()
            tra = float(raw_df.loc[m, "op_setting_3"].mean())
            color = regime_colors(r % 10)
            ax0.scatter(
                op1,
                op2,
                s=12,
                alpha=0.55,
                color=color,
                label=f"R{r} (TRA={tra:.0f}, n={int(m.sum())})",
            )
            ax0.annotate(
                f"R{r}",
                xy=(float(np.mean(op1)), float(np.mean(op2))),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
            )
        ax0.set_title(f"{ds_id}: 6 Operating Regimes (KMeans)", fontsize=11, fontweight="bold")
        ax0.set_xlabel("op_setting_1 (Altitude, kft)", fontsize=9)
        ax0.set_ylabel("op_setting_2 (Mach)", fontsize=9)
        ax0.legend(fontsize=7.5, loc="upper left", frameon=True)
        ax0.grid(True, linestyle="--", alpha=0.3)

        # Col 1: Before normalization (Raw sensor distribution by regime)
        ax1 = axes[row_idx, 1]
        raw_vals = raw_df[sensor_col].to_numpy(dtype=np.float64)
        bins_raw = np.linspace(float(raw_vals.min()), float(raw_vals.max()), 60)
        for r in unique_reg:
            m = regimes == r
            ax1.hist(
                raw_vals[m],
                bins=bins_raw,
                density=True,
                alpha=0.45,
                color=regime_colors(r % 10),
                label=f"R{r}",
            )
        ax1.set_title(
            f"{ds_id}: BEFORE Norm ({sensor_col} Raw)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel(f"Raw {sensor_col}", fontsize=9)
        ax1.set_ylabel("Density", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Col 2: After within-regime Z-score normalization
        ax2 = axes[row_idx, 2]
        norm_vals = norm_df[sensor_col].to_numpy(dtype=np.float64)
        bins_norm = np.linspace(-4.0, 4.0, 60)
        for r in unique_reg:
            m = regimes == r
            ax2.hist(
                norm_vals[m],
                bins=bins_norm,
                density=True,
                alpha=0.42,
                color=regime_colors(r % 10),
                label=f"R{r}",
            )
        ax2.axvline(0.0, color="#111827", linestyle="--", linewidth=1.1, alpha=0.7)
        ax2.set_title(
            f"{ds_id}: AFTER Within-Regime Z-Score ({sensor_col})",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel(f"Normalized {sensor_col} (Z-score)", fontsize=9)
        ax2.set_ylabel("Density", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        "Multi-Operating-Condition Handling: KMeans Regime Clustering & Within-Regime Z-Score",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest
