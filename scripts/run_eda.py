"""Phase 1 Exploratory Data Analysis across all four NASA C-MAPSS sub-datasets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MPL_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.config import (  # noqa: E402
    DATASET_META,
    DATASETS,
    DROP_SENSORS_CANONICAL,
    FIGURES_DIR,
    INFORMATIVE_SENSORS,
    RESULTS_DIR,
    ROLLING_WINDOWS,
    RUL_CLIP,
    WINDOW_SIZE,
    ensure_project_dirs,
)
from src.data import (  # noqa: E402
    RegimeNormalizer,
    analyze_sensor_variance,
    load_cmapss_dataset,
)
from src.utils import (  # noqa: E402
    plot_operating_regimes_and_norm,
    plot_sensor_degradation,
    plot_sensor_variance_comparison,
)


def run_eda() -> dict:
    """Execute Phase 1 EDA across FD001-FD004 and save figures and JSON summary."""
    ensure_project_dirs()

    summary: dict = {
        "datasets": {},
        "drop_sensors_canonical": list(DROP_SENSORS_CANONICAL),
        "informative_sensors": list(INFORMATIVE_SENSORS),
        "rul_clip": RUL_CLIP,
        "window_size": WINDOW_SIZE,
        "rolling_windows": list(ROLLING_WINDOWS),
        "figures_generated": [],
    }

    variance_by_dataset = {}
    multi_regime_raw_and_norm = {}

    for ds_id in DATASETS:
        meta = DATASET_META[ds_id]
        n_regimes = int(meta["op_conditions"])

        train_df, test_df, y_test_raw = load_cmapss_dataset(ds_id, rul_clip=RUL_CLIP)
        var_df = analyze_sensor_variance(train_df, n_regimes=n_regimes)
        variance_by_dataset[ds_id] = var_df

        normalizer = RegimeNormalizer(
            n_regimes=n_regimes,
            sensor_cols=INFORMATIVE_SENSORS,
            use_regime_norm=(n_regimes > 1),
        )
        train_norm = normalizer.fit_transform(train_df)

        # For FD002/FD004 plot regime-normalized sensor trajectories so degradation is clear
        fig_sensor_path = plot_sensor_degradation(
            train_norm if n_regimes > 1 else train_df,
            dataset_id=ds_id,
            normalized=(n_regimes > 1),
            save_path=FIGURES_DIR / f"eda_sensors_{ds_id}.png",
        )
        summary["figures_generated"].append(fig_sensor_path.relative_to(PROJECT_ROOT).as_posix())

        if ds_id in ("FD002", "FD004"):
            multi_regime_raw_and_norm[ds_id] = (train_df, train_norm)

        train_lengths = train_df.groupby("unit_id")["cycle"].max()
        test_lengths = test_df.groupby("unit_id")["cycle"].max()

        emp_const = sorted(var_df.loc[var_df["is_near_constant"], "sensor"].tolist())
        emp_informative = [
            s for s in var_df.loc[~var_df["is_near_constant"], "sensor"].tolist()
        ]

        sensor_stats = {}
        for _, row in var_df.iterrows():
            sensor_stats[str(row["sensor"])] = {
                "global_mean": round(float(row["global_mean"]), 6),
                "global_std": round(float(row["global_std"]), 6),
                "within_regime_std": round(float(row["within_regime_std"]), 6),
                "within_regime_iqr": round(float(row["within_regime_iqr"]), 6),
                "corr_with_rul": round(float(row["corr_with_rul"]), 4),
                "is_canonical_drop": bool(row["is_canonical_drop"]),
                "is_near_constant": bool(row["is_near_constant"]),
            }

        summary["datasets"][ds_id] = {
            "op_conditions": n_regimes,
            "fault_modes": int(meta["fault_modes"]),
            "description": str(meta["description"]),
            "n_train_units": int(train_df["unit_id"].nunique()),
            "n_test_units": int(test_df["unit_id"].nunique()),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "train_cycle_min": int(train_lengths.min()),
            "train_cycle_max": int(train_lengths.max()),
            "train_cycle_mean": round(float(train_lengths.mean()), 2),
            "train_cycle_median": round(float(train_lengths.median()), 2),
            "test_cycle_min": int(test_lengths.min()),
            "test_cycle_max": int(test_lengths.max()),
            "test_cycle_mean": round(float(test_lengths.mean()), 2),
            "test_rul_min": int(y_test_raw.min()),
            "test_rul_max": int(y_test_raw.max()),
            "test_rul_mean": round(float(y_test_raw.mean()), 2),
            "constant_sensors_empirical": emp_const,
            "informative_sensors_empirical": emp_informative,
            "matches_canonical_drop": emp_const == sorted(DROP_SENSORS_CANONICAL),
            "sensor_stats": sensor_stats,
        }

    var_fig_path = plot_sensor_variance_comparison(
        variance_by_dataset,
        save_path=FIGURES_DIR / "eda_sensor_variance.png",
    )
    summary["figures_generated"].append(var_fig_path.relative_to(PROJECT_ROOT).as_posix())

    reg_fig_path = plot_operating_regimes_and_norm(
        multi_regime_raw_and_norm,
        sensor_col="sensor_11",
        save_path=FIGURES_DIR / "eda_regimes_FD002_FD004.png",
    )
    summary["figures_generated"].append(reg_fig_path.relative_to(PROJECT_ROOT).as_posix())

    out_json = RESULTS_DIR / "eda_sensor_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_eda()
    print("=== Phase 1 EDA Complete ===")
    for ds_id, ds_info in summary["datasets"].items():
        print(
            f"{ds_id}: train_units={ds_info['n_train_units']} "
            f"(rows={ds_info['n_train_rows']}, cycles={ds_info['train_cycle_min']}-{ds_info['train_cycle_max']}), "
            f"test_units={ds_info['n_test_units']} "
            f"(min_len={ds_info['test_cycle_min']}, RUL={ds_info['test_rul_min']}-{ds_info['test_rul_max']}), "
            f"matches_canonical_drop={ds_info['matches_canonical_drop']}"
        )
    print(f"Summary JSON: {RESULTS_DIR / 'eda_sensor_summary.json'}")
    print(f"Figures ({len(summary['figures_generated'])}): {summary['figures_generated']}")


if __name__ == "__main__":
    main()
