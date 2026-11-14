"""Run Phase 6 fleet maintenance policy simulation on validation trajectories.

Reads saved validation run-to-failure trajectory predictions from
``reports/results/val_trajectories_{FD00X}.{parquet,csv,npz}`` (or falls back
to synthetic demo trajectories only when ``--demo`` is explicitly passed),
compares predictive RUL-threshold policies against fixed-interval baselines,
and saves ``reports/results/maintenance_simulation.json`` and
``reports/figures/maintenance_cost_comparison.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.maintenance import (
    CostAssumptions,
    compare_maintenance_policies,
    generate_demo_trajectories,
    load_trajectories_from_file,
)

DATASETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_THRESHOLDS: tuple[int, ...] = (10, 15, 20, 25, 30, 35)
DEFAULT_FIXED_INTERVALS: tuple[int, ...] = (100, 120, 140, 150, 175, 200)


def discover_validation_trajectories(
    results_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load saved validation trajectories per dataset from ``results_dir``."""
    per_dataset: dict[str, list[dict[str, Any]]] = {}
    for ds in DATASETS:
        for ext in (".parquet", ".csv", ".npz"):
            candidate = results_dir / f"val_trajectories_{ds}{ext}"
            if candidate.exists() and candidate.stat().st_size > 0:
                per_dataset[ds] = load_trajectories_from_file(candidate, dataset_id=ds)
                break
    return per_dataset


def plot_maintenance_comparison(summary: dict[str, Any], output_path: Path) -> None:
    """Create a 2-panel maintenance cost and risk/waste comparison figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_sweep = summary["fixed_interval_sweep"]
    pred_sweep = summary["predictive_sweep"]
    pred_def = summary["predictive_default"]
    fixed_def = summary["fixed_default"]
    best_fixed = summary["best_fixed_interval"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.0), dpi=180)
    fig.patch.set_facecolor("white")

    # Panel 1: Annual 300-Engine Fleet Cost ($M)
    fx_labels = [f"Fixed\n{r['interval_cycles']}" for r in fixed_sweep]
    fx_costs = [r["annual_cost_300_fleet"] / 1e6 for r in fixed_sweep]
    pr_labels = [f"Pred\n<={r['threshold_cycles']}" for r in pred_sweep]
    pr_costs = [r["annual_cost_300_fleet"] / 1e6 for r in pred_sweep]

    all_labels = fx_labels + pr_labels
    all_costs = fx_costs + pr_costs
    x = np.arange(len(all_labels))
    colors = ["#7B8FA1" if r["interval_cycles"] != fixed_def["interval_cycles"] else "#4B6587" for r in fixed_sweep]
    colors += [
        "#1E8449" if r["threshold_cycles"] == pred_def["threshold_cycles"] else "#52BE80"
        for r in pred_sweep
    ]

    bars = ax1.bar(x, all_costs, color=colors, edgecolor="#1F2933", linewidth=0.6, width=0.68)
    ax1.axvline(len(fx_labels) - 0.5, color="#334E68", linestyle="--", linewidth=1.1, alpha=0.7)
    ax1.axhline(
        fixed_def["annual_cost_300_fleet"] / 1e6,
        color="#4B6587",
        linestyle=":",
        linewidth=1.2,
        label=f"Fixed {fixed_def['interval_cycles']} baseline (${fixed_def['annual_cost_300_fleet']/1e6:.1f}M)",
    )
    ax1.axhline(
        best_fixed["annual_cost_300_fleet"] / 1e6,
        color="#D97706",
        linestyle="-.",
        linewidth=1.2,
        label=f"Best fixed ({best_fixed['interval_cycles']} cyc: ${best_fixed['annual_cost_300_fleet']/1e6:.1f}M)",
    )

    for bar, val in zip(bars, all_costs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max(all_costs) * 0.02,
            f"${val:.1f}M",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="semibold",
            color="#102A43",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )

    savings_m = pred_def["annual_savings_300_fleet"] / 1e6
    savings_pct = pred_def["savings_pct"]
    best_sav_m = summary["predictive_vs_best_fixed"]["annual_savings_300_fleet"] / 1e6
    best_sav_pct = summary["predictive_vs_best_fixed"]["savings_pct"]

    ax1.set_title(
        f"300-Engine Fleet Annual Maintenance Cost\n"
        f"Pred (RUL<=20) saves ${savings_m:.1f}M/yr ({savings_pct:.1f}%) vs Fixed 150 | "
        f"${best_sav_m:.1f}M/yr ({best_sav_pct:.1f}%) vs Best Fixed",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylabel("Annual Fleet Cost ($ Millions)", fontsize=10)
    ax1.set_xlabel("Policy: Fixed Interval (cycles) | Predictive RUL Threshold (cycles)", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, fontsize=8.5)
    ax1.set_ylim(0, max(all_costs) * 1.18)
    ax1.grid(axis="y", alpha=0.25, linestyle="-")
    ax1.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel 2: Unplanned Failure Rate vs Early Pull Rate (%)
    fx_fail_pct = [100.0 * r["unplanned_failure_rate"] for r in fixed_sweep]
    fx_early_pct = [100.0 * r["early_pull_rate"] for r in fixed_sweep]
    pr_fail_pct = [100.0 * r["unplanned_failure_rate"] for r in pred_sweep]
    pr_early_pct = [100.0 * r["early_pull_rate"] for r in pred_sweep]

    w = 0.36
    x2 = np.arange(len(all_labels))
    fail_vals = fx_fail_pct + pr_fail_pct
    early_vals = fx_early_pct + pr_early_pct

    ax2.bar(
        x2 - w / 2,
        fail_vals,
        width=w,
        color="#C0392B",
        alpha=0.88,
        label="Unplanned Failures (% engines)",
    )
    ax2.bar(
        x2 + w / 2,
        early_vals,
        width=w,
        color="#2E86C1",
        alpha=0.85,
        label="Unnecessarily Early Pulls (% engines)",
    )
    ax2.axvline(len(fx_labels) - 0.5, color="#334E68", linestyle="--", linewidth=1.1, alpha=0.7)

    ax2.set_title(
        "Safety vs Asset Utilization Trade-off\n"
        f"Failures avoided vs Fixed 150: {pred_def['unplanned_failures_avoided_vs_fixed']} "
        f"(Fixed: {fixed_def['unplanned_failures']} -> Pred: {pred_def['unplanned_failures']})",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylabel("Percentage of Engine Lives (%)", fontsize=10)
    ax2.set_xlabel("Policy: Fixed Interval (cycles) | Predictive RUL Threshold (cycles)", fontsize=9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(all_labels, fontsize=8.5)
    ax2.set_ylim(0, max(max(fail_vals), max(early_vals), 10.0) * 1.18)
    ax2.grid(axis="y", alpha=0.25, linestyle="-")
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_simulation(
    results_dir: Path,
    output_json: Path,
    output_fig: Path,
    demo: bool = False,
    costs: CostAssumptions | None = None,
) -> dict[str, Any]:
    """Execute fleet maintenance simulation and persist JSON + PNG artifacts."""
    cfg = costs if costs is not None else CostAssumptions()

    if demo:
        demo_trajs = generate_demo_trajectories(n_engines=120, seed=42)
        per_dataset_trajs = {"DEMO": demo_trajs}
        combined_trajs = demo_trajs
        source = "demo_synthetic"
    else:
        per_dataset_trajs = discover_validation_trajectories(results_dir)
        if not per_dataset_trajs:
            raise FileNotFoundError(
                f"No validation trajectory files (val_trajectories_FD00X.{{parquet,csv,npz}}) "
                f"found in {results_dir}. Run scripts/run_experiments.py first, or pass --demo "
                f"for synthetic unit testing."
            )
        combined_trajs = [t for ds_list in per_dataset_trajs.values() for t in ds_list]
        source = "validation_trajectories"

    overall = compare_maintenance_policies(
        combined_trajs,
        thresholds=DEFAULT_THRESHOLDS,
        fixed_intervals=DEFAULT_FIXED_INTERVALS,
        default_threshold=20,
        default_fixed_interval=150,
        costs=cfg,
    )

    per_dataset_results: dict[str, Any] = {}
    for ds_name, ds_trajs in per_dataset_trajs.items():
        per_dataset_results[ds_name] = compare_maintenance_policies(
            ds_trajs,
            thresholds=DEFAULT_THRESHOLDS,
            fixed_intervals=DEFAULT_FIXED_INTERVALS,
            default_threshold=20,
            default_fixed_interval=150,
            costs=cfg,
        )

    payload: dict[str, Any] = {
        "source": source,
        "datasets_included": list(per_dataset_trajs.keys()),
        "n_engines_total": len(combined_trajs),
        **overall,
        "per_dataset": per_dataset_results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_maintenance_comparison(payload, output_fig)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "results",
        help="Directory containing val_trajectories_FD00X.{parquet,csv,npz}.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "results" / "maintenance_simulation.json",
        help="Output JSON path for simulation results.",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "maintenance_cost_comparison.png",
        help="Output PNG path for maintenance comparison chart.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic demo trajectories (for testing before real trajectories exist).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_simulation(
        results_dir=args.results_dir,
        output_json=args.output_json,
        output_fig=args.output_fig,
        demo=args.demo,
    )
    pred = summary["predictive_default"]
    fixed = summary["fixed_default"]
    best_fx = summary["best_fixed_interval"]
    vs_best = summary["predictive_vs_best_fixed"]
    print(f"[maintenance_sim] source={summary['source']} engines={summary['n_engines_total']}")
    print(
        f"  Fixed {fixed['interval_cycles']}: fails={fixed['unplanned_failures']}, "
        f"early={fixed['engines_pulled_early']}, "
        f"annual=${fixed['annual_cost_300_fleet']/1e6:.2f}M"
    )
    print(
        f"  Best Fixed ({best_fx['interval_cycles']}): fails={best_fx['unplanned_failures']}, "
        f"early={best_fx['engines_pulled_early']}, "
        f"annual=${best_fx['annual_cost_300_fleet']/1e6:.2f}M"
    )
    print(
        f"  Predictive (thr={pred['threshold_cycles']}): fails={pred['unplanned_failures']}, "
        f"early={pred['engines_pulled_early']}, "
        f"annual=${pred['annual_cost_300_fleet']/1e6:.2f}M"
    )
    print(
        f"  Savings vs Fixed 150: ${pred['annual_savings_300_fleet']/1e6:.2f}M/yr "
        f"({pred['savings_pct']:.1f}%), avoided {pred['unplanned_failures_avoided_vs_fixed']} failures"
    )
    print(
        f"  Savings vs Best Fixed ({best_fx['interval_cycles']}): "
        f"${vs_best['annual_savings_300_fleet']/1e6:.2f}M/yr ({vs_best['savings_pct']:.1f}%)"
    )


if __name__ == "__main__":
    main()
