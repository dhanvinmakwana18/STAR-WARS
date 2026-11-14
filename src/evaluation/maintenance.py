"""Fleet maintenance policy simulator and dollar-impact quantification.

Simulates run-to-failure engine trajectories under both predictive RUL-threshold
maintenance and fixed-interval overhaul baselines, scaling outcomes to a
300-engine commercial turbofan fleet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    """Financial and operational assumptions for fleet maintenance simulation."""

    scheduled_maintenance_cost: float = 120_000.0
    unscheduled_removal_cost: float = 750_000.0
    aog_cost_per_day: float = 150_000.0
    aog_days_per_failure: float = 2.0
    early_pull_cost_per_cycle: float = 1_200.0
    # Cycles within [threshold - grace, threshold + grace] are well-timed;
    # true_rul > threshold + grace is counted as an unnecessarily early pull.
    early_pull_grace_cycles: int = 10
    fleet_size: int = 300
    cycles_per_engine_year: float = 800.0

    @property
    def unplanned_failure_cost(self) -> float:
        """Total direct cost per unplanned in-service failure (removal + AOG)."""
        return float(
            self.unscheduled_removal_cost + self.aog_cost_per_day * self.aog_days_per_failure
        )


def _normalize_trajectory(traj: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and sort one engine trajectory chronologically by cycle."""
    if "unit_id" not in traj:
        raise KeyError("Trajectory dict must contain 'unit_id'.")

    cycles_raw = traj.get("cycles", traj.get("cycle"))
    true_raw = traj.get("true_rul", traj.get("true_rul_unclipped"))
    pred_raw = traj.get("pred_rul")
    if cycles_raw is None or true_raw is None or pred_raw is None:
        raise KeyError(
            "Trajectory dict must contain 'cycles' (or 'cycle'), "
            "'true_rul' (or 'true_rul_unclipped'), and 'pred_rul'."
        )

    cycles = np.asarray(cycles_raw, dtype=np.float64).ravel()
    true_rul = np.asarray(true_raw, dtype=np.float64).ravel()
    pred_rul = np.asarray(pred_raw, dtype=np.float64).ravel()

    if cycles.size == 0:
        raise ValueError(f"Trajectory for unit_id={traj['unit_id']} is empty.")
    if not (cycles.shape == true_rul.shape == pred_rul.shape):
        raise ValueError(
            f"Trajectory arrays for unit_id={traj['unit_id']} must have equal length."
        )

    order = np.argsort(cycles, kind="mergesort")
    return traj["unit_id"], cycles[order], true_rul[order], pred_rul[order]


def _scale_to_annual_fleet(
    cost_per_engine_life: float,
    mean_true_lifespan: float,
    costs: CostAssumptions,
) -> float:
    """Scale per-engine-episode cost to annual cost for a fleet of ``fleet_size``.

    Because ``cost_per_engine_life`` already monetizes wasted remaining life via
    ``early_pull_cost_per_cycle``, annual episode frequency is scaled by the
    fleet's natural engine lifespan (``cycles_per_engine_year / mean_true_lifespan``)
    so wasted cycles are not double-counted.
    """
    if mean_true_lifespan <= 0.0:
        raise ValueError("mean_true_lifespan must be positive.")
    annual_lives_per_engine = costs.cycles_per_engine_year / mean_true_lifespan
    return float(cost_per_engine_life * costs.fleet_size * annual_lives_per_engine)


def simulate_fixed_interval_policy(
    trajectories: Sequence[dict[str, Any]],
    interval_cycles: int = 150,
    costs: CostAssumptions | None = None,
    reference_threshold: int = 20,
) -> dict[str, Any]:
    """Simulate a fixed-interval scheduled overhaul policy across run-to-failure trajectories.

    An engine is overhauled at ``cycle == interval_cycles`` if it survives at least
    that long (``max_cycle >= interval_cycles``), incurring ``scheduled_maintenance_cost``
    plus ``wasted_cycles = max(0, true_rul_at_pull - reference_threshold)`` valued at
    ``early_pull_cost_per_cycle``. If the engine fails earlier (``max_cycle < interval_cycles``),
    it incurs an unplanned failure (removal + AOG).
    """
    if not trajectories:
        raise ValueError("trajectories must not be empty.")
    if interval_cycles <= 0:
        raise ValueError("interval_cycles must be positive.")
    cfg = costs if costs is not None else CostAssumptions()

    n_engines = len(trajectories)
    scheduled_pulls = 0
    unplanned_failures = 0
    engines_pulled_early = 0
    total_wasted_cycles = 0.0
    total_utilized_cycles = 0.0
    total_true_lifespan = 0.0

    for traj in trajectories:
        _, cycles, true_rul, _ = _normalize_trajectory(traj)
        max_cycle = float(cycles[-1] + max(0.0, true_rul[-1]))
        total_true_lifespan += max_cycle

        if max_cycle < float(interval_cycles):
            unplanned_failures += 1
            total_utilized_cycles += max_cycle
        else:
            scheduled_pulls += 1
            total_utilized_cycles += float(interval_cycles)
            idx = int(np.searchsorted(cycles, float(interval_cycles), side="left"))
            if idx < cycles.size and cycles[idx] == float(interval_cycles):
                true_rul_at_pull = max(0.0, float(true_rul[idx]))
            else:
                true_rul_at_pull = max(0.0, max_cycle - float(interval_cycles))

            wasted = max(0.0, true_rul_at_pull - float(reference_threshold))
            total_wasted_cycles += wasted
            if true_rul_at_pull > float(reference_threshold + cfg.early_pull_grace_cycles):
                engines_pulled_early += 1

    scheduled_cost = scheduled_pulls * cfg.scheduled_maintenance_cost
    failure_cost = unplanned_failures * cfg.unplanned_failure_cost
    early_pull_cost = total_wasted_cycles * cfg.early_pull_cost_per_cycle
    total_cost = float(scheduled_cost + failure_cost + early_pull_cost)

    cost_per_engine_life = float(total_cost / n_engines)
    mean_true_lifespan = float(total_true_lifespan / n_engines)
    annual_cost_300_fleet = _scale_to_annual_fleet(
        cost_per_engine_life, mean_true_lifespan, cfg
    )

    return {
        "policy": f"fixed_interval_{interval_cycles}",
        "interval_cycles": int(interval_cycles),
        "reference_threshold": int(reference_threshold),
        "n_engines": int(n_engines),
        "scheduled_pulls": int(scheduled_pulls),
        "unplanned_failures": int(unplanned_failures),
        "unplanned_failure_rate": float(unplanned_failures / n_engines),
        "unplanned_failures_avoided_vs_fixed": 0,
        "engines_pulled_early": int(engines_pulled_early),
        "early_pull_rate": float(engines_pulled_early / n_engines),
        "total_wasted_cycles": float(total_wasted_cycles),
        "avg_wasted_cycles": float(total_wasted_cycles / n_engines),
        "avg_utilized_cycles": float(total_utilized_cycles / n_engines),
        "mean_true_lifespan": mean_true_lifespan,
        "scheduled_cost": float(scheduled_cost),
        "failure_cost": float(failure_cost),
        "early_pull_cost": float(early_pull_cost),
        "total_cost": total_cost,
        "cost_per_engine_life": cost_per_engine_life,
        "annual_cost_300_fleet": annual_cost_300_fleet,
        "annual_savings_300_fleet": 0.0,
        "savings_pct": 0.0,
    }


def simulate_predictive_policy(
    trajectories: Sequence[dict[str, Any]],
    threshold_cycles: int = 20,
    costs: CostAssumptions | None = None,
    baseline_interval_cycles: int | None = 150,
) -> dict[str, Any]:
    """Simulate a predictive RUL-threshold maintenance policy cycle by cycle.

    For each engine trajectory:
    - Finds the first cycle ``t`` where ``pred_rul[t] <= threshold_cycles`` and
      ``true_rul[t] > 0`` -> schedules maintenance pull at cycle ``t``.
      - Records ``true_rul_at_pull = true_rul[t]``.
      - If ``true_rul_at_pull > threshold_cycles + costs.early_pull_grace_cycles``,
        counts as an unnecessarily early pull with
        ``wasted_cycles = true_rul_at_pull - threshold_cycles``.
    - If never flagged while ``true_rul > 0``, counts as an unplanned failure
      incurring ``unscheduled_removal_cost + aog_cost_per_day * aog_days_per_failure``.
    """
    if not trajectories:
        raise ValueError("trajectories must not be empty.")
    if threshold_cycles < 0:
        raise ValueError("threshold_cycles must be non-negative.")
    cfg = costs if costs is not None else CostAssumptions()

    n_engines = len(trajectories)
    scheduled_pulls = 0
    unplanned_failures = 0
    engines_pulled_early = 0
    total_wasted_cycles = 0.0
    total_utilized_cycles = 0.0
    total_true_lifespan = 0.0
    pull_true_ruls: list[float] = []

    for traj in trajectories:
        _, cycles, true_rul, pred_rul = _normalize_trajectory(traj)
        max_cycle = float(cycles[-1] + max(0.0, true_rul[-1]))
        total_true_lifespan += max_cycle

        trigger_mask = (pred_rul <= float(threshold_cycles)) & (true_rul > 0.0)
        trigger_indices = np.flatnonzero(trigger_mask)

        if trigger_indices.size > 0:
            t = int(trigger_indices[0])
            scheduled_pulls += 1
            true_rul_at_pull = float(true_rul[t])
            pull_true_ruls.append(true_rul_at_pull)
            total_utilized_cycles += float(cycles[t])

            if true_rul_at_pull > float(threshold_cycles + cfg.early_pull_grace_cycles):
                engines_pulled_early += 1
                total_wasted_cycles += float(true_rul_at_pull - float(threshold_cycles))
        else:
            unplanned_failures += 1
            total_utilized_cycles += max_cycle

    scheduled_cost = scheduled_pulls * cfg.scheduled_maintenance_cost
    failure_cost = unplanned_failures * cfg.unplanned_failure_cost
    early_pull_cost = total_wasted_cycles * cfg.early_pull_cost_per_cycle
    total_cost = float(scheduled_cost + failure_cost + early_pull_cost)

    cost_per_engine_life = float(total_cost / n_engines)
    mean_true_lifespan = float(total_true_lifespan / n_engines)
    annual_cost_300_fleet = _scale_to_annual_fleet(
        cost_per_engine_life, mean_true_lifespan, cfg
    )

    if baseline_interval_cycles is not None:
        fixed_ref = simulate_fixed_interval_policy(
            trajectories,
            interval_cycles=baseline_interval_cycles,
            costs=cfg,
        )
        failures_avoided = int(fixed_ref["unplanned_failures"] - unplanned_failures)
        annual_savings = float(fixed_ref["annual_cost_300_fleet"] - annual_cost_300_fleet)
        savings_pct = (
            float(100.0 * annual_savings / fixed_ref["annual_cost_300_fleet"])
            if fixed_ref["annual_cost_300_fleet"] > 0.0
            else 0.0
        )
    else:
        failures_avoided = 0
        annual_savings = 0.0
        savings_pct = 0.0

    return {
        "policy": f"predictive_thr_{threshold_cycles}",
        "threshold_cycles": int(threshold_cycles),
        "baseline_interval_cycles": baseline_interval_cycles,
        "n_engines": int(n_engines),
        "scheduled_pulls": int(scheduled_pulls),
        "unplanned_failures": int(unplanned_failures),
        "unplanned_failure_rate": float(unplanned_failures / n_engines),
        "unplanned_failures_avoided_vs_fixed": failures_avoided,
        "engines_pulled_early": int(engines_pulled_early),
        "early_pull_rate": float(engines_pulled_early / n_engines),
        "avg_true_rul_at_pull": float(np.mean(pull_true_ruls)) if pull_true_ruls else 0.0,
        "total_wasted_cycles": float(total_wasted_cycles),
        "avg_wasted_cycles": float(total_wasted_cycles / n_engines),
        "avg_utilized_cycles": float(total_utilized_cycles / n_engines),
        "mean_true_lifespan": mean_true_lifespan,
        "scheduled_cost": float(scheduled_cost),
        "failure_cost": float(failure_cost),
        "early_pull_cost": float(early_pull_cost),
        "total_cost": total_cost,
        "cost_per_engine_life": cost_per_engine_life,
        "annual_cost_300_fleet": annual_cost_300_fleet,
        "annual_savings_300_fleet": annual_savings,
        "savings_pct": savings_pct,
    }


def compare_maintenance_policies(
    trajectories: Sequence[dict[str, Any]],
    thresholds: Sequence[int] = (10, 15, 20, 25, 30, 35),
    fixed_intervals: Sequence[int] = (100, 120, 140, 150, 175, 200),
    default_threshold: int = 20,
    default_fixed_interval: int = 150,
    costs: CostAssumptions | None = None,
) -> dict[str, Any]:
    """Run predictive threshold sweep and fixed-interval baseline sweep, identifying savings."""
    cfg = costs if costs is not None else CostAssumptions()

    fixed_sweep = [
        simulate_fixed_interval_policy(
            trajectories,
            interval_cycles=int(iv),
            costs=cfg,
            reference_threshold=default_threshold,
        )
        for iv in fixed_intervals
    ]
    best_fixed = min(fixed_sweep, key=lambda r: r["annual_cost_300_fleet"])
    fixed_default = next(
        (r for r in fixed_sweep if r["interval_cycles"] == default_fixed_interval),
        simulate_fixed_interval_policy(
            trajectories,
            interval_cycles=default_fixed_interval,
            costs=cfg,
            reference_threshold=default_threshold,
        ),
    )

    predictive_sweep = [
        simulate_predictive_policy(
            trajectories,
            threshold_cycles=int(thr),
            costs=cfg,
            baseline_interval_cycles=default_fixed_interval,
        )
        for thr in thresholds
    ]
    predictive_default = next(
        (r for r in predictive_sweep if r["threshold_cycles"] == default_threshold),
        simulate_predictive_policy(
            trajectories,
            threshold_cycles=default_threshold,
            costs=cfg,
            baseline_interval_cycles=default_fixed_interval,
        ),
    )
    best_predictive = min(predictive_sweep, key=lambda r: r["annual_cost_300_fleet"])

    vs_best_fixed_savings = float(
        best_fixed["annual_cost_300_fleet"] - predictive_default["annual_cost_300_fleet"]
    )
    vs_best_fixed_pct = (
        float(100.0 * vs_best_fixed_savings / best_fixed["annual_cost_300_fleet"])
        if best_fixed["annual_cost_300_fleet"] > 0.0
        else 0.0
    )

    return {
        "cost_assumptions": asdict(cfg),
        "n_engines": int(len(trajectories)),
        "default_threshold_cycles": int(default_threshold),
        "default_fixed_interval_cycles": int(default_fixed_interval),
        "predictive_default": predictive_default,
        "best_predictive": best_predictive,
        "fixed_default": fixed_default,
        "best_fixed_interval": best_fixed,
        "predictive_vs_best_fixed": {
            "best_fixed_interval_cycles": int(best_fixed["interval_cycles"]),
            "unplanned_failures_avoided_vs_best_fixed": int(
                best_fixed["unplanned_failures"] - predictive_default["unplanned_failures"]
            ),
            "annual_savings_300_fleet": vs_best_fixed_savings,
            "savings_pct": vs_best_fixed_pct,
        },
        "predictive_sweep": predictive_sweep,
        "fixed_interval_sweep": fixed_sweep,
    }


def load_trajectories_from_dataframe(
    df: pd.DataFrame,
    dataset_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a trajectory DataFrame into simulator-ready engine dictionaries."""
    if df.empty:
        raise ValueError("Trajectory DataFrame is empty.")

    cycle_col = "cycle" if "cycle" in df.columns else "cycles"
    true_col = (
        "true_rul_unclipped"
        if "true_rul_unclipped" in df.columns
        else ("true_rul" if "true_rul" in df.columns else "RUL_raw")
    )
    pred_col = "pred_rul" if "pred_rul" in df.columns else "pred"

    missing = [
        c for c in ("unit_id", cycle_col, true_col, pred_col) if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Trajectory DataFrame missing required columns: {missing}")

    trajectories: list[dict[str, Any]] = []
    if "dataset_id" in df.columns:
        groups = df.groupby(["dataset_id", "unit_id"], sort=True)
    else:
        groups = df.groupby("unit_id", sort=True)

    for key, grp in groups:
        grp_sorted = grp.sort_values(cycle_col, kind="mergesort")
        if "dataset_id" in df.columns:
            ds_val, uid = str(key[0]), int(key[1])
        else:
            raw_uid = key[0] if isinstance(key, tuple) else key
            ds_val, uid = dataset_id, int(raw_uid)
        entry: dict[str, Any] = {
            "unit_id": uid,
            "cycles": grp_sorted[cycle_col].to_numpy(dtype=np.float64),
            "true_rul": grp_sorted[true_col].to_numpy(dtype=np.float64),
            "pred_rul": grp_sorted[pred_col].to_numpy(dtype=np.float64),
        }
        if ds_val is not None:
            entry["dataset_id"] = ds_val
        trajectories.append(entry)
    return trajectories


def load_trajectories_from_file(
    path: Path | str,
    dataset_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load validation engine trajectories from `.csv`, `.parquet`, or `.npz`."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
        return load_trajectories_from_dataframe(df, dataset_id=dataset_id)
    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
        return load_trajectories_from_dataframe(df, dataset_id=dataset_id)
    if suffix == ".npz":
        data = np.load(file_path, allow_pickle=False)
        required = {"unit_id", "pred_rul"}
        if not required.issubset(set(data.files)):
            raise KeyError(f"NPZ file {file_path} missing required keys {required}")
        cycle_key = "cycle" if "cycle" in data.files else "cycles"
        true_key = (
            "true_rul_unclipped"
            if "true_rul_unclipped" in data.files
            else ("true_rul" if "true_rul" in data.files else "RUL_raw")
        )
        df = pd.DataFrame(
            {
                "unit_id": data["unit_id"],
                "cycle": data[cycle_key],
                "true_rul_unclipped": data[true_key],
                "pred_rul": data["pred_rul"],
            }
        )
        return load_trajectories_from_dataframe(df, dataset_id=dataset_id)

    raise ValueError(f"Unsupported trajectory file extension: {file_path.suffix}")


def generate_demo_trajectories(
    n_engines: int = 120,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic run-to-failure engine trajectories ONLY for `--demo` / unit tests."""
    rng = np.random.default_rng(seed)
    trajectories: list[dict[str, Any]] = []
    for uid in range(1, n_engines + 1):
        max_cycle = int(rng.integers(135, 320))
        cycles = np.arange(1, max_cycle + 1, dtype=np.float64)
        true_rul = max_cycle - cycles
        noise = rng.normal(loc=0.0, scale=6.0, size=cycles.size)
        bias = rng.normal(loc=-2.0, scale=4.0)
        pred_rul = np.clip(np.minimum(true_rul, 125.0) + bias + noise, 0.0, 130.0)
        trajectories.append(
            {
                "unit_id": uid,
                "dataset_id": "DEMO",
                "cycles": cycles,
                "true_rul": true_rul,
                "pred_rul": pred_rul,
            }
        )
    return trajectories
