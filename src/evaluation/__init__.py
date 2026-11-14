"""Evaluation metrics, literature benchmarks, and fleet maintenance simulation."""

from __future__ import annotations

from src.evaluation.benchmarks import (
    PUBLISHED_BENCHMARKS,
    PUBLISHED_BENCHMARKS_DF,
    benchmarks_to_dataframe,
    compare_against_benchmarks,
)
from src.evaluation.maintenance import (
    CostAssumptions,
    compare_maintenance_policies,
    generate_demo_trajectories,
    load_trajectories_from_dataframe,
    load_trajectories_from_file,
    simulate_fixed_interval_policy,
    simulate_predictive_policy,
)
from src.evaluation.metrics import (
    early_late_breakdown,
    evaluate_predictions,
    mae,
    mean_nasa_score,
    nasa_asymmetric_score,
    rmse,
)

__all__ = [
    "PUBLISHED_BENCHMARKS",
    "PUBLISHED_BENCHMARKS_DF",
    "benchmarks_to_dataframe",
    "compare_against_benchmarks",
    "CostAssumptions",
    "compare_maintenance_policies",
    "generate_demo_trajectories",
    "load_trajectories_from_dataframe",
    "load_trajectories_from_file",
    "simulate_fixed_interval_policy",
    "simulate_predictive_policy",
    "early_late_breakdown",
    "evaluate_predictions",
    "mae",
    "mean_nasa_score",
    "nasa_asymmetric_score",
    "rmse",
]
