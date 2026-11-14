"""Central project configuration, paths, and C-MAPSS dataset constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
RESULTS_DIR: Path = REPORTS_DIR / "results"

# Windows-safe Matplotlib cache directory and loky core-count hint
_MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

DATASETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

DATASET_META: dict[str, dict[str, int | str]] = {
    "FD001": {
        "op_conditions": 1,
        "fault_modes": 1,
        "description": "Single operating condition (sea level), single fault mode (HPC degradation)",
    },
    "FD002": {
        "op_conditions": 6,
        "fault_modes": 1,
        "description": "Six operating conditions, single fault mode (HPC degradation)",
    },
    "FD003": {
        "op_conditions": 1,
        "fault_modes": 2,
        "description": "Single operating condition (sea level), two fault modes (HPC + Fan degradation)",
    },
    "FD004": {
        "op_conditions": 6,
        "fault_modes": 2,
        "description": "Six operating conditions, two fault modes (HPC + Fan degradation)",
    },
}

OP_SETTING_COLS: list[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS: list[str] = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS: list[str] = ["unit_id", "cycle"] + OP_SETTING_COLS + SENSOR_COLS

DROP_SENSORS_CANONICAL: list[str] = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]
INFORMATIVE_SENSORS: list[str] = [
    c for c in SENSOR_COLS if c not in DROP_SENSORS_CANONICAL
]

RUL_CLIP: int = 125
WINDOW_SIZE: int = 30
ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20)
RANDOM_SEED: int = 42


def ensure_project_dirs() -> None:
    """Create standard output directories if they do not exist."""
    for directory in (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        RESULTS_DIR,
        _MPL_CACHE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
