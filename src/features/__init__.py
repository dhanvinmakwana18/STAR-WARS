"""Feature engineering, latent fault-mode clustering, and sequence windowing."""

from src.features.engineering import (
    FaultModeClusterer,
    add_tabular_features,
    build_sequence_windows,
)
from src.features.physics_degradation import (
    SAXENA_FEATURE_COLS,
    SaxenaPhysicsExtractor,
    fit_saxena_curve_nls,
    saxena_model_ab,
)

__all__ = [
    "add_tabular_features",
    "FaultModeClusterer",
    "build_sequence_windows",
    "SAXENA_FEATURE_COLS",
    "SaxenaPhysicsExtractor",
    "fit_saxena_curve_nls",
    "saxena_model_ab",
]
