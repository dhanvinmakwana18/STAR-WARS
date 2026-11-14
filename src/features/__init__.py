"""Feature engineering, latent fault-mode clustering, and sequence windowing."""

from src.features.engineering import (
    FaultModeClusterer,
    add_tabular_features,
    build_sequence_windows,
)

__all__ = [
    "add_tabular_features",
    "FaultModeClusterer",
    "build_sequence_windows",
]
