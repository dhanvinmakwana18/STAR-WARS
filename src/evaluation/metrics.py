"""Evaluation metrics for NASA C-MAPSS turbofan Remaining Useful Life (RUL).

Implements standard regression metrics (RMSE, MAE), the NASA asymmetric
scoring function from Saxena et al. (2008) Eq. 11, early/late error diagnostics,
and a unified evaluation summary function.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

_MAX_EXPONENT = 500.0


def _validate_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert target and prediction inputs to matching 1D float64 arrays."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.size == 0:
        raise ValueError("y_true and y_pred must contain at least one sample.")
    if yt.shape != yp.shape:
        raise ValueError(
            f"Shape mismatch: y_true has {yt.shape[0]} elements, y_pred has {yp.shape[0]}."
        )
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        raise ValueError("y_true and y_pred must contain only finite values.")
    return yt, yp


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Root Mean Squared Error (RMSE) between true and predicted RUL."""
    yt, yp = _validate_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Mean Absolute Error (MAE) between true and predicted RUL."""
    yt, yp = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(yp - yt)))


def nasa_asymmetric_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    a1_early: float = 13.0,
    a2_late: float = 10.0,
) -> float:
    """Compute the NASA C-MAPSS asymmetric scoring function (Saxena et al., 2008, Eq. 11).

    Let ``d_i = y_pred_i - y_true_i``:
    - Early prediction (``d_i < 0``, underestimating RUL):
      ``s_i = exp(-d_i / a1_early) - 1 = exp(|d_i| / 13) - 1``
    - Late prediction (``d_i >= 0``, overestimating RUL):
      ``s_i = exp(d_i / a2_late) - 1 = exp(|d_i| / 10) - 1``

    Why ``a2_late = 10.0`` vs ``a1_early = 13.0``:
    Because ``10 < 13``, ``exp(|d| / 10) > exp(|d| / 13)`` for every nonzero error.
    Overestimating RUL (predicting the engine still has life left when failure is
    closer) incurs an exponentially harsher penalty than an equal-magnitude early
    warning, reflecting the asymmetric safety and AOG cost of missed failures.

    Exponent values are guarded at ``500.0`` solely to prevent float64 overflow on
    degenerate outliers; for normal C-MAPSS RUL ranges (<= 200), ``max |d| / 10 <= 20``
    and ``exp(20)`` fits comfortably in float64 without hitting the guard.
    """
    if a1_early <= 0.0 or a2_late <= 0.0:
        raise ValueError("a1_early and a2_late must be strictly positive.")

    yt, yp = _validate_pair(y_true, y_pred)
    d = yp - yt

    exponents = np.where(d < 0.0, -d / a1_early, d / a2_late)
    exponents = np.clip(exponents, a_min=None, a_max=_MAX_EXPONENT)
    scores = np.exp(exponents) - 1.0
    return float(np.sum(scores))


def mean_nasa_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    a1_early: float = 13.0,
    a2_late: float = 10.0,
) -> float:
    """Compute mean per-engine NASA asymmetric score (total score / n_samples)."""
    yt, yp = _validate_pair(y_true, y_pred)
    total = nasa_asymmetric_score(yt, yp, a1_early=a1_early, a2_late=a2_late)
    return float(total / yt.size)


def early_late_breakdown(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, Any]:
    """Break down prediction errors into early (d < 0), late (d > 0), and exact (d == 0).

    Here ``d = y_pred - y_true``:
    - ``n_early``: count with ``d < 0`` (conservative / early warning)
    - ``n_late``: count with ``d > 0`` (dangerous RUL overestimation)
    - ``n_exact``: count with ``d == 0``
    - ``early_pct`` / ``late_pct``: percentages in ``[0, 100]``
    - ``mean_bias``: ``mean(y_pred - y_true)`` (positive indicates late bias)
    """
    yt, yp = _validate_pair(y_true, y_pred)
    d = yp - yt
    n = int(yt.size)
    n_early = int(np.sum(d < 0.0))
    n_late = int(np.sum(d > 0.0))
    n_exact = int(np.sum(d == 0.0))
    return {
        "n_early": n_early,
        "n_late": n_late,
        "n_exact": n_exact,
        "early_pct": float(100.0 * n_early / n),
        "late_pct": float(100.0 * n_late / n),
        "mean_bias": float(np.mean(d)),
    }


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    clip_cap: int | None = None,
    clip_pred: bool = False,
) -> dict[str, Any]:
    """Evaluate RUL predictions across RMSE, MAE, NASA score, and early/late diagnostics.

    Parameters
    ----------
    y_true:
        Ground-truth RUL values.
    y_pred:
        Predicted RUL values.
    clip_cap:
        Optional piecewise-linear upper cap (e.g. ``125`` for C-MAPSS benchmark
        comparison). When set, ``y_true`` is clipped at ``upper=clip_cap`` (and
        ``y_pred`` is clipped to ``[0, clip_cap]`` only if ``clip_pred=True``).
    clip_pred:
        If True and ``clip_cap`` is provided, clips predictions to ``[0, clip_cap]``.
    """
    yt, yp = _validate_pair(y_true, y_pred)
    if clip_cap is not None:
        if clip_cap <= 0:
            raise ValueError("clip_cap must be positive when provided.")
        yt = np.minimum(yt, float(clip_cap))
        if clip_pred:
            yp = np.clip(yp, 0.0, float(clip_cap))

    breakdown = early_late_breakdown(yt, yp)
    n_samples = int(yt.size)
    total_nasa = nasa_asymmetric_score(yt, yp)

    return {
        "rmse": rmse(yt, yp),
        "mae": mae(yt, yp),
        "nasa_score": total_nasa,
        "mean_nasa_score": float(total_nasa / n_samples),
        "n_samples": n_samples,
        "n_early": breakdown["n_early"],
        "n_late": breakdown["n_late"],
        "n_exact": breakdown["n_exact"],
        "early_pct": breakdown["early_pct"],
        "late_pct": breakdown["late_pct"],
        "late_rate": float(breakdown["n_late"] / n_samples),
        "mean_bias": breakdown["mean_bias"],
    }
