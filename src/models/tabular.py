"""Tabular regression models (Random Forest and XGBoost) with split-conformal prediction intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


_CANONICAL_MODEL_TYPES = {
    "rf": "random_forest",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
}


def compute_conformal_quantile_from_residuals(
    abs_residuals: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Compute finite-sample split-conformal quantile from non-negative absolute residuals."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    res = np.asarray(abs_residuals, dtype=np.float64).ravel()
    if res.size == 0:
        raise ValueError("abs_residuals must be non-empty.")
    n = res.size
    q_level = min(1.0, float(np.ceil((1.0 - alpha) * (n + 1)) / n))
    return float(np.quantile(res, q_level))


def compute_conformal_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Compute split-conformal absolute-residual quantile for (1 - alpha) coverage."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    return compute_conformal_quantile_from_residuals(np.abs(yt - yp), alpha=alpha)


def _canonical_model_type(model_type: str) -> str:
    key = model_type.strip().lower()
    if key not in _CANONICAL_MODEL_TYPES:
        raise ValueError(
            f"Unsupported tabular model_type='{model_type}'. "
            "Expected 'random_forest' (or 'rf') or 'xgboost' (or 'xgb')."
        )
    return _CANONICAL_MODEL_TYPES[key]


class TabularRULModel:
    """Wrapper for RandomForestRegressor / XGBRegressor with conformal prediction intervals."""

    def __init__(
        self,
        model_type: str = "xgboost",
        feature_cols: list[str] | None = None,
        alpha: float = 0.10,
        random_state: int = 42,
        model_params: dict[str, Any] | None = None,
        early_stopping_rounds: int | None = 30,
        clip_lower: float | None = 0.0,
        clip_upper: float | None = None,
    ) -> None:
        self.model_type = _canonical_model_type(model_type)
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.model_params = dict(model_params) if model_params is not None else {}
        self.early_stopping_rounds = early_stopping_rounds
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

        self.model: RandomForestRegressor | XGBRegressor | None = None
        self.feature_cols_: list[str] | None = (
            list(feature_cols) if feature_cols is not None else None
        )
        self.conformal_q_: float = 0.0
        self.val_abs_residuals_: np.ndarray | None = None
        self.val_residual_std_: float = 0.0
        self.best_iteration_: int | None = None
        self.calibration_source_: str = "uncalibrated"
        self.metadata_: dict[str, Any] = {}

    def _extract_features(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_cols: list[str] | None = None,
        is_fit: bool = False,
    ) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            cols = feature_cols or self.feature_cols_
            if cols is None:
                cols = list(X.columns)
            if is_fit:
                self.feature_cols_ = list(cols)
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise KeyError(f"Missing feature columns in input DataFrame: {missing[:5]}")
            arr = X[cols].to_numpy(dtype=np.float32)
            return arr

        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for tabular features, got shape {arr.shape}")
        if is_fit and self.feature_cols_ is None:
            self.feature_cols_ = [f"f_{i}" for i in range(arr.shape[1])]
        return arr

    def _apply_clip(self, preds: np.ndarray) -> np.ndarray:
        out = np.asarray(preds, dtype=np.float64)
        if self.clip_lower is not None or self.clip_upper is not None:
            out = np.clip(out, self.clip_lower, self.clip_upper)
        return out

    def _build_estimator(self, has_val: bool) -> RandomForestRegressor | XGBRegressor:
        if self.model_type == "random_forest":
            defaults: dict[str, Any] = {
                "n_estimators": 160,
                "max_depth": 18,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "n_jobs": -1,
                "random_state": self.random_state,
            }
            defaults.update(self.model_params)
            return RandomForestRegressor(**defaults)

        defaults = {
            "n_estimators": 1200 if has_val else 350,
            "learning_rate": 0.04,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": self.random_state,
            "eval_metric": "rmse",
        }
        defaults.update(self.model_params)
        if has_val and self.early_stopping_rounds:
            defaults.setdefault("early_stopping_rounds", self.early_stopping_rounds)
        else:
            defaults.pop("early_stopping_rounds", None)
        return XGBRegressor(**defaults)

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        feature_cols: list[str] | None = None,
    ) -> TabularRULModel:
        """Fit the tabular regressor and calibrate split-conformal bands on validation data."""
        cols = feature_cols if feature_cols is not None else self.feature_cols
        X_tr = self._extract_features(X_train, feature_cols=cols, is_fit=True)
        y_tr = np.asarray(y_train, dtype=np.float64).ravel()
        if X_tr.shape[0] != y_tr.shape[0]:
            raise ValueError(f"X_train rows ({X_tr.shape[0]}) != y_train length ({y_tr.shape[0]})")

        has_val = X_val is not None and y_val is not None
        self.model = self._build_estimator(has_val=has_val)

        if has_val:
            X_va = self._extract_features(X_val, feature_cols=self.feature_cols_, is_fit=False)
            y_va = np.asarray(y_val, dtype=np.float64).ravel()
            if self.model_type == "xgboost":
                self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                best_iter = getattr(self.model, "best_iteration", None)
                self.best_iteration_ = int(best_iter) if best_iter is not None else None
            else:
                self.model.fit(X_tr, y_tr)
            self.calibrate_conformal(X_va, y_va, alpha=self.alpha)
            self.calibration_source_ = "validation"
        else:
            self.model.fit(X_tr, y_tr)
            self.calibrate_conformal(X_tr, y_tr, alpha=self.alpha)
            self.calibration_source_ = "train_fallback"

        return self

    def calibrate_conformal(
        self,
        X_cal: pd.DataFrame | np.ndarray,
        y_cal: pd.Series | np.ndarray,
        alpha: float | None = None,
    ) -> float:
        """Calibrate split-conformal residual quantiles on a held-out calibration set."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before calibrating conformal intervals.")
        alpha_val = self.alpha if alpha is None else float(alpha)
        pred_cal = self.predict(X_cal)
        y_true = np.asarray(y_cal, dtype=np.float64).ravel()
        residuals = y_true - pred_cal
        self.val_abs_residuals_ = np.abs(residuals)
        self.val_residual_std_ = float(np.std(residuals))
        self.conformal_q_ = compute_conformal_quantile_from_residuals(
            self.val_abs_residuals_, alpha=alpha_val
        )
        return self.conformal_q_

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict RUL values for tabular feature snapshots."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = self._extract_features(X, feature_cols=self.feature_cols_, is_fit=False)
        raw_preds = self.model.predict(X_arr)
        return self._apply_clip(raw_preds)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame | np.ndarray,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict RUL with uncertainty std and conformal (1 - alpha) prediction intervals.

        Returns:
            (pred, pred_std, ci_low, ci_high) where ci_low <= pred <= ci_high.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        alpha_val = self.alpha if alpha is None else float(alpha)
        X_arr = self._extract_features(X, feature_cols=self.feature_cols_, is_fit=False)
        pred = self._apply_clip(self.model.predict(X_arr))

        if self.val_abs_residuals_ is not None and self.val_abs_residuals_.size > 0:
            q_hat = compute_conformal_quantile_from_residuals(
                self.val_abs_residuals_, alpha=alpha_val
            )
        else:
            q_hat = float(self.conformal_q_)

        if self.model_type == "random_forest" and hasattr(self.model, "estimators_"):
            tree_preds = np.stack(
                [est.predict(X_arr) for est in self.model.estimators_],
                axis=0,
            )
            pred_std = np.std(tree_preds, axis=0)
        else:
            fallback_std = self.val_residual_std_ if self.val_residual_std_ > 0 else q_hat / 1.64485
            pred_std = np.full_like(pred, fallback_std, dtype=np.float64)

        ci_low = pred - q_hat
        if self.clip_lower is not None:
            ci_low = np.maximum(self.clip_lower, ci_low)
        ci_low = np.minimum(ci_low, pred)
        ci_high = np.maximum(pred, pred + q_hat)
        return pred, pred_std, ci_low, ci_high

    def predict_interval(
        self,
        X: pd.DataFrame | np.ndarray,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience wrapper returning (pred, ci_low, ci_high)."""
        pred, _, ci_low, ci_high = self.predict_with_uncertainty(X, alpha=alpha)
        return pred, ci_low, ci_high

    def feature_importances(self, top_k: int | None = None) -> pd.DataFrame:
        """Return sorted feature importances as a DataFrame."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        importances = np.asarray(self.model.feature_importances_, dtype=np.float64)
        cols = self.feature_cols_ or [f"f_{i}" for i in range(len(importances))]
        df = (
            pd.DataFrame({"feature": cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        if top_k is not None:
            return df.head(top_k).reset_index(drop=True)
        return df

    def save(self, path: str | Path) -> Path:
        """Persist fitted TabularRULModel via joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> TabularRULModel:
        """Load a persisted TabularRULModel via joblib."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected TabularRULModel in {path}, got {type(obj).__name__}")
        return obj


def train_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    feature_cols: list[str] | None = None,
    **kwargs: Any,
) -> TabularRULModel:
    """Train a RandomForestRegressor with optional validation conformal calibration."""
    model = TabularRULModel(model_type="random_forest", feature_cols=feature_cols, **kwargs)
    return model.fit(X_train, y_train, X_val=X_val, y_val=y_val)


def train_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    feature_cols: list[str] | None = None,
    **kwargs: Any,
) -> TabularRULModel:
    """Train an XGBRegressor (hist, early stopping on val) with conformal calibration."""
    model = TabularRULModel(model_type="xgboost", feature_cols=feature_cols, **kwargs)
    return model.fit(X_train, y_train, X_val=X_val, y_val=y_val)


def save_tabular_model(model: TabularRULModel, path: str | Path) -> Path:
    """Save a TabularRULModel to disk via joblib."""
    return model.save(path)


def load_tabular_model(path: str | Path) -> TabularRULModel:
    """Load a TabularRULModel from disk via joblib."""
    return TabularRULModel.load(path)
