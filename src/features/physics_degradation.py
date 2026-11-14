"""Physics-informed degradation parameter extraction based on Saxena et al. (2008) Eq. 4/5.

Models the generalized exponential health/degradation index from the C-MAPSS
generative specification:

    h(t) = 1 - exp(-a * t^b)

where ``a > 0`` and ``b > 0`` are unit-specific degradation-rate parameters and ``t``
is the operating cycle. To prevent numerical ill-conditioning across orders of
magnitude in ``a`` (e.g. 1e-9 to 1e-2 as ``b`` varies between 1.0 and 4.0), nonlinear
least squares is parameterized in log-scale ``alpha = ln(a)``:

    h(t; alpha, b) = 1 - exp(-exp(alpha + b * ln(t)))

All prefix curve fits at cycle ``t`` use strictly cycles ``1..t`` of that engine
to guarantee zero future information leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.config import INFORMATIVE_SENSORS

# Default top HPC/EGT margin sensors from Phase 1 EDA if RUL is not present at fit time
DEFAULT_HPC_EGT_SENSORS: tuple[str, ...] = (
    "sensor_11",  # Ps30 — HPC outlet static pressure
    "sensor_4",   # T50  — LPT outlet / EGT total temperature
    "sensor_12",  # phi  — Fuel flow to Ps30 ratio
    "sensor_7",   # P30  — HPC outlet total pressure
    "sensor_15",  # BPR  — Bypass ratio
    "sensor_21",  # W32  — LPT coolant bleed
    "sensor_20",  # W31  — HPT coolant bleed
    "sensor_17",  # htBleed — Bleed enthalpy
)

# Canonical sign orientation (+1 means raw sensor increases with degradation; -1 means decreases)
DEFAULT_HPC_EGT_SIGNS: dict[str, float] = {
    "sensor_2": 1.0,
    "sensor_3": 1.0,
    "sensor_4": 1.0,
    "sensor_7": -1.0,
    "sensor_8": 1.0,
    "sensor_9": 1.0,
    "sensor_11": 1.0,
    "sensor_12": -1.0,
    "sensor_13": 1.0,
    "sensor_14": 1.0,
    "sensor_15": 1.0,
    "sensor_17": 1.0,
    "sensor_20": -1.0,
    "sensor_21": -1.0,
}

SAXENA_FEATURE_COLS: list[str] = [
    "saxena_log_a",
    "saxena_b",
    "saxena_hi",
    "saxena_composite",
    "saxena_r2",
]


def saxena_model_ab(t: np.ndarray | float, a: float, b: float) -> np.ndarray:
    """Evaluate h(t) = 1 - exp(-a * t^b) in direct (a, b) parameterization."""
    t_arr = np.maximum(np.asarray(t, dtype=np.float64), 1.0)
    u = np.clip(float(a) * np.power(t_arr, float(b)), 0.0, 1e6)
    return 1.0 - np.exp(-u)


def saxena_model_reparam(t: np.ndarray, alpha: float, b: float) -> np.ndarray:
    """Evaluate h(t) = 1 - exp(-exp(alpha + b * ln(t))) where alpha = ln(a)."""
    ln_t = np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0))
    z = np.clip(float(alpha) + float(b) * ln_t, -30.0, 10.0)
    return 1.0 - np.exp(-np.exp(z))


def saxena_jacobian_reparam(t: np.ndarray, alpha: float, b: float) -> np.ndarray:
    """Analytical Jacobian [dh/d(alpha), dh/db] for scipy.optimize.curve_fit."""
    ln_t = np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0))
    z = np.clip(float(alpha) + float(b) * ln_t, -30.0, 10.0)
    u = np.exp(z)
    e = np.exp(-u)
    ja = e * u
    jb = ja * ln_t
    return np.column_stack([ja, jb])


def _warm_start_log_linear(
    ln_t: np.ndarray,
    d_obs: np.ndarray,
    prior_alpha: float = -12.0,
    prior_b: float = 2.2,
    alpha_bounds: tuple[float, float] = (-26.0, -1.0),
    b_bounds: tuple[float, float] = (0.4, 5.0),
) -> tuple[float, float]:
    """Compute weighted log-log linear regression warm start for (alpha, b)."""
    d_clipped = np.clip(d_obs, 1e-4, 0.995)
    u = np.log(np.maximum(-np.log(1.0 - d_clipped), 1e-10))
    w = (d_clipped ** 2) + 1e-3
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12 or ln_t.size < 3:
        return float(prior_alpha), float(prior_b)

    mt = float(np.dot(w, ln_t) / w_sum)
    mu = float(np.dot(w, u) / w_sum)
    dt = ln_t - mt
    denom = float(np.dot(w, dt * dt))
    if denom <= 1e-9:
        return float(prior_alpha), float(prior_b)

    b0 = float(np.dot(w, dt * (u - mu)) / denom)
    b0 = float(np.clip(b0, b_bounds[0], b_bounds[1]))
    alpha0 = float(np.clip(mu - b0 * mt, alpha_bounds[0], alpha_bounds[1]))
    return alpha0, b0


def _fit_lm_prefix(
    ln_t: np.ndarray,
    d_obs: np.ndarray,
    alpha0: float,
    b0: float,
    n_iter: int = 10,
    alpha_bounds: tuple[float, float] = (-26.0, -0.5),
    b_bounds: tuple[float, float] = (0.4, 5.0),
) -> tuple[float, float, float]:
    """Damped Gauss-Newton / Levenberg-Marquardt nonlinear least squares on a prefix."""
    alpha = float(np.clip(alpha0, alpha_bounds[0], alpha_bounds[1]))
    b = float(np.clip(b0, b_bounds[0], b_bounds[1]))
    lam = 1e-2

    z = np.clip(alpha + b * ln_t, -30.0, 10.0)
    u = np.exp(z)
    e = np.exp(-u)
    pred = 1.0 - e
    best_sse = float(np.sum((d_obs - pred) ** 2))

    for _ in range(n_iter):
        r = d_obs - pred
        ja = e * u
        jb = ja * ln_t
        g0 = float(np.dot(ja, r))
        g1 = float(np.dot(jb, r))
        h00 = float(np.dot(ja, ja)) + lam
        h01 = float(np.dot(ja, jb))
        h11 = float(np.dot(jb, jb)) + lam
        det = h00 * h11 - h01 * h01
        if det <= 1e-18:
            break

        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        cand_alpha = float(np.clip(alpha + da, alpha_bounds[0], alpha_bounds[1]))
        cand_b = float(np.clip(b + db, b_bounds[0], b_bounds[1]))

        z_c = np.clip(cand_alpha + cand_b * ln_t, -30.0, 10.0)
        u_c = np.exp(z_c)
        e_c = np.exp(-u_c)
        pred_c = 1.0 - e_c
        sse_c = float(np.sum((d_obs - pred_c) ** 2))

        if sse_c <= best_sse:
            alpha, b = cand_alpha, cand_b
            z, u, e, pred = z_c, u_c, e_c, pred_c
            best_sse = sse_c
            lam = max(1e-5, lam * 0.5)
            if abs(da) < 1e-5 and abs(db) < 1e-5:
                break
        else:
            lam = min(1e2, lam * 4.0)

    ss_tot = float(np.sum((d_obs - np.mean(d_obs)) ** 2))
    r2 = float(np.clip(1.0 - best_sse / max(ss_tot, 1e-8), 0.0, 1.0)) if ss_tot > 1e-7 else 0.0
    return alpha, b, r2


def fit_saxena_curve_nls(
    cycles: np.ndarray,
    d_obs: np.ndarray,
    prior_alpha: float = -12.0,
    prior_b: float = 2.2,
    alpha_bounds: tuple[float, float] = (-26.0, -0.5),
    b_bounds: tuple[float, float] = (0.4, 5.0),
    use_scipy: bool = True,
) -> dict[str, float]:
    """Fit h(t) = 1 - exp(-a * t^b) via nonlinear least squares on a 1D trajectory.

    Uses weighted log-log warm starting followed by ``scipy.optimize.curve_fit``
    with analytical Jacobian (falling back to damped Levenberg-Marquardt if needed).
    """
    t_arr = np.maximum(np.asarray(cycles, dtype=np.float64).ravel(), 1.0)
    d_arr = np.clip(np.asarray(d_obs, dtype=np.float64).ravel(), 1e-4, 0.995)
    if t_arr.shape != d_arr.shape or t_arr.size == 0:
        raise ValueError("cycles and d_obs must be non-empty 1D arrays of matching length.")

    ln_t = np.log(t_arr)
    alpha0, b0 = _warm_start_log_linear(
        ln_t,
        d_arr,
        prior_alpha=prior_alpha,
        prior_b=prior_b,
        alpha_bounds=alpha_bounds,
        b_bounds=b_bounds,
    )

    if use_scipy and t_arr.size >= 4:
        try:
            popt, _ = curve_fit(
                saxena_model_reparam,
                t_arr,
                d_arr,
                p0=[alpha0, b0],
                jac=saxena_jacobian_reparam,
                bounds=([alpha_bounds[0], b_bounds[0]], [alpha_bounds[1], b_bounds[1]]),
                maxfev=150,
            )
            alpha_fit, b_fit = float(popt[0]), float(popt[1])
            pred = saxena_model_reparam(t_arr, alpha_fit, b_fit)
            sse = float(np.sum((d_arr - pred) ** 2))
            ss_tot = float(np.sum((d_arr - np.mean(d_arr)) ** 2))
            r2 = float(np.clip(1.0 - sse / max(ss_tot, 1e-8), 0.0, 1.0)) if ss_tot > 1e-7 else 0.0
        except Exception:
            alpha_fit, b_fit, r2 = _fit_lm_prefix(
                ln_t, d_arr, alpha0, b0, alpha_bounds=alpha_bounds, b_bounds=b_bounds
            )
    else:
        alpha_fit, b_fit, r2 = _fit_lm_prefix(
            ln_t, d_arr, alpha0, b0, alpha_bounds=alpha_bounds, b_bounds=b_bounds
        )

    a_fit = float(np.exp(alpha_fit))
    log10_a = float(alpha_fit / np.log(10.0))
    hi_last = float(saxena_model_reparam(t_arr[-1:], alpha_fit, b_fit)[0])
    return {
        "a": a_fit,
        "alpha": alpha_fit,
        "log10_a": log10_a,
        "b": b_fit,
        "r2": r2,
        "hi_last": hi_last,
    }


class SaxenaPhysicsExtractor:
    """Causal per-engine extractor for Saxena et al. (2008) (a, b) degradation parameters.

    1. Fits composite sensor weights on the regime-normalized training split using
       the top ``top_k_sensors`` most correlated with RUL (HPC/EGT margin proxies).
    2. Constructs a causally smoothed, baseline-referenced composite signal in ``[0, 1)``.
    3. At each cycle ``t``, fits ``h(tau) = 1 - exp(-a * tau^b)`` strictly over the
       causal prefix ``tau in 1..t`` via nonlinear least squares, applying empirical-Bayes
       shrinkage toward the training prior when ``t < min_cycles`` (where curve fitting
       is underdetermined).
    """

    def __init__(
        self,
        sensor_cols: list[str] | None = None,
        top_k_sensors: int = 8,
        ewm_span: int = 10,
        baseline_cycles: int = 10,
        min_cycles: int = 18,
        shrinkage_ramp: int = 30,
        nls_stride: int = 1,
    ) -> None:
        self.sensor_cols = (
            list(sensor_cols) if sensor_cols is not None else list(INFORMATIVE_SENSORS)
        )
        self.top_k_sensors = int(top_k_sensors)
        self.ewm_span = int(ewm_span)
        self.baseline_cycles = int(baseline_cycles)
        self.min_cycles = int(min_cycles)
        self.shrinkage_ramp = int(max(1, shrinkage_ramp))
        self.nls_stride = int(max(1, nls_stride))

        self.selected_sensors_: list[str] = []
        self.sensor_weights_: np.ndarray | None = None
        self.excursion_scale_: float = 1.0
        self.prior_alpha_: float = -12.5
        self.prior_b_: float = 2.25
        self.is_fitted_: bool = False

    def _group_cols(self, df: pd.DataFrame) -> list[str]:
        return ["dataset_id", "unit_id"] if "dataset_id" in df.columns else ["unit_id"]

    def _compute_causal_composite(
        self,
        df: pd.DataFrame,
        fit_scale: bool = False,
    ) -> np.ndarray:
        """Compute causal smoothed composite signal d_obs(t) in [1e-4, 0.995] per engine."""
        if not self.selected_sensors_ or self.sensor_weights_ is None:
            raise RuntimeError("SaxenaPhysicsExtractor sensor weights are not initialized.")

        raw_proj = df[self.selected_sensors_].to_numpy(dtype=np.float64) @ self.sensor_weights_
        group_cols = self._group_cols(df)

        temp = pd.DataFrame({"_proj": raw_proj}, index=df.index)
        for c in group_cols:
            temp[c] = df[c].to_numpy()

        shifts = np.zeros(len(df), dtype=np.float64)
        alpha_ewm = 2.0 / (float(self.ewm_span) + 1.0)

        for _, grp_idx in temp.groupby(group_cols, sort=False).groups.items():
            pos = df.index.get_indexer(grp_idx)
            vals = raw_proj[pos]
            n = len(vals)
            if n == 0:
                continue
            # Causal EWMA
            smooth = np.empty(n, dtype=np.float64)
            smooth[0] = vals[0]
            for i in range(1, n):
                smooth[i] = alpha_ewm * vals[i] + (1.0 - alpha_ewm) * smooth[i - 1]

            # Causal initial baseline (expanding mean over first baseline_cycles, fixed thereafter)
            b_len = min(n, self.baseline_cycles)
            prefix_means = np.cumsum(smooth[:b_len]) / np.arange(1, b_len + 1, dtype=np.float64)
            base = np.empty(n, dtype=np.float64)
            base[:b_len] = prefix_means
            if n > b_len:
                base[b_len:] = prefix_means[-1]

            shifts[pos] = np.maximum(0.0, smooth - base)

        if fit_scale:
            pos_shifts = shifts[shifts > 1e-6]
            self.excursion_scale_ = (
                float(np.percentile(pos_shifts, 97.5)) if pos_shifts.size > 10 else 2.5
            )
            self.excursion_scale_ = max(self.excursion_scale_, 0.25)

        d_obs = np.clip(shifts / self.excursion_scale_, 1e-4, 0.995)
        return d_obs

    def fit(self, train_df: pd.DataFrame) -> "SaxenaPhysicsExtractor":
        """Fit composite sensor weights and empirical (alpha, b) priors on normalized train_df."""
        avail = [c for c in self.sensor_cols if c in train_df.columns]
        if not avail:
            raise ValueError("No valid sensor_cols found in train_df.")

        if "RUL" in train_df.columns and len(train_df) > 10:
            rul = train_df["RUL"].to_numpy(dtype=np.float64)
            corrs: list[tuple[str, float]] = []
            for col in avail:
                s_arr = train_df[col].to_numpy(dtype=np.float64)
                if np.std(s_arr) > 1e-8:
                    r = float(np.corrcoef(s_arr, rul)[0, 1])
                    if np.isfinite(r):
                        corrs.append((col, r))
            corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            chosen = corrs[: min(self.top_k_sensors, len(corrs))]
            self.selected_sensors_ = [c for c, _ in chosen]
            w = np.array([-np.sign(r) * abs(r) for _, r in chosen], dtype=np.float64)
        else:
            self.selected_sensors_ = [
                c for c in DEFAULT_HPC_EGT_SENSORS if c in avail
            ][: self.top_k_sensors]
            w = np.array(
                [DEFAULT_HPC_EGT_SIGNS.get(c, 1.0) for c in self.selected_sensors_],
                dtype=np.float64,
            )

        w_norm = float(np.sum(np.abs(w)))
        self.sensor_weights_ = w / max(w_norm, 1e-8)

        group_cols = self._group_cols(train_df)
        work = train_df.sort_values(group_cols + ["cycle"], kind="mergesort").reset_index(drop=True)
        d_obs = self._compute_causal_composite(work, fit_scale=True)

        alphas: list[float] = []
        bs: list[float] = []
        for _, grp in work.groupby(group_cols, sort=False):
            idx = grp.index.to_numpy()
            if len(idx) >= self.min_cycles:
                t_arr = grp["cycle"].to_numpy(dtype=np.float64)
                res = fit_saxena_curve_nls(
                    t_arr,
                    d_obs[idx],
                    prior_alpha=-12.5,
                    prior_b=2.25,
                    use_scipy=True,
                )
                if res["r2"] > 0.4:
                    alphas.append(res["alpha"])
                    bs.append(res["b"])

        if alphas and bs:
            self.prior_alpha_ = float(np.median(alphas))
            self.prior_b_ = float(np.median(bs))
        else:
            self.prior_alpha_ = -12.5
            self.prior_b_ = 2.25

        self.is_fitted_ = True
        return self

    def _fit_engine_prefixes(
        self,
        cycles: np.ndarray,
        d_obs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Causally fit (a_t, b_t) on every prefix 1..t of a single engine trajectory."""
        n = len(cycles)
        t_arr = np.maximum(cycles.astype(np.float64), 1.0)
        ln_t = np.log(t_arr)

        out_a = np.empty(n, dtype=np.float32)
        out_log_a = np.empty(n, dtype=np.float32)
        out_b = np.empty(n, dtype=np.float32)
        out_hi = np.empty(n, dtype=np.float32)
        out_r2 = np.zeros(n, dtype=np.float32)

        curr_alpha = float(self.prior_alpha_)
        curr_b = float(self.prior_b_)
        curr_r2 = 0.0

        for idx in range(n):
            n_pts = idx + 1
            if n_pts < self.min_cycles:
                alpha_t = float(self.prior_alpha_)
                b_t = float(self.prior_b_)
                r2_t = 0.0
            else:
                should_refit = (
                    (n_pts == self.min_cycles)
                    or ((n_pts - self.min_cycles) % self.nls_stride == 0)
                )
                if should_refit:
                    t_sub = ln_t[:n_pts]
                    d_sub = d_obs[:n_pts]
                    a0, b0 = _warm_start_log_linear(
                        t_sub,
                        d_sub,
                        prior_alpha=self.prior_alpha_,
                        prior_b=self.prior_b_,
                    )
                    raw_alpha, raw_b, raw_r2 = _fit_lm_prefix(
                        t_sub, d_sub, a0, b0, n_iter=8
                    )
                    # Empirical-Bayes shrinkage weight based on sample length & signal excursion
                    len_weight = min(1.0, (n_pts - self.min_cycles + 1) / float(self.shrinkage_ramp))
                    sig_weight = min(1.0, float(np.max(d_sub)) / 0.20)
                    trust = float(np.clip(len_weight * sig_weight, 0.0, 1.0))

                    curr_alpha = trust * raw_alpha + (1.0 - trust) * self.prior_alpha_
                    curr_b = trust * raw_b + (1.0 - trust) * self.prior_b_
                    curr_r2 = raw_r2 * trust

                alpha_t = curr_alpha
                b_t = curr_b
                r2_t = curr_r2

            out_a[idx] = float(np.exp(np.clip(alpha_t, -26.0, 0.0)))
            out_log_a[idx] = float(alpha_t / np.log(10.0))
            out_b[idx] = float(b_t)
            z_now = float(np.clip(alpha_t + b_t * ln_t[idx], -30.0, 10.0))
            out_hi[idx] = float(1.0 - np.exp(-np.exp(z_now)))
            out_r2[idx] = float(r2_t)

        return {
            "saxena_a": out_a,
            "saxena_log_a": out_log_a,
            "saxena_b": out_b,
            "saxena_hi": out_hi,
            "saxena_composite": d_obs.astype(np.float32),
            "saxena_r2": out_r2,
        }

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Attach causal Saxena physics features to a copy of df."""
        if not self.is_fitted_:
            raise RuntimeError("SaxenaPhysicsExtractor must be fitted before transform().")

        if len(df) == 0:
            out = df.copy()
            for col in ["saxena_a", *SAXENA_FEATURE_COLS]:
                out[col] = pd.Series(dtype=np.float32, index=out.index)
            return out, list(SAXENA_FEATURE_COLS)

        group_cols = self._group_cols(df)
        work = df.copy()
        work["_orig_pos"] = np.arange(len(work), dtype=np.int64)
        work_sorted = work.sort_values(group_cols + ["cycle"], kind="mergesort").reset_index(drop=True)

        d_obs = self._compute_causal_composite(work_sorted, fit_scale=False)

        n_rows = len(work_sorted)
        col_buffers: dict[str, np.ndarray] = {
            "saxena_a": np.empty(n_rows, dtype=np.float32),
            "saxena_log_a": np.empty(n_rows, dtype=np.float32),
            "saxena_b": np.empty(n_rows, dtype=np.float32),
            "saxena_hi": np.empty(n_rows, dtype=np.float32),
            "saxena_composite": d_obs.astype(np.float32),
            "saxena_r2": np.empty(n_rows, dtype=np.float32),
        }

        for _, grp in work_sorted.groupby(group_cols, sort=False):
            idx = grp.index.to_numpy()
            cycles = grp["cycle"].to_numpy(dtype=np.float64)
            res = self._fit_engine_prefixes(cycles, d_obs[idx])
            for k, arr in res.items():
                col_buffers[k][idx] = arr

        feat_df = pd.DataFrame(col_buffers, index=work_sorted.index)
        feat_df["_orig_pos"] = work_sorted["_orig_pos"].to_numpy()
        feat_df = feat_df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
        feat_df.index = df.index

        out = df.copy()
        for k in col_buffers:
            out[k] = feat_df[k]

        return out, list(SAXENA_FEATURE_COLS)

    def fit_transform(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Fit on train_df and transform it."""
        return self.fit(train_df).transform(train_df)

    def save(self, path: str | Path) -> Path:
        """Persist fitted SaxenaPhysicsExtractor via joblib."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "SaxenaPhysicsExtractor":
        """Load persisted SaxenaPhysicsExtractor from disk."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}.")
        return obj
