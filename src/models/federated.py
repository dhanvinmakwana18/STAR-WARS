"""Federated learning (FedAvg, McMahan et al. 2017) and conformal calibration for C-MAPSS.

Implements the simulated 3-operator split and Federated Averaging (FedAvg)
protocol specified in ``docs/aerofed_research_memo.md`` (§3–§4):

- **Client A (`client_a_high_alt_mach`)**: High-altitude / high-Mach duty cycle proxy
- **Client B (`client_b_low_alt_throttle`)**: Low-altitude / low-throttle duty cycle proxy
- **Client C (`client_c_mixed`)**: Mixed multi-regime duty cycle

Note: This is a simulated partition of a single public dataset (NASA C-MAPSS
FD002/FD004), not real cross-airline proprietary telemetry. Engines (``unit_id``)
are partitioned strictly disjointly across clients so no engine trajectory is
shared between operators.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import evaluate_predictions
from src.models.sequence import (
    build_sequence_model,
    predict_deterministic,
    train_sequence_model,
)
from src.models.tabular import compute_conformal_quantile_from_residuals

CLIENT_IDS: tuple[str, str, str] = (
    "client_a_high_alt_mach",
    "client_b_low_alt_throttle",
    "client_c_mixed",
)

CLIENT_LABELS: dict[str, str] = {
    "client_a_high_alt_mach": "Client A (High-Alt / High-Mach)",
    "client_b_low_alt_throttle": "Client B (Low-Alt / Low-Throttle)",
    "client_c_mixed": "Client C (Mixed Regime)",
}


@dataclass
class ClientSplitSummary:
    """Summary statistics for a simulated operator client split."""

    client_id: str
    client_label: str
    n_units: int
    n_cycles: int
    high_alt_mach_frac: float
    low_alt_throttle_frac: float
    mid_regime_frac: float
    mean_altitude_z: float
    mean_mach_z: float


def classify_cycle_duty_group(df: pd.DataFrame) -> np.ndarray:
    """Classify each cycle row into duty category: 0=High-Alt/Mach, 1=Low-Alt/Throttle, 2=Mid/Mixed.

    Works on either raw operational settings (Alt in 0..42 kft, Mach in 0..0.84, TRA in 60..100)
    or Z-scored operational settings from ``RegimeNormalizer``.
    """
    alt = df["op_setting_1"].to_numpy(dtype=np.float64)
    mach = df["op_setting_2"].to_numpy(dtype=np.float64)
    tra = df["op_setting_3"].to_numpy(dtype=np.float64)

    # Detect whether op_setting_1 is raw (max > 10) or standardized Z-score
    if np.max(alt) > 10.0:
        is_high = (alt >= 34.0) & (mach >= 0.78)
        is_low = (alt <= 12.0) | (tra <= 70.0)
    else:
        is_high = (alt > 0.55) & (mach > 0.55)
        is_low = (alt < -0.45) | (tra < -1.0)

    cat = np.full(len(df), 2, dtype=np.int64)
    cat[is_low] = 1
    cat[is_high] = 0
    return cat


def partition_units_into_operator_clients(
    df: pd.DataFrame,
) -> tuple[dict[str, list[int]], pd.DataFrame]:
    """Partition engine ``unit_id``s into 3 disjoint simulated operator clients.

    Engines are ranked by their empirical duty-cycle index
    ``(frac_high_alt_mach - frac_low_alt_throttle)``:
    - Top third -> Client A (High-Altitude / High-Mach dominant)
    - Bottom third -> Client B (Low-Altitude / Low-Throttle dominant)
    - Middle third -> Client C (Mixed regime)

    Returns:
        (client_units_map, unit_profile_df)
    """
    if "unit_id" not in df.columns:
        raise KeyError("DataFrame must contain 'unit_id' column.")

    duty_cat = classify_cycle_duty_group(df)
    work = pd.DataFrame(
        {
            "unit_id": df["unit_id"].to_numpy(dtype=np.int64),
            "duty_cat": duty_cat,
            "op_setting_1": df["op_setting_1"].to_numpy(dtype=np.float64),
            "op_setting_2": df["op_setting_2"].to_numpy(dtype=np.float64),
        }
    )

    records: list[dict[str, Any]] = []
    for uid, grp in work.groupby("unit_id", sort=True):
        cats = grp["duty_cat"].to_numpy()
        n = float(len(cats))
        f_high = float(np.sum(cats == 0) / n)
        f_low = float(np.sum(cats == 1) / n)
        f_mid = float(np.sum(cats == 2) / n)
        records.append(
            {
                "unit_id": int(uid),
                "n_cycles": int(len(cats)),
                "frac_high_alt_mach": f_high,
                "frac_low_alt_throttle": f_low,
                "frac_mid": f_mid,
                "duty_score": f_high - f_low,
                "mean_op1": float(grp["op_setting_1"].mean()),
                "mean_op2": float(grp["op_setting_2"].mean()),
            }
        )

    prof_df = pd.DataFrame(records).sort_values(
        ["duty_score", "unit_id"], ascending=[False, True]
    ).reset_index(drop=True)

    n_units = len(prof_df)
    if n_units < 3:
        raise ValueError(f"Need at least 3 engines to partition into 3 clients, got {n_units}.")

    n_a = n_units // 3
    n_b = n_units // 3

    units_sorted = prof_df["unit_id"].to_numpy(dtype=np.int64)
    client_a_units = sorted(units_sorted[:n_a].tolist())
    client_b_units = sorted(units_sorted[n_units - n_b :].tolist())
    client_c_units = sorted(units_sorted[n_a : n_units - n_b].tolist())

    unit_to_client: dict[int, str] = {}
    for u in client_a_units:
        unit_to_client[u] = "client_a_high_alt_mach"
    for u in client_b_units:
        unit_to_client[u] = "client_b_low_alt_throttle"
    for u in client_c_units:
        unit_to_client[u] = "client_c_mixed"

    prof_df["client_id"] = [unit_to_client[int(u)] for u in prof_df["unit_id"]]
    prof_df["client_label"] = [CLIENT_LABELS[cid] for cid in prof_df["client_id"]]
    prof_df = prof_df.sort_values("unit_id").reset_index(drop=True)

    client_map = {
        "client_a_high_alt_mach": client_a_units,
        "client_b_low_alt_throttle": client_b_units,
        "client_c_mixed": client_c_units,
    }
    return client_map, prof_df


def build_client_sequence_datasets(
    df: pd.DataFrame,
    X_all: np.ndarray,
    y_all: np.ndarray,
    meta_df: pd.DataFrame,
    client_units: dict[str, list[int]],
    apply_regime_bias: bool = True,
    dominant_keep_prob: float = 1.0,
    non_dominant_keep_prob: float = 0.25,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """Split sequence windows (X_all, y_all) across the 3 simulated operator clients.

    Each client receives strictly windows from its own disjoint ``unit_id``s.
    When ``apply_regime_bias=True`` (for training windows), Client A retains 100% of
    its high-altitude/high-Mach windows and subsamples non-dominant regimes at
    ``non_dominant_keep_prob`` (and symmetrically for Client B on low-altitude/low-throttle
    windows), modeling operators whose logged flight telemetry is concentrated in their
    primary mission envelope. For test/evaluation splits, pass ``apply_regime_bias=False``
    so all engines are evaluated without subsampling.
    """
    rng = np.random.default_rng(seed)

    # Map (unit_id, cycle) -> duty_cat (0=High, 1=Low, 2=Mid)
    duty_cats = classify_cycle_duty_group(df)
    key_to_cat: dict[tuple[int, int], int] = {
        (int(u), int(c)): int(cat)
        for u, c, cat in zip(df["unit_id"].to_numpy(), df["cycle"].to_numpy(), duty_cats)
    }
    win_cats = np.array(
        [
            key_to_cat.get((int(u), int(c)), 2)
            for u, c in zip(meta_df["unit_id"].to_numpy(), meta_df["cycle"].to_numpy())
        ],
        dtype=np.int64,
    )

    out: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    win_units = meta_df["unit_id"].to_numpy(dtype=np.int64)

    for cid in CLIENT_IDS:
        u_set = set(client_units[cid])
        u_mask = np.isin(win_units, list(u_set))

        if apply_regime_bias and cid in ("client_a_high_alt_mach", "client_b_low_alt_throttle"):
            dom_cat = 0 if cid == "client_a_high_alt_mach" else 1
            is_dom = win_cats == dom_cat
            probs = np.where(is_dom, dominant_keep_prob, non_dominant_keep_prob)
            draws = rng.random(len(win_units)) < probs
            sel_mask = u_mask & draws
            if int(np.sum(sel_mask)) < 16:
                sel_mask = u_mask
        else:
            sel_mask = u_mask

        out[cid] = (
            X_all[sel_mask].copy(),
            y_all[sel_mask].copy(),
            meta_df.loc[sel_mask].reset_index(drop=True),
        )

    return out


def fedavg_aggregate_state_dicts(
    client_states: list[dict[str, torch.Tensor]],
    client_weights: list[float] | np.ndarray,
) -> dict[str, torch.Tensor]:
    """Compute sample-weighted Federated Averaging (FedAvg, McMahan et al. 2017) over state_dicts."""
    if not client_states:
        raise ValueError("client_states must be non-empty.")
    weights = np.asarray(client_weights, dtype=np.float64)
    if len(client_states) != weights.size or np.sum(weights) <= 0.0:
        raise ValueError("Invalid client_weights for fedavg_aggregate_state_dicts.")
    w_norm = weights / np.sum(weights)

    agg_state: dict[str, torch.Tensor] = {}
    keys = list(client_states[0].keys())
    for k in keys:
        first_tensor = client_states[0][k]
        if first_tensor.is_floating_point():
            acc = torch.zeros_like(first_tensor, dtype=torch.float64)
            for state, w in zip(client_states, w_norm):
                acc += state[k].detach().to(dtype=torch.float64, device=first_tensor.device) * float(w)
            agg_state[k] = acc.to(dtype=first_tensor.dtype)
        else:
            # Integer buffers (e.g. BatchNorm num_batches_tracked): take max
            agg_state[k] = first_tensor.detach().clone()

    return agg_state


def _train_local_client_epochs(
    local_model: nn.Module,
    X_client: np.ndarray,
    y_client: np.ndarray,
    local_epochs: int = 2,
    batch_size: int = 128,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    prox_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
    device: str = "cpu",
) -> tuple[dict[str, torch.Tensor], float]:
    """Run ``local_epochs`` of local SGD/AdamW on a single operator client's dataset."""
    local_model.to(device)
    local_model.train()

    ds = TensorDataset(
        torch.from_numpy(np.asarray(X_client, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_client, dtype=np.float32).ravel()),
    )
    drop_last = len(ds) > batch_size and (len(ds) % batch_size == 1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    last_epoch_sq_err = 0.0
    last_epoch_n = 0

    for _ in range(max(1, int(local_epochs))):
        sq_err = 0.0
        n_tot = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if xb.shape[0] == 1:
                for m in local_model.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()

            optimizer.zero_grad(set_to_none=True)
            preds = local_model(xb)
            loss = criterion(preds, yb)

            if prox_mu > 0.0 and global_params is not None:
                prox_term = torch.tensor(0.0, device=device)
                for p_loc, p_glob in zip(local_model.parameters(), global_params):
                    prox_term = prox_term + torch.sum((p_loc - p_glob) ** 2)
                loss = loss + 0.5 * float(prox_mu) * prox_term

            loss.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                sq_err += float(torch.sum((preds - yb) ** 2).item())
                n_tot += int(yb.numel())

        last_epoch_sq_err = sq_err
        last_epoch_n = n_tot

    train_rmse = float(np.sqrt(last_epoch_sq_err / max(1, last_epoch_n)))
    state_copy = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
    return state_copy, train_rmse


def train_fedavg_model(
    arch: str,
    n_features: int,
    seq_len: int,
    client_train_data: dict[str, tuple[np.ndarray, np.ndarray]],
    client_val_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    rounds: int = 20,
    local_epochs: int = 2,
    batch_size: int = 128,
    lr: float = 1.2e-3,
    lr_decay: float = 0.96,
    weight_decay: float = 1e-4,
    prox_mu: float = 0.0,
    model_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train a global sequence model across simulated operator clients via FedAvg.

    Raw client trajectories never leave each client: only state_dict tensors and
    sample counts ``n_k`` are aggregated at the end of each communication round.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    kw = dict(model_kwargs) if model_kwargs is not None else {}
    global_model = build_sequence_model(arch, n_features=n_features, seq_len=seq_len, **kw)
    global_model.to(device)

    active_clients = [cid for cid in CLIENT_IDS if cid in client_train_data and len(client_train_data[cid][0]) > 0]
    if not active_clients:
        raise ValueError("No non-empty clients provided in client_train_data.")

    client_sizes = [float(len(client_train_data[cid][0])) for cid in active_clients]
    best_val_rmse = float("inf")
    best_state = copy.deepcopy(global_model.state_dict())
    history: list[dict[str, Any]] = []

    for r in range(1, int(rounds) + 1):
        round_lr = float(lr * (lr_decay ** (r - 1)))
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        global_params = [p.detach().clone().to(device) for p in global_model.parameters()]

        local_states: list[dict[str, torch.Tensor]] = []
        local_rmses: dict[str, float] = {}

        for cid in active_clients:
            X_c, y_c = client_train_data[cid]
            local_net = build_sequence_model(arch, n_features=n_features, seq_len=seq_len, **kw)
            local_net.load_state_dict(global_state)
            c_state, c_rmse = _train_local_client_epochs(
                local_net,
                X_c,
                y_c,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=round_lr,
                weight_decay=weight_decay,
                prox_mu=prox_mu,
                global_params=global_params if prox_mu > 0.0 else None,
                device=device,
            )
            local_states.append(c_state)
            local_rmses[cid] = c_rmse

        agg_state = fedavg_aggregate_state_dicts(local_states, client_sizes)
        global_model.load_state_dict(agg_state)
        global_model.eval()

        row: dict[str, Any] = {
            "round": r,
            "lr": round_lr,
            "local_epochs": int(local_epochs),
            "mean_client_train_rmse": float(np.mean(list(local_rmses.values()))),
        }
        for cid, tr_r in local_rmses.items():
            row[f"train_rmse_{cid}"] = tr_r

        if client_val_data:
            all_val_y: list[np.ndarray] = []
            all_val_p: list[np.ndarray] = []
            for cid, (X_v, y_v) in client_val_data.items():
                if len(X_v) == 0:
                    continue
                preds_v = predict_deterministic(
                    global_model, X_v, device=device, clip_lower=0.0, clip_upper=125.0
                )
                ev = evaluate_predictions(y_v, preds_v, clip_cap=125, clip_pred=True)
                row[f"val_rmse_{cid}"] = ev["rmse"]
                row[f"val_nasa_{cid}"] = ev["nasa_score"]
                all_val_y.append(np.asarray(y_v, dtype=np.float64))
                all_val_p.append(preds_v)

            if all_val_y:
                y_cat = np.concatenate(all_val_y)
                p_cat = np.concatenate(all_val_p)
                ev_all = evaluate_predictions(y_cat, p_cat, clip_cap=125, clip_pred=True)
                val_rmse_all = float(ev_all["rmse"])
                row["val_rmse_pooled"] = val_rmse_all
                row["val_nasa_pooled"] = float(ev_all["nasa_score"])
                if val_rmse_all < best_val_rmse:
                    best_val_rmse = val_rmse_all
                    best_state = copy.deepcopy(global_model.state_dict())

        history.append(row)

    global_model.load_state_dict(best_state)
    global_model.eval()
    return global_model, history


def calibrate_and_evaluate_conformal(
    model: nn.Module,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 0.10,
    clip_cap: float = 125.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Calibrate split-conformal intervals (Vovk et al.) on (X_cal, y_cal) and audit on (X_test, y_test)."""
    pred_cal = predict_deterministic(model, X_cal, device=device, clip_lower=0.0, clip_upper=clip_cap)
    yt_cal = np.minimum(np.asarray(y_cal, dtype=np.float64).ravel(), clip_cap)
    abs_res_cal = np.abs(yt_cal - pred_cal)
    q_hat = compute_conformal_quantile_from_residuals(abs_res_cal, alpha=alpha)

    pred_test = predict_deterministic(model, X_test, device=device, clip_lower=0.0, clip_upper=clip_cap)
    yt_test = np.minimum(np.asarray(y_test, dtype=np.float64).ravel(), clip_cap)

    ci_low = np.maximum(0.0, pred_test - q_hat)
    ci_high = np.minimum(clip_cap, pred_test + q_hat)
    ci_high_unclipped = pred_test + q_hat

    covered_clipped = (yt_test >= ci_low) & (yt_test <= ci_high)
    covered_unclipped = (yt_test >= ci_low) & (yt_test <= ci_high_unclipped)

    ev = evaluate_predictions(yt_test, pred_test, clip_cap=int(clip_cap), clip_pred=True)
    return {
        "nominal_coverage": float(1.0 - alpha),
        "conformal_q_cycles": float(q_hat),
        "empirical_coverage": float(np.mean(covered_unclipped)),
        "empirical_coverage_clipped_band": float(np.mean(covered_clipped)),
        "mean_interval_width_cycles": float(np.mean(ci_high_unclipped - ci_low)),
        "mean_interval_width_clipped": float(np.mean(ci_high - ci_low)),
        "rmse": float(ev["rmse"]),
        "mae": float(ev["mae"]),
        "nasa_score": float(ev["nasa_score"]),
        "n_test_engines": int(len(yt_test)),
        "pred_test": pred_test,
        "ci_low": ci_low,
        "ci_high": ci_high_unclipped,
    }
