"""PyTorch sequence models (Deep 1D-CNN & BiLSTM+Attention) with MC Dropout & conformal calibration."""

from __future__ import annotations

import copy
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.tabular import compute_conformal_quantile_from_residuals


class _ResidualConvBlock(nn.Module):
    """Residual 1D convolutional block with BatchNorm, GELU, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act2(out + residual)


class _TemporalAttentionPool(nn.Module):
    """Additive attention pooling over temporal steps concatenated with the final step."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:
        # seq_feats: (B, L, D)
        weights = torch.softmax(self.score(seq_feats), dim=1)  # (B, L, 1)
        attn_ctx = torch.sum(weights * seq_feats, dim=1)  # (B, D)
        last_step = seq_feats[:, -1, :]  # (B, D)
        return torch.cat([attn_ctx, last_step], dim=-1)  # (B, 2*D)


class Deep1DCNNRegressor(nn.Module):
    """Multi-scale 1D-CNN with temporal bin pooling and dilated residual context."""

    def __init__(
        self,
        n_features: int,
        seq_len: int = 30,
        channels: tuple[int, ...] | list[int] = (32, 48, 64),
        kernel_sizes: tuple[int, ...] | list[int] = (7, 5, 3),
        dropout: float = 0.30,
        fc_hidden: int = 64,
        n_temporal_bins: int = 6,
    ) -> None:
        super().__init__()
        channels_list = [int(c) for c in channels]
        kernels_list = [int(k) for k in kernel_sizes]
        if len(channels_list) != len(kernels_list) or len(channels_list) == 0:
            raise ValueError("channels and kernel_sizes must be non-empty and equal length.")

        self.init_kwargs: dict[str, Any] = {
            "n_features": int(n_features),
            "seq_len": int(seq_len),
            "channels": channels_list,
            "kernel_sizes": kernels_list,
            "dropout": float(dropout),
            "fc_hidden": int(fc_hidden),
            "n_temporal_bins": int(n_temporal_bins),
        }

        layers: list[nn.Module] = []
        in_c = int(n_features)
        for idx, (out_c, k) in enumerate(zip(channels_list, kernels_list)):
            layers.extend(
                [
                    nn.Conv1d(in_c, out_c, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(out_c),
                    nn.GELU(),
                ]
            )
            if idx < len(channels_list) - 1:
                layers.append(nn.Dropout(dropout * 0.4))
            in_c = out_c
        self.conv_stack = nn.Sequential(*layers)

        self.dilated_ctx = nn.Sequential(
            nn.Conv1d(in_c, in_c, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(in_c),
            nn.GELU(),
        )
        self.tpool = nn.AdaptiveAvgPool1d(int(n_temporal_bins))
        feat_dim = (int(n_temporal_bins) + 1) * in_c
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        self.conformal_q_: float = 0.0
        self.conformal_scale_: float | None = None
        self.conformal_floor_: float = 1.0
        self.val_nonconformity_: np.ndarray | None = None
        self.val_abs_residuals_: np.ndarray | None = None
        self.feature_cols_: list[str] | None = None
        self.window_size_: int = int(seq_len)
        self.metadata_: dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        h = x.transpose(1, 2)  # (B, n_features, seq_len)
        h = self.conv_stack(h)
        h = h + self.dilated_ctx(h)
        bins = self.tpool(h).flatten(1)
        last_step = h[:, :, -1]
        return self.head(torch.cat([bins, last_step], dim=1)).squeeze(-1)


class BiLSTMRegressor(nn.Module):
    """Bidirectional LSTM with temporal attention pooling for RUL prediction."""

    def __init__(
        self,
        n_features: int,
        seq_len: int = 30,
        hidden_size: int = 48,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.25,
        fc_hidden: int = 48,
    ) -> None:
        super().__init__()
        self.init_kwargs: dict[str, Any] = {
            "n_features": int(n_features),
            "seq_len": int(seq_len),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "bidirectional": bool(bidirectional),
            "dropout": float(dropout),
            "fc_hidden": int(fc_hidden),
        }

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(lstm_dim)
        self.pool = _TemporalAttentionPool(lstm_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_dim, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        self.conformal_q_: float = 0.0
        self.conformal_scale_: float | None = None
        self.conformal_floor_: float = 1.0
        self.val_nonconformity_: np.ndarray | None = None
        self.val_abs_residuals_: np.ndarray | None = None
        self.feature_cols_: list[str] | None = None
        self.window_size_: int = int(seq_len)
        self.metadata_: dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        h = self.input_proj(x)
        lstm_out, _ = self.lstm(h)
        lstm_out = self.norm(lstm_out)
        pooled = self.pool(lstm_out)
        return self.head(pooled).squeeze(-1)


LSTMRegressor = BiLSTMRegressor

_ARCH_REGISTRY: dict[str, type[nn.Module]] = {
    "Deep1DCNNRegressor": Deep1DCNNRegressor,
    "1d_cnn": Deep1DCNNRegressor,
    "cnn": Deep1DCNNRegressor,
    "BiLSTMRegressor": BiLSTMRegressor,
    "LSTMRegressor": BiLSTMRegressor,
    "bilstm": BiLSTMRegressor,
    "lstm": BiLSTMRegressor,
}


def build_sequence_model(arch: str, n_features: int, **kwargs: Any) -> nn.Module:
    """Instantiate a sequence model by architecture name."""
    if arch not in _ARCH_REGISTRY:
        raise ValueError(f"Unknown sequence architecture '{arch}'. Valid: {list(_ARCH_REGISTRY)}")
    return _ARCH_REGISTRY[arch](n_features=n_features, **kwargs)


def enable_mc_dropout(model: nn.Module) -> None:
    """Activate Dropout (and LSTM inter-layer dropout) while keeping normalization in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.AlphaDropout, nn.LSTM)):
            module.train()


def _unwrap_module(model: nn.Module | SequenceRULModel) -> nn.Module:
    if isinstance(model, SequenceRULModel):
        if model.model is None:
            raise RuntimeError("SequenceRULModel has not been initialized or fitted yet.")
        return model.model
    return model


@torch.no_grad()
def predict_deterministic(
    model: nn.Module | SequenceRULModel,
    X: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
    clip_lower: float | None = 0.0,
    clip_upper: float | None = None,
) -> np.ndarray:
    """Run deterministic inference on 3D windows (N, seq_len, n_features)."""
    net = _unwrap_module(model)
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 3:
        raise ValueError(f"Expected 3D sequence input (N, T, F), got shape {X_arr.shape}")

    was_training = net.training
    net.to(device)
    net.eval()
    preds: list[np.ndarray] = []
    try:
        for start in range(0, X_arr.shape[0], batch_size):
            xb = torch.from_numpy(X_arr[start : start + batch_size]).to(device)
            out = net(xb).detach().cpu().numpy().astype(np.float64)
            preds.append(out)
    finally:
        net.train(was_training)

    out_all = np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float64)
    if clip_lower is not None or clip_upper is not None:
        out_all = np.clip(out_all, clip_lower, clip_upper)
    return out_all


@torch.no_grad()
def _mc_forward_raw(
    net: nn.Module,
    X_arr: np.ndarray,
    n_mc: int = 30,
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Perform n_mc stochastic forward passes and return unclipped (mean, std)."""
    if n_mc <= 1:
        mean_pred = predict_deterministic(
            net, X_arr, batch_size=batch_size, device=device, clip_lower=None, clip_upper=None
        )
        return mean_pred, np.zeros_like(mean_pred, dtype=np.float64)

    was_training = net.training
    net.to(device)
    enable_mc_dropout(net)

    n_samples = X_arr.shape[0]
    sum_preds = np.zeros((n_samples,), dtype=np.float64)
    sum_sq_preds = np.zeros((n_samples,), dtype=np.float64)

    try:
        for _ in range(n_mc):
            pass_preds: list[np.ndarray] = []
            for start in range(0, n_samples, batch_size):
                xb = torch.from_numpy(X_arr[start : start + batch_size]).to(device)
                out = net(xb).detach().cpu().numpy().astype(np.float64)
                pass_preds.append(out)
            vec = np.concatenate(pass_preds, axis=0) if pass_preds else np.empty((0,), dtype=np.float64)
            sum_preds += vec
            sum_sq_preds += vec * vec
    finally:
        net.train(was_training)

    mean_pred = sum_preds / float(n_mc)
    var_pred = np.maximum(0.0, (sum_sq_preds / float(n_mc)) - (mean_pred * mean_pred))
    std_pred = np.sqrt(var_pred)
    return mean_pred, std_pred


def calibrate_sequence_conformal(
    model: nn.Module | SequenceRULModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_mc: int = 30,
    alpha: float = 0.10,
    batch_size: int = 1024,
    device: str = "cpu",
    clip_lower: float | None = 0.0,
    clip_upper: float | None = None,
    max_cal_samples: int | None = 3000,
) -> dict[str, float]:
    """Calibrate split-conformal residual bands around MC Dropout predictions on validation data."""
    net = _unwrap_module(model)
    X_arr = np.asarray(X_val, dtype=np.float32)
    y_true = np.asarray(y_val, dtype=np.float64).ravel()
    if X_arr.shape[0] != y_true.shape[0]:
        raise ValueError(f"X_val rows ({X_arr.shape[0]}) != y_val length ({y_true.shape[0]})")

    if max_cal_samples is not None and X_arr.shape[0] > max_cal_samples:
        idx = np.linspace(0, X_arr.shape[0] - 1, int(max_cal_samples), dtype=np.int64)
        X_arr = X_arr[idx]
        y_true = y_true[idx]

    val_mean, val_std = _mc_forward_raw(
        net, X_arr, n_mc=n_mc, batch_size=batch_size, device=device
    )
    if clip_lower is not None or clip_upper is not None:
        val_mean = np.clip(val_mean, clip_lower, clip_upper)

    abs_res = np.abs(y_true - val_mean)
    floor = max(1.0, float(np.median(val_std))) if val_std.size > 0 else 1.0
    nonconf = abs_res / (val_std + floor)

    q_add = compute_conformal_quantile_from_residuals(abs_res, alpha=alpha)
    q_scale = compute_conformal_quantile_from_residuals(nonconf, alpha=alpha)

    net.conformal_q_ = q_add
    net.conformal_scale_ = q_scale
    net.conformal_floor_ = floor
    net.val_nonconformity_ = nonconf
    net.val_abs_residuals_ = abs_res

    half_width = q_scale * (val_std + floor)
    ci_low = np.maximum(0.0, val_mean - half_width) if clip_lower is not None else val_mean - half_width
    ci_high = val_mean + half_width
    coverage = float(np.mean((y_true >= ci_low) & (y_true <= ci_high)))

    return {
        "conformal_q": q_add,
        "conformal_scale": q_scale,
        "conformal_floor": floor,
        "val_mc_rmse": float(np.sqrt(np.mean((y_true - val_mean) ** 2))),
        "val_mean_mc_std": float(np.mean(val_std)),
        "val_empirical_coverage": coverage,
        "val_mean_interval_width": float(np.mean(ci_high - ci_low)),
    }


def predict_with_uncertainty(
    model: nn.Module | SequenceRULModel,
    X: np.ndarray,
    n_mc: int = 30,
    alpha: float = 0.10,
    batch_size: int = 512,
    device: str = "cpu",
    clip_lower: float | None = 0.0,
    clip_upper: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict RUL via MC Dropout + conformal residual calibration.

    Returns:
        (pred_mean, pred_std, ci_low, ci_high) with ci_low <= pred_mean <= ci_high.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    net = _unwrap_module(model)
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 3:
        raise ValueError(f"Expected 3D sequence input (N, T, F), got shape {X_arr.shape}")

    pred_mean, pred_std = _mc_forward_raw(
        net, X_arr, n_mc=n_mc, batch_size=batch_size, device=device
    )
    if clip_lower is not None or clip_upper is not None:
        pred_mean = np.clip(pred_mean, clip_lower, clip_upper)

    val_nonconf = getattr(net, "val_nonconformity_", None)
    val_abs_res = getattr(net, "val_abs_residuals_", None)
    floor = float(getattr(net, "conformal_floor_", 1.0))

    if n_mc > 1 and val_nonconf is not None and len(val_nonconf) > 0:
        q_scale = compute_conformal_quantile_from_residuals(val_nonconf, alpha=alpha)
        half_width = q_scale * (pred_std + floor)
    elif val_abs_res is not None and len(val_abs_res) > 0:
        q_add = compute_conformal_quantile_from_residuals(val_abs_res, alpha=alpha)
        half_width = np.full_like(pred_mean, q_add, dtype=np.float64)
    elif getattr(net, "conformal_q_", 0.0) > 0.0:
        half_width = np.full_like(pred_mean, float(net.conformal_q_), dtype=np.float64)
    else:
        z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
        half_width = z * pred_std

    ci_low = pred_mean - half_width
    if clip_lower is not None:
        ci_low = np.maximum(clip_lower, ci_low)
    ci_low = np.minimum(ci_low, pred_mean)
    ci_high = np.maximum(pred_mean, pred_mean + half_width)
    return pred_mean, pred_std, ci_low, ci_high


def save_sequence_checkpoint(
    model: nn.Module | SequenceRULModel,
    path: str | Path,
    feature_cols: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Save sequence model weights and conformal calibration in a weights_only-safe .pt file."""
    net = _unwrap_module(model)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fcols = feature_cols if feature_cols is not None else getattr(net, "feature_cols_", None)
    meta = dict(getattr(net, "metadata_", {}))
    if extra_meta:
        meta.update(extra_meta)

    val_nonconf = getattr(net, "val_nonconformity_", None)
    val_abs_res = getattr(net, "val_abs_residuals_", None)

    payload: dict[str, Any] = {
        "arch": net.__class__.__name__,
        "init_kwargs": dict(getattr(net, "init_kwargs", {})),
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "conformal_q": float(getattr(net, "conformal_q_", 0.0)),
        "conformal_scale": (
            float(net.conformal_scale_)
            if getattr(net, "conformal_scale_", None) is not None
            else None
        ),
        "conformal_floor": float(getattr(net, "conformal_floor_", 1.0)),
        "val_nonconformity": (
            torch.from_numpy(np.asarray(val_nonconf, dtype=np.float32))
            if val_nonconf is not None
            else None
        ),
        "val_abs_residuals": (
            torch.from_numpy(np.asarray(val_abs_res, dtype=np.float32))
            if val_abs_res is not None
            else None
        ),
        "feature_cols": list(fcols) if fcols is not None else None,
        "window_size": int(getattr(net, "window_size_", 30)),
        "metadata": meta,
    }
    torch.save(payload, out_path)
    return out_path


def load_sequence_checkpoint(path: str | Path, map_location: str = "cpu") -> nn.Module:
    """Load a sequence model checkpoint saved via save_sequence_checkpoint."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    arch = ckpt["arch"]
    init_kwargs = ckpt["init_kwargs"]
    net = build_sequence_model(arch, **init_kwargs)
    net.load_state_dict(ckpt["state_dict"])
    net.conformal_q_ = float(ckpt.get("conformal_q", 0.0))
    net.conformal_scale_ = ckpt.get("conformal_scale", None)
    net.conformal_floor_ = float(ckpt.get("conformal_floor", 1.0))

    val_nc = ckpt.get("val_nonconformity", None)
    net.val_nonconformity_ = val_nc.cpu().numpy().astype(np.float64) if val_nc is not None else None
    val_ar = ckpt.get("val_abs_residuals", None)
    net.val_abs_residuals_ = val_ar.cpu().numpy().astype(np.float64) if val_ar is not None else None

    net.feature_cols_ = ckpt.get("feature_cols", None)
    net.window_size_ = int(ckpt.get("window_size", init_kwargs.get("seq_len", 30)))
    net.metadata_ = dict(ckpt.get("metadata", {}))
    net.to(map_location)
    net.eval()
    return net


def train_sequence_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 35,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    grad_clip: float = 1.0,
    scheduler_type: str = "cosine",
    loss_type: str = "mse",
    n_mc: int = 30,
    alpha: float = 0.10,
    checkpoint_path: str | Path | None = None,
    feature_cols: list[str] | None = None,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a PyTorch sequence model with AdamW, LR scheduling, clipping, and early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.float32).ravel()
    if X_tr.ndim != 3:
        raise ValueError(f"Expected 3D X_train (N, T, F), got shape {X_tr.shape}")
    if X_tr.shape[0] != y_tr.shape[0]:
        raise ValueError(f"X_train rows ({X_tr.shape[0]}) != y_train length ({y_tr.shape[0]})")

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_va = np.asarray(X_val, dtype=np.float32)
        y_va = np.asarray(y_val, dtype=np.float32).ravel()
    else:
        X_va, y_va = None, None

    model.to(device)
    if feature_cols is not None:
        model.feature_cols_ = list(feature_cols)
    model.window_size_ = int(X_tr.shape[1])

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    drop_last = len(train_ds) > batch_size and (len(train_ds) % batch_size == 1)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )

    criterion: nn.Module
    if loss_type.lower() == "huber":
        criterion = nn.HuberLoss(delta=15.0)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type.lower() == "plateau" and has_val:
        scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, patience // 2)
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=1e-5
        )

    best_val_rmse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    history: dict[str, Any] = {"train_rmse": [], "val_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_sq_err = 0.0
        total_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if xb.shape[0] == 1:
                # BatchNorm1d requires >1 sample in train mode.
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                total_sq_err += float(torch.sum((preds - yb) ** 2).item())
                total_n += int(yb.numel())

        train_rmse = float(np.sqrt(total_sq_err / max(1, total_n)))
        history["train_rmse"].append(train_rmse)

        if has_val and X_va is not None and y_va is not None:
            val_preds = predict_deterministic(
                model, X_va, batch_size=max(1024, batch_size), device=device,
                clip_lower=None, clip_upper=None
            )
            val_rmse = float(np.sqrt(np.mean((y_va.astype(np.float64) - val_preds) ** 2)))
            history["val_rmse"].append(val_rmse)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_rmse)
            else:
                scheduler.step()

            if val_rmse < best_val_rmse - 1e-5:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                if checkpoint_path is not None:
                    save_sequence_checkpoint(model, checkpoint_path, feature_cols=feature_cols)
            else:
                patience_ctr += 1

            if verbose and (epoch == 1 or epoch % 5 == 0 or patience_ctr == 0):
                print(
                    f"  Epoch {epoch:02d}/{epochs} | train_rmse={train_rmse:.3f} | "
                    f"val_rmse={val_rmse:.3f} (best={best_val_rmse:.3f} @ {best_epoch})",
                    flush=True,
                )

            if patience_ctr >= patience:
                break
        else:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            if train_rmse < best_val_rmse:
                best_val_rmse = train_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    if has_val and X_va is not None and y_va is not None:
        cal_stats = calibrate_sequence_conformal(
            model, X_va, y_va, n_mc=n_mc, alpha=alpha,
            batch_size=max(1024, batch_size), device=device
        )
    else:
        cal_stats = calibrate_sequence_conformal(
            model, X_tr, y_tr, n_mc=n_mc, alpha=alpha,
            batch_size=max(1024, batch_size), device=device
        )

    history["best_epoch"] = best_epoch
    history["best_val_rmse"] = best_val_rmse
    history["calibration"] = cal_stats

    if checkpoint_path is not None:
        save_sequence_checkpoint(
            model,
            checkpoint_path,
            feature_cols=feature_cols,
            extra_meta={"best_epoch": best_epoch, "best_val_rmse": best_val_rmse, **cal_stats},
        )

    return model, history


class SequenceRULModel:
    """High-level scikit-learn-style wrapper around PyTorch sequence regressors."""

    def __init__(
        self,
        arch: str = "1d_cnn",
        n_features: int | None = None,
        seq_len: int = 30,
        model_kwargs: dict[str, Any] | None = None,
        epochs: int = 35,
        batch_size: int = 256,
        lr: float = 1.5e-3,
        weight_decay: float = 1e-4,
        patience: int = 7,
        n_mc: int = 30,
        alpha: float = 0.10,
        seed: int = 42,
        device: str = "cpu",
        clip_lower: float | None = 0.0,
        clip_upper: float | None = None,
    ) -> None:
        self.arch = arch
        self.n_features = n_features
        self.seq_len = int(seq_len)
        self.model_kwargs = dict(model_kwargs) if model_kwargs is not None else {}
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.n_mc = int(n_mc)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.device = device
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.model: nn.Module | None = None
        self.history_: dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_cols: list[str] | None = None,
        checkpoint_path: str | Path | None = None,
        verbose: bool = False,
    ) -> SequenceRULModel:
        X_arr = np.asarray(X_train, dtype=np.float32)
        in_features = self.n_features or int(X_arr.shape[2])
        seq_len = int(X_arr.shape[1])
        self.n_features = in_features
        self.seq_len = seq_len
        self.model = build_sequence_model(
            self.arch, n_features=in_features, seq_len=seq_len, **self.model_kwargs
        )
        self.model, self.history_ = train_sequence_model(
            self.model,
            X_arr,
            y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            n_mc=self.n_mc,
            alpha=self.alpha,
            checkpoint_path=checkpoint_path,
            feature_cols=feature_cols,
            seed=self.seed,
            device=self.device,
            verbose=verbose,
        )
        return self

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SequenceRULModel is not fitted yet.")
        return predict_deterministic(
            self.model,
            X,
            batch_size=batch_size,
            device=self.device,
            clip_lower=self.clip_lower,
            clip_upper=self.clip_upper,
        )

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_mc: int | None = None,
        alpha: float | None = None,
        batch_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("SequenceRULModel is not fitted yet.")
        return predict_with_uncertainty(
            self.model,
            X,
            n_mc=self.n_mc if n_mc is None else int(n_mc),
            alpha=self.alpha if alpha is None else float(alpha),
            batch_size=batch_size,
            device=self.device,
            clip_lower=self.clip_lower,
            clip_upper=self.clip_upper,
        )

    def save(
        self,
        path: str | Path,
        feature_cols: list[str] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        if self.model is None:
            raise RuntimeError("SequenceRULModel is not fitted yet.")
        return save_sequence_checkpoint(
            self.model, path, feature_cols=feature_cols, extra_meta=extra_meta
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> SequenceRULModel:
        net = load_sequence_checkpoint(path, map_location=map_location)
        wrapper = cls(
            arch=net.__class__.__name__,
            n_features=int(net.init_kwargs["n_features"]),
            seq_len=int(getattr(net, "window_size_", net.init_kwargs.get("seq_len", 30))),
            device=map_location,
        )
        wrapper.model = net
        return wrapper
