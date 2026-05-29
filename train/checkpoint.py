"""完整的 checkpoint 工具:保存/加载 model + optimizer + scaler + 步数 + best。

Week 4 引入。两类文件:
- best.pt    : val loss 最低时保存,推理用
- latest.pt  : 周期性滚动保存,断点续训用
"""
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    best_val: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'step': step,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state':    scaler.state_dict() if scaler is not None else None,
        'best_val':        best_val,
        'config':          config,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = 'cpu',
) -> dict[str, Any]:
    """加载 checkpoint。返回原始 payload 以便取 step / best_val / config。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scaler is not None and ckpt.get('scaler_state') is not None:
        scaler.load_state_dict(ckpt['scaler_state'])
    return ckpt
