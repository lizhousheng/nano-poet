"""训练通用工具:加载数据、采样 batch、估计 loss。"""
import time
from typing import Callable

import torch
from torch.amp import autocast

from configs.config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    EVAL_INTERVAL,
    EVAL_ITERS,
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
)


def _try_directml() -> 'torch.device | None':
    """装了 torch-directml 且有可用设备就返回它的 device 对象,否则 None。

    DirectML 是 Windows 上 AMD 卡 / 老 Intel 核显跑 PyTorch 的唯一通路
    (CUDA 是 NVIDIA 专属、ROCm 仅 Linux、XPU 仅较新 Intel)。
    它注册成 'privateuseone' 后端,返回的是 device 对象而非字符串。
    """
    try:
        import torch_directml
    except ImportError:
        return None
    try:
        if torch_directml.device_count() > 0:
            return torch_directml.device()
    except Exception:
        return None
    return None


def get_device() -> 'str | torch.device':
    """挑当前环境能用的最佳设备。返回值可直接喂给 .to();autocast 请走 autocast_ctx()。

    探测顺序:CUDA -> XPU -> MPS -> DirectML -> CPU。注意:
    - CUDA 只认 NVIDIA(以及 Linux 上 ROCm 版 torch 的 AMD 卡)。
    - Intel 独显/新核显走 XPU,但需要装 xpu 版 torch(cu128 版探测不到)。
    - Apple 芯片走 MPS。
    - Windows 上的 AMD 卡 / 老 Intel 核显走 DirectML(需 `pip install torch-directml`),
      返回的是 device 对象(type='privateuseone'),不支持 AMP,只能 fp32。
    """
    if torch.cuda.is_available():                              # NVIDIA,或 Linux ROCm-AMD
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():      # Intel GPU(需 xpu 版 torch)
        return 'xpu'
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():                  # Apple Silicon
        return 'mps'
    dml = _try_directml()                                       # AMD / 老 Intel 核显(Windows)
    if dml is not None:
        return dml
    return 'cpu'


def device_type(device: 'str | torch.device') -> str:
    """取 autocast 用的 device_type 字符串。device 可能是字符串或 torch.device 对象。"""
    return device if isinstance(device, str) else device.type


def amp_enabled(device: 'str | torch.device') -> bool:
    """混合精度(autocast fp16 + GradScaler)只在 CUDA 上启用。

    其他设备(xpu / mps / cpu / DirectML)退回 fp32:GradScaler(enabled=False) 变直通,
    训练照常跑,只是不省显存、不加速 —— 保证 train_v05~v10 在任何设备上都能正确跑完,
    而不是在非 CUDA 上崩或静默不更新。
    """
    return device_type(device) == 'cuda'


def autocast_ctx(device: 'str | torch.device', dtype: torch.dtype = torch.float16):
    """统一的 autocast 上下文:CUDA 上启用 fp16,其余设备 no-op(fp32)。

    关键:不能把 'privateuseone'(DirectML)/'mps' 等不支持 AMP 的 device_type 直接传给
    autocast —— 即使 enabled=False 也会 AssertionError。所以禁用时统一用 'cpu' 占位。
    """
    on = amp_enabled(device)
    return autocast(device_type='cuda' if on else 'cpu', dtype=dtype, enabled=on)


def load_data(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """加载 train/val 张量并搬到 device 上。"""
    train = torch.load(TRAIN_DATA_FILE, weights_only=True).to(device)
    val   = torch.load(VAL_DATA_FILE,   weights_only=True).to(device)
    return train, val


def make_get_batch(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
) -> Callable[[str], tuple[torch.Tensor, torch.Tensor]]:
    """构造闭包形式的 batch sampler。数据已经在 device 上,采出来的也在。"""

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i : i + block_size]     for i in ix])
        y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
        return x, y

    return get_batch


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    get_batch: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    eval_iters: int = EVAL_ITERS,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    get_batch: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    num_steps: int,
    eval_interval: int = EVAL_INTERVAL,
    extra_log: Callable[[dict[str, float]], str] | None = None,
) -> None:
    """通用训练循环。extra_log 可选,接收当前 losses 返回额外标记字符串。"""
    t0 = time.time()
    for step in range(num_steps):
        x, y = get_batch('train')
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == num_steps - 1:
            losses = estimate_loss(model, get_batch)
            tag = '  ' + extra_log(losses) if extra_log else ''
            elapsed = time.time() - t0
            print(
                f'step {step:5d} | train {losses["train"]:.4f} '
                f'| val {losses["val"]:.4f} | {elapsed:.1f}s{tag}'
            )

    print(f'\n[DONE] 训练完成,总用时 {time.time() - t0:.1f}s')
