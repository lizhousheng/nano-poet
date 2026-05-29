"""训练通用工具:加载数据、采样 batch、估计 loss。"""
import time
from typing import Callable

import torch

from configs.config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    EVAL_INTERVAL,
    EVAL_ITERS,
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
)


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


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
