"""Warmup + cosine decay 学习率调度。

Week 3 引入。原始 v0.3 用恒定 lr=3e-4 → val 4.14;
加上 schedule 后 v0.4 → val 4.21~4.24(略提升,主要是训练健康度更好)。
"""
import math


def warmup_cosine(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """阶段 1 线性 warmup,阶段 2 cosine decay 到 min_lr。"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def apply_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg['lr'] = lr


if __name__ == '__main__':
    # 打印几个关键点,方便核对曲线形状
    warmup, total, mx, mn = 200, 8000, 3e-4, 3e-5
    for s in [0, 50, 100, 200, 500, 2000, 4000, 6000, 7999]:
        print(f'step {s:5d}: lr = {warmup_cosine(s, warmup, total, mx, mn):.6f}')
