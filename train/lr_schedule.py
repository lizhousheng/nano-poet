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
    """阶段 1 线性 warmup,阶段 2 cosine decay 到 min_lr。

    为什么要 warmup:刚开始权重是随机的,直接上大 lr 容易把模型"冲飞"。先从接近 0
    线性涨到 max_lr,让训练平稳起步;之后再沿余弦曲线缓慢降到 min_lr 精细收敛。
    """
    if step < warmup_steps:
        # 阶段 1:从 max_lr/warmup_steps 线性涨到 max_lr
        return max_lr * (step + 1) / warmup_steps
    # 阶段 2:progress 从 0 → 1 表示 decay 走了多少
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)                # 防止越界(step 超过 total_steps)
    # cos 从 0 到 π,coeff 从 1 平滑降到 0;再映射到 [min_lr, max_lr] 区间
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
