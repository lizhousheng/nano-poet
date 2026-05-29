"""Week 3 · MiniGPT v0.5 = v0.4 + AMP + grad clip + weight decay。

新东西:
- AMP (Automatic Mixed Precision):前向用 fp16,反向自动维护 fp32 主权重
- 梯度裁剪:防止梯度爆炸
- weight_decay=0.1,betas=(0.9, 0.95):GPT-2 同款 AdamW 配置

──────────────────────────────────────────────────────────
【老师开讲:AMP 和那个 scaler 到底在防什么?】
平时算数用 fp32(32 位浮点),精度高但慢、占显存。AMP 改用 fp16(16 位)来算,快一倍、
省一半显存 —— 代价是 fp16 像一把"刻度粗"的尺子:太小的数会被直接四舍五入成 0(叫"下溢")。

麻烦在于:反向传播算出的**梯度**常常非常小,用 fp16 一量就变成 0,参数就更新不动了。
GradScaler 的妙招是:先把 loss 乘上一个大倍数(比如 ×1024)再反向 —— 梯度也跟着放大 1024 倍,
小数变成"量得到"的数;等要更新参数前,再把梯度除回去还原真实大小。一放一收,精度就保住了。

所以下面训练步比 v0.4 多了 scaler 的几行,五步的含义看 forward 循环里的行内注释。
(注:只有 CUDA 真正开 fp16,其它设备 autocast/scaler 会自动变成"直通",照常跑 fp32。
 完整讲解见 courses/03_optim_engineering.ipynb)
──────────────────────────────────────────────────────────
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.setup_device import auto_setup
auto_setup()  # 缺 GPU 后端时在 import torch 之前自动安装(见 scripts/setup_device.py)

import torch
from torch.amp import GradScaler

from configs.config import CHECKPOINT_DIR, MiniGPTv03Cfg, TOKENIZER_FILE
from model.minigpt_v03 import MiniGPTv03
from tokenizer.tokenizer import CharTokenizer
from train.lr_schedule import apply_lr, warmup_cosine
from train.utils import amp_enabled, autocast_ctx, get_device, load_data, make_get_batch


def main() -> None:
    device = get_device()
    print(f'device = {device}')

    cfg = MiniGPTv03Cfg
    num_steps     = 8000
    warmup_steps  = 200
    max_lr        = 3e-4
    min_lr        = 3e-5
    weight_decay  = 0.1
    grad_clip     = 1.0
    eval_interval = 500

    tokenizer = CharTokenizer.load(TOKENIZER_FILE)
    train_data, val_data = load_data(device)
    get_batch = make_get_batch(train_data, val_data, block_size=cfg.block_size)

    model = MiniGPTv03(
        tokenizer.vocab_size, cfg.n_embed, cfg.n_head, cfg.n_layer,
        cfg.block_size, cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=amp_enabled(device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params / 1e6:.2f}M')
    print('配置: schedule + AMP + grad clip + weight decay')
    print('对照: v03=4.14, v04 仅 schedule=4.24\n')

    @torch.no_grad()
    def estimate_loss_amp(eval_iters: int = 20) -> dict[str, float]:
        out = {}
        model.eval()
        for split in ('train', 'val'):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(split)
                with autocast_ctx(device):
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    t0 = time.time()
    best_val = float('inf')

    for step in range(num_steps):
        lr = warmup_cosine(step, warmup_steps, num_steps, max_lr, min_lr)
        apply_lr(optimizer, lr)

        x, y = get_batch('train')

        # autocast:前向时自动把部分算子用 fp16 跑,更快更省显存(非 CUDA 上是 no-op)
        with autocast_ctx(device):
            _, loss = model(x, y)

        # AMP 的反向比普通版多了 scaler 的几步,目的是防止 fp16 梯度下溢成 0:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()            # 1) 把 loss 放大再反向,梯度也跟着放大
        scaler.unscale_(optimizer)               # 2) 裁剪前先把梯度缩放还原回真实大小
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 3) 梯度裁剪防爆炸
        scaler.step(optimizer)                   # 4) 若梯度无 inf/nan 才真正更新参数
        scaler.update()                          # 5) 根据这步是否溢出,动态调整缩放倍数

        if step % eval_interval == 0 or step == num_steps - 1:
            losses = estimate_loss_amp()
            elapsed = time.time() - t0
            tag = ''
            if losses['val'] < 4.14: tag += ' [<v03]'
            if losses['val'] < best_val:
                best_val = losses['val']
                tag += ' [best]'
            print(
                f'step {step:5d} | lr {lr:.5f} | train {losses["train"]:.4f} '
                f'| val {losses["val"]:.4f} | {elapsed:.1f}s{tag}'
            )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / 'minigpt_v05.pt'
    torch.save(model.state_dict(), ckpt)
    print(f'\n[DONE] best val {best_val:.4f}, weights -> {ckpt}')


if __name__ == '__main__':
    main()
