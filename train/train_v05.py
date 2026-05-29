"""Week 3 · MiniGPT v0.5 = v0.4 + AMP + grad clip + weight decay。

新东西:
- AMP (Automatic Mixed Precision):前向用 fp16,反向自动维护 fp32 主权重
- 梯度裁剪:防止梯度爆炸
- weight_decay=0.1,betas=(0.9, 0.95):GPT-2 同款 AdamW 配置
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

        with autocast_ctx(device):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

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
