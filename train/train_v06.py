"""Week 4 · MiniGPT v0.6 = v0.5 + 完整 Checkpoint。

新东西:把模型/优化器/scaler/step/best_val/config 全部打包保存。
- best.pt    val 创新低时覆盖
- latest.pt  每 1000 步覆盖,用于 resume

后续用 train_v07.py --resume 可以从 latest.pt 继续训。
"""
import sys
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CHECKPOINT_DIR, MiniGPTv03Cfg, TOKENIZER_FILE
from model.minigpt_v03 import MiniGPTv03
from tokenizer.tokenizer import CharTokenizer
from train.checkpoint import save_checkpoint
from train.lr_schedule import apply_lr, warmup_cosine
from train.utils import get_device, load_data, make_get_batch

CKPT_DIR_V06 = CHECKPOINT_DIR / 'v06'


def main() -> None:
    device = get_device()
    print(f'device = {device}')

    cfg = MiniGPTv03Cfg
    num_steps           = 8000
    warmup_steps        = 200
    max_lr              = 3e-4
    min_lr              = 3e-5
    weight_decay        = 0.1
    grad_clip           = 1.0
    eval_interval       = 500
    checkpoint_interval = 1000

    tokenizer = CharTokenizer.load(TOKENIZER_FILE)
    train_data, val_data = load_data(device)
    get_batch = make_get_batch(train_data, val_data, block_size=cfg.block_size)

    config = {
        'n_embed':    cfg.n_embed,
        'n_head':     cfg.n_head,
        'n_layer':    cfg.n_layer,
        'block_size': cfg.block_size,
        'dropout':    cfg.dropout,
        'vocab_size': tokenizer.vocab_size,
    }

    model = MiniGPTv03(
        config['vocab_size'], config['n_embed'], config['n_head'], config['n_layer'],
        config['block_size'], config['dropout'],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )
    scaler = GradScaler()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params / 1e6:.2f}M')
    print(f'Checkpoint 目录: {CKPT_DIR_V06}\n')

    @torch.no_grad()
    def estimate_loss_amp(eval_iters: int = 20) -> dict[str, float]:
        out = {}
        model.eval()
        for split in ('train', 'val'):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(split)
                with autocast(device_type=device, dtype=torch.float16):
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
        with autocast(device_type=device, dtype=torch.float16):
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
            if losses['val'] < best_val:
                best_val = losses['val']
                save_checkpoint(
                    CKPT_DIR_V06 / 'best.pt',
                    step=step, model=model, optimizer=optimizer, scaler=scaler,
                    best_val=best_val, config=config,
                )
                tag = ' [best]'
            print(
                f'step {step:5d} | lr {lr:.5f} | train {losses["train"]:.4f} '
                f'| val {losses["val"]:.4f} | {elapsed:.1f}s{tag}'
            )

        if step > 0 and step % checkpoint_interval == 0:
            save_checkpoint(
                CKPT_DIR_V06 / 'latest.pt',
                step=step, model=model, optimizer=optimizer, scaler=scaler,
                best_val=best_val, config=config,
            )

    save_checkpoint(
        CKPT_DIR_V06 / 'latest.pt',
        step=num_steps, model=model, optimizer=optimizer, scaler=scaler,
        best_val=best_val, config=config,
    )
    print(f'\n[DONE] best val {best_val:.4f}')
    for fn in CKPT_DIR_V06.iterdir():
        print(f'  {fn.name}: {fn.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
