"""Week 4 · MiniGPT v0.7 = v0.6 + TensorBoard + resume。

新东西:
- 支持 --resume:从 checkpoints/v06/latest.pt 继续训
- 每步把 lr / loss / grad_norm 写进 TensorBoard
- 每 eval_interval 步把 train/val loss 写进去
- 新一轮可以用更小的 lr (continue training trick)

用法:
    python train/train_v07.py             # 从头训
    python train/train_v07.py --resume    # 从 v06 latest 继续
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.setup_device import auto_setup
auto_setup()  # 缺 GPU 后端时在 import torch 之前自动安装(见 scripts/setup_device.py)

import torch
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from configs.config import CHECKPOINT_DIR, MiniGPTv03Cfg, PROJECT_ROOT, TOKENIZER_FILE
from model.minigpt_v03 import MiniGPTv03
from tokenizer.tokenizer import CharTokenizer
from train.checkpoint import load_checkpoint, save_checkpoint
from train.lr_schedule import apply_lr, warmup_cosine
from train.utils import amp_enabled, autocast_ctx, get_device, load_data, make_get_batch

CKPT_DIR_V07 = CHECKPOINT_DIR / 'v07'
LOG_DIR_BASE = PROJECT_ROOT / 'tb_logs'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='从 checkpoints/v06/latest.pt 续训')
    parser.add_argument('--extra-steps', type=int, default=4000,
                        help='resume 时要再训多少步')
    args = parser.parse_args()

    device = get_device()
    tokenizer = CharTokenizer.load(TOKENIZER_FILE)
    train_data, val_data = load_data(device)

    if args.resume:
        # 从 v06 的 latest 加载,但用更小的 lr 微调
        resume_ckpt = CHECKPOINT_DIR / 'v06' / 'latest.pt'
        assert resume_ckpt.exists(), f'找不到 {resume_ckpt},先跑 train_v06.py'
        max_lr, min_lr, warmup = 1e-4, 1e-5, 100
        num_steps = args.extra_steps
        run_tag   = 'v07_resume'
    else:
        resume_ckpt = None
        max_lr, min_lr, warmup = 3e-4, 3e-5, 200
        num_steps = 8000
        run_tag   = 'v07_fresh'

    weight_decay        = 0.1
    grad_clip           = 1.0
    eval_interval       = 250
    checkpoint_interval = 1000

    cfg = MiniGPTv03Cfg
    get_batch = make_get_batch(train_data, val_data, block_size=cfg.block_size)

    config = {
        'n_embed': cfg.n_embed, 'n_head': cfg.n_head, 'n_layer': cfg.n_layer,
        'block_size': cfg.block_size, 'dropout': cfg.dropout,
        'vocab_size': tokenizer.vocab_size,
    }

    model = MiniGPTv03(
        config['vocab_size'], config['n_embed'], config['n_head'], config['n_layer'],
        config['block_size'], config['dropout'],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=amp_enabled(device))

    start_step = 0
    best_val   = float('inf')
    if resume_ckpt:
        prev = load_checkpoint(resume_ckpt, model=model, optimizer=optimizer, scaler=scaler, map_location=device)
        start_step = prev['step']
        best_val   = prev['best_val']
        print(f'[resume] from step {start_step}, prev best val {best_val:.4f}')

    run_name = f'{run_tag}_{datetime.now().strftime("%m%d_%H%M%S")}'
    writer = SummaryWriter(LOG_DIR_BASE / run_name)
    print(f'TensorBoard log: {LOG_DIR_BASE / run_name}')
    print(f'max_lr={max_lr}, total_steps={num_steps}, warmup={warmup}\n')

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
    for local_step in range(num_steps):
        # local_step:本次运行的第几步(0..num_steps),决定 lr 曲线
        # global_step:累计总步数(续训时接着上次的 step 往后数),用于日志/checkpoint 记账
        global_step = start_step + local_step

        # 注意 lr 用 local_step:每次 resume 都重新走一遍 warmup→cosine,而不是从曲线尾巴接
        lr = warmup_cosine(local_step, warmup, num_steps, max_lr, min_lr)
        apply_lr(optimizer, lr)

        x, y = get_batch('train')
        with autocast_ctx(device):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar('lr', lr, global_step)
        writer.add_scalar('train/loss_step', loss.item(), global_step)
        writer.add_scalar('grad_norm', grad_norm.item(), global_step)

        if local_step % eval_interval == 0 or local_step == num_steps - 1:
            losses = estimate_loss_amp()
            writer.add_scalar('train/loss_eval', losses['train'], global_step)
            writer.add_scalar('val/loss',        losses['val'],   global_step)

            tag = ''
            if losses['val'] < best_val:
                best_val = losses['val']
                save_checkpoint(
                    CKPT_DIR_V07 / 'best.pt',
                    step=global_step, model=model, optimizer=optimizer, scaler=scaler,
                    best_val=best_val, config=config,
                )
                tag = ' [best]'
            elapsed = time.time() - t0
            print(
                f'step {global_step:6d} | lr {lr:.6f} | train {losses["train"]:.4f} '
                f'| val {losses["val"]:.4f} | grad {grad_norm.item():.2f} | {elapsed:.1f}s{tag}'
            )

        if local_step > 0 and local_step % checkpoint_interval == 0:
            save_checkpoint(
                CKPT_DIR_V07 / 'latest.pt',
                step=global_step, model=model, optimizer=optimizer, scaler=scaler,
                best_val=best_val, config=config,
            )

    save_checkpoint(
        CKPT_DIR_V07 / 'latest.pt',
        step=start_step + num_steps, model=model, optimizer=optimizer, scaler=scaler,
        best_val=best_val, config=config,
    )
    writer.close()
    print(f'\n[DONE] best val {best_val:.4f}, logs -> {LOG_DIR_BASE / run_name}')


if __name__ == '__main__':
    main()
