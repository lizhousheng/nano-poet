"""Week 5 · MiniGPT v0.8 = 扩规模 25M + 唐宋数据。

变化:
- 模型从 6.37M → 25M (n_embed 192→384, n_layer 6→8, block_size 128→256)
- 数据从 4M → 22M 字符 (加宋诗 + 宋词)
- vocab 从 9563 → ~14500
- 第一次出现真正的"金句"——loss 跌到 ~3.5
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    CHECKPOINT_DIR,
    MiniGPTv08Cfg,
    TB_LOG_DIR,
    TOKENIZER_V2_FILE,
    TRAIN_DATA_V2_FILE,
    TrainV08Cfg,
    VAL_DATA_V2_FILE,
)
from model.minigpt_v03 import MiniGPTv03
from tokenizer.tokenizer import CharTokenizer
from train.checkpoint import save_checkpoint
from train.lr_schedule import apply_lr, warmup_cosine
from train.utils import amp_enabled, get_device

CKPT_DIR_V08 = CHECKPOINT_DIR / 'v08'


def main() -> None:
    device = get_device()
    cfg = MiniGPTv08Cfg
    tcfg = TrainV08Cfg

    tokenizer = CharTokenizer.load(TOKENIZER_V2_FILE)
    train_data = torch.load(TRAIN_DATA_V2_FILE, weights_only=True).to(device)
    val_data   = torch.load(VAL_DATA_V2_FILE,   weights_only=True).to(device)
    print(f'vocab_size = {tokenizer.vocab_size}')
    print(f'train: {train_data.shape}, val: {val_data.shape}')

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - cfg.block_size - 1, (tcfg.batch_size,))
        x = torch.stack([d[i : i + cfg.block_size]         for i in ix])
        y = torch.stack([d[i + 1 : i + 1 + cfg.block_size] for i in ix])
        return x, y

    config = {
        'n_embed': cfg.n_embed, 'n_head': cfg.n_head, 'n_layer': cfg.n_layer,
        'block_size': cfg.block_size, 'dropout': cfg.dropout,
        'vocab_size': tokenizer.vocab_size,
        'tokenizer_version': 'v2',
    }

    model = MiniGPTv03(
        config['vocab_size'], config['n_embed'], config['n_head'], config['n_layer'],
        config['block_size'], config['dropout'],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.max_lr, weight_decay=tcfg.weight_decay, betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=amp_enabled(device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params / 1e6:.2f}M  (应该 ~25M)')

    run_name = f'v08_{datetime.now().strftime("%m%d_%H%M")}'
    writer = SummaryWriter(TB_LOG_DIR / run_name)
    print(f'TB log: {TB_LOG_DIR / run_name}')
    print(f'ckpt:   {CKPT_DIR_V08}\n')

    @torch.no_grad()
    def estimate_loss_amp(eval_iters: int = 20) -> dict[str, float]:
        out = {}
        model.eval()
        for split in ('train', 'val'):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(split)
                with autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled(device)):
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    t0 = time.time()
    best_val = float('inf')

    for step in range(tcfg.num_steps):
        lr = warmup_cosine(step, tcfg.warmup_steps, tcfg.num_steps, tcfg.max_lr, tcfg.min_lr)
        apply_lr(optimizer, lr)

        x, y = get_batch('train')
        with autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled(device)):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar('lr', lr, step)
        writer.add_scalar('train/loss_step', loss.item(), step)
        writer.add_scalar('grad_norm', grad_norm.item(), step)

        if step % tcfg.eval_interval == 0 or step == tcfg.num_steps - 1:
            losses = estimate_loss_amp()
            elapsed = time.time() - t0
            writer.add_scalar('train/loss_eval', losses['train'], step)
            writer.add_scalar('val/loss',        losses['val'],   step)

            tag = ''
            if losses['val'] < best_val:
                best_val = losses['val']
                save_checkpoint(
                    CKPT_DIR_V08 / 'best.pt',
                    step=step, model=model, optimizer=optimizer, scaler=scaler,
                    best_val=best_val, config=config,
                )
                tag = ' [best]'
            eta = elapsed / max(step + 1, 1) * (tcfg.num_steps - step - 1)
            print(
                f'step {step:5d} | lr {lr:.5f} | train {losses["train"]:.4f} '
                f'| val {losses["val"]:.4f} | {elapsed:.0f}s (ETA {eta:.0f}s){tag}'
            )

        if step > 0 and step % tcfg.checkpoint_interval == 0:
            save_checkpoint(
                CKPT_DIR_V08 / 'latest.pt',
                step=step, model=model, optimizer=optimizer, scaler=scaler,
                best_val=best_val, config=config,
            )

    save_checkpoint(
        CKPT_DIR_V08 / 'latest.pt',
        step=tcfg.num_steps, model=model, optimizer=optimizer, scaler=scaler,
        best_val=best_val, config=config,
    )
    writer.close()
    print(f'\n[DONE] best val {best_val:.4f}  (对照 v07=4.21)')


if __name__ == '__main__':
    main()
