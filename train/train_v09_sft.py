"""Week 6 · MiniGPT v0.9 = v0.8 + SFT 指令微调。

关键技术:
1. **vocab 扩展**:v3 比 v2 多 5 个特殊 token。新建一个 vocab=14482 的模型,
   把 v0.8(vocab=14477)的 token_embedding / lm_head 的前 14477 行拷过来,
   后 5 行保留随机初始化 —— 这样旧字的能力一字不丢。
2. **prompt mask**:loss 只算在答案部分(<|开始|> 之后),prompt 部分置 -100。
3. **小 lr**:5e-5,比预训练小 6 倍,防止破坏 v0.8 已学好的语言能力。
"""
import math
import pickle
import random
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
    SFT_DATA_FILE,
    SFTCfg,
    TB_LOG_DIR,
    TOKENIZER_V3_FILE,
)
from model.minigpt_v03 import MiniGPTv03
from train.checkpoint import load_checkpoint, save_checkpoint
from train.utils import amp_enabled, get_device

CKPT_DIR_SFT  = CHECKPOINT_DIR / 'v09_sft'
CKPT_V08_BEST = CHECKPOINT_DIR / 'v08' / 'best.pt'


def transfer_weights_to_larger_vocab(
    new_model: torch.nn.Module,
    v08_state: dict[str, torch.Tensor],
    old_vocab: int,
) -> None:
    """v08 (vocab=14477) → v09 (vocab=14482):新增 5 行用随机初始化。"""
    new_state = new_model.state_dict()
    for key, val in v08_state.items():
        if key in ('token_embedding.weight', 'lm_head.weight'):
            new_state[key][:old_vocab] = val
        elif key == 'lm_head.bias' and 'lm_head.bias' in new_state:
            new_state[key][:old_vocab] = val
        else:
            new_state[key] = val
    new_model.load_state_dict(new_state)


def main() -> None:
    device = get_device()
    cfg = SFTCfg

    # ===== 1. tokenizer v3 =====
    with open(TOKENIZER_V3_FILE, 'rb') as f:
        tok = pickle.load(f)
    stoi, itos = tok['stoi'], tok['itos']
    new_vocab = tok['vocab_size']
    old_vocab = tok['old_vocab_size']
    print(f'vocab: {old_vocab} -> {new_vocab}')

    PAD_ID   = stoi['<|pad|>']
    START_ID = stoi['<|开始|>']

    # ===== 2. v08 best checkpoint =====
    assert CKPT_V08_BEST.exists(), f'找不到 v08 best,先跑 train_v08.py'
    ckpt08 = torch.load(CKPT_V08_BEST, map_location=device, weights_only=False)
    config08 = ckpt08['config']
    print(f'v08: {config08["n_embed"]} embed, {config08["n_layer"]} layers')

    # ===== 3. 新模型 + 迁移权重 =====
    model = MiniGPTv03(
        new_vocab,
        config08['n_embed'], config08['n_head'], config08['n_layer'],
        config08['block_size'], config08['dropout'],
    ).to(device)
    transfer_weights_to_larger_vocab(model, ckpt08['model_state'], old_vocab)
    print(f'参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # ===== 4. SFT 数据切分 =====
    samples = torch.load(SFT_DATA_FILE, weights_only=False)
    random.seed(42)
    random.shuffle(samples)
    n = int(0.95 * len(samples))
    sft_train, sft_val = samples[:n], samples[n:]
    print(f'SFT train: {len(sft_train)}, val: {len(sft_val)}')

    def make_batch(pool: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """随机抽 batch,pad 到 max_len,prompt 部分的 y 设为 -100。"""
        indices = random.sample(range(len(pool)), cfg.batch_size)
        x = torch.full((cfg.batch_size, cfg.max_len), PAD_ID, dtype=torch.long)
        y = torch.full((cfg.batch_size, cfg.max_len), -100,   dtype=torch.long)
        for i, idx in enumerate(indices):
            ids = pool[idx][: cfg.max_len]
            L = len(ids) - 1
            x[i, :L] = torch.tensor(ids[:-1])
            y[i, :L] = torch.tensor(ids[1:])
            try:
                start = ids.index(START_ID)
                y[i, :start] = -100              # prompt 部分不算 loss
            except ValueError:
                pass
        return x.to(device), y.to(device)

    # ===== 5. 优化器 / scaler / TB =====
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=amp_enabled(device))

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.max_lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(cfg.num_steps - cfg.warmup_steps, 1)
        progress = min(progress, 1.0)
        return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.max_lr - cfg.min_lr)

    @torch.no_grad()
    def estimate_sft_loss(eval_iters: int = 10) -> dict[str, float]:
        out = {}
        model.eval()
        for split, pool in (('train', sft_train), ('val', sft_val)):
            losses: list[float] = []
            for _ in range(eval_iters):
                x, y = make_batch(pool)
                with autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled(device)):
                    _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    run_name = f'v09_sft_{datetime.now().strftime("%m%d_%H%M")}'
    writer = SummaryWriter(TB_LOG_DIR / run_name)
    print(f'TB log: {TB_LOG_DIR / run_name}\n')

    # ===== 6. 训练 =====
    t0 = time.time()
    best_val = float('inf')

    for step in range(cfg.num_steps):
        lr = lr_at(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = make_batch(sft_train)
        with autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled(device)):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar('lr', lr, step)
        writer.add_scalar('train/loss_step', loss.item(), step)
        writer.add_scalar('grad_norm', grad_norm.item(), step)

        if step % cfg.eval_interval == 0 or step == cfg.num_steps - 1:
            losses = estimate_sft_loss()
            writer.add_scalar('train/loss_eval', losses['train'], step)
            writer.add_scalar('val/loss',        losses['val'],   step)

            tag = ''
            if losses['val'] < best_val:
                best_val = losses['val']
                config_sft = {**config08, 'vocab_size': new_vocab}
                save_checkpoint(
                    CKPT_DIR_SFT / 'best.pt',
                    step=step, model=model, optimizer=optimizer, scaler=scaler,
                    best_val=best_val, config=config_sft,
                )
                tag = ' [best]'
            elapsed = time.time() - t0
            eta = elapsed / max(step + 1, 1) * (cfg.num_steps - step - 1)
            print(
                f'step {step:5d} | lr {lr:.6f} | train {losses["train"]:.4f} '
                f'| val {losses["val"]:.4f} | grad {grad_norm.item():.2f} '
                f'| {elapsed:.0f}s (ETA {eta:.0f}s){tag}'
            )

    writer.close()
    print(f'\n[DONE] SFT best val {best_val:.4f}  (SFT loss << 预训练 loss,因为 prompt 被 mask)')


if __name__ == '__main__':
    main()
