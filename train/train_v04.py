"""Week 3 · MiniGPT v0.4 = v0.3 结构 + LR Schedule。

唯一新东西:用 warmup_cosine 替代恒定 lr。
对照:v03 恒定 lr=3e-4 → val 4.14;v04 schedule → val ~4.21~4.24。
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CHECKPOINT_DIR, MiniGPTv03Cfg, TOKENIZER_FILE
from model.minigpt_v03 import MiniGPTv03
from tokenizer.tokenizer import CharTokenizer
from train.lr_schedule import apply_lr, warmup_cosine
from train.utils import estimate_loss, get_device, load_data, make_get_batch


def main() -> None:
    device = get_device()
    print(f'device = {device}')

    # 超参
    cfg = MiniGPTv03Cfg
    num_steps    = 8000
    warmup_steps = 200
    max_lr       = 3e-4
    min_lr       = 3e-5
    eval_interval = 500

    tokenizer = CharTokenizer.load(TOKENIZER_FILE)
    train_data, val_data = load_data(device)
    get_batch = make_get_batch(train_data, val_data, block_size=cfg.block_size)

    model = MiniGPTv03(
        tokenizer.vocab_size, cfg.n_embed, cfg.n_head, cfg.n_layer,
        cfg.block_size, cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params / 1e6:.2f}M  (应该是 6.37M)')
    print(f'对照 v03 恒定 lr=3e-4 → val 4.14\n')

    t0 = time.time()
    best_val = float('inf')

    for step in range(num_steps):
        lr = warmup_cosine(step, warmup_steps, num_steps, max_lr, min_lr)
        apply_lr(optimizer, lr)

        x, y = get_batch('train')
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == num_steps - 1:
            losses = estimate_loss(model, get_batch)
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
    ckpt = CHECKPOINT_DIR / 'minigpt_v04.pt'
    torch.save(model.state_dict(), ckpt)
    print(f'\n[DONE] best val {best_val:.4f}, weights -> {ckpt}')


if __name__ == '__main__':
    main()
