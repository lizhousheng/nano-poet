"""训练 MiniGPT v0.1(单头 Attention),保存到 checkpoints/minigpt_v01.pt。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    CHECKPOINT_DIR,
    LEARNING_RATE,
    MiniGPTv01Cfg,
    NUM_STEPS,
    TOKENIZER_FILE,
)
from model.minigpt_v01 import MiniGPTv01
from tokenizer.tokenizer import CharTokenizer
from train.utils import get_device, load_data, make_get_batch, train_loop


def main() -> None:
    device = get_device()
    print(f'device = {device}')

    tokenizer = CharTokenizer.load(TOKENIZER_FILE)
    train_data, val_data = load_data(device)
    get_batch = make_get_batch(train_data, val_data, block_size=MiniGPTv01Cfg.block_size)

    model = MiniGPTv01(
        vocab_size=tokenizer.vocab_size,
        n_embed=MiniGPTv01Cfg.n_embed,
        head_size=MiniGPTv01Cfg.head_size,
        block_size=MiniGPTv01Cfg.block_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f'参数量: {n_params / 1e6:.2f}M  '
        f'(n_embed={MiniGPTv01Cfg.n_embed}, head_size={MiniGPTv01Cfg.head_size})'
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_loop(model, optimizer, get_batch, num_steps=NUM_STEPS)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / 'minigpt_v01.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f'[DONE] 模型已保存到 {ckpt_path}')


if __name__ == '__main__':
    main()
