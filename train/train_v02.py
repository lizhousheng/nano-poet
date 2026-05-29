"""训练 MiniGPT v0.2(多头 + FFN + LN + 残差,单 Block)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.setup_device import auto_setup
auto_setup()  # 缺 GPU 后端时在 import torch 之前自动安装(见 scripts/setup_device.py)

import torch

from configs.config import (
    CHECKPOINT_DIR,
    LEARNING_RATE,
    MiniGPTv02Cfg,
    NUM_STEPS,
    TOKENIZER_FILE,
)
from model.minigpt_v02 import MiniGPTv02
from tokenizer.tokenizer import CharTokenizer
from train.utils import get_device, load_data, make_get_batch, train_loop


def main() -> None:
    device = get_device()
    print(f'device = {device}')

    tokenizer = CharTokenizer.load(TOKENIZER_FILE)
    train_data, val_data = load_data(device)
    get_batch = make_get_batch(train_data, val_data, block_size=MiniGPTv02Cfg.block_size)

    model = MiniGPTv02(
        vocab_size=tokenizer.vocab_size,
        n_embed=MiniGPTv02Cfg.n_embed,
        n_head=MiniGPTv02Cfg.n_head,
        block_size=MiniGPTv02Cfg.block_size,
        dropout=MiniGPTv02Cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f'参数量: {n_params / 1e6:.2f}M  '
        f'(n_embed={MiniGPTv02Cfg.n_embed}, n_head={MiniGPTv02Cfg.n_head})'
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train_loop(model, optimizer, get_batch, num_steps=NUM_STEPS)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / 'minigpt_v02.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f'[DONE] 模型已保存到 {ckpt_path}')


if __name__ == '__main__':
    main()
