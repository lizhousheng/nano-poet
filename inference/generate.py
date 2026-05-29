"""加载某个 checkpoint,自回归生成一段诗。

用法:
    python inference/generate.py --model v03 --start 月 --tokens 200
    python inference/generate.py --model bigram --start 春
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    CHECKPOINT_DIR,
    MiniGPTv01Cfg,
    MiniGPTv02Cfg,
    MiniGPTv03Cfg,
    TOKENIZER_FILE,
)
from model.bigram import BigramModel
from model.minigpt_v01 import MiniGPTv01
from model.minigpt_v02 import MiniGPTv02
from model.minigpt_v03 import MiniGPTv03
from tokenizer.tokenizer import CharTokenizer
from train.utils import get_device


def build_model(name: str, vocab_size: int) -> tuple[torch.nn.Module, Path]:
    if name == 'bigram':
        return BigramModel(vocab_size), CHECKPOINT_DIR / 'bigram.pt'
    if name == 'v01':
        m = MiniGPTv01(vocab_size, MiniGPTv01Cfg.n_embed, MiniGPTv01Cfg.head_size, MiniGPTv01Cfg.block_size)
        return m, CHECKPOINT_DIR / 'minigpt_v01.pt'
    if name == 'v02':
        m = MiniGPTv02(
            vocab_size,
            MiniGPTv02Cfg.n_embed, MiniGPTv02Cfg.n_head,
            MiniGPTv02Cfg.block_size, MiniGPTv02Cfg.dropout,
        )
        return m, CHECKPOINT_DIR / 'minigpt_v02.pt'
    if name == 'v03':
        m = MiniGPTv03(
            vocab_size,
            MiniGPTv03Cfg.n_embed, MiniGPTv03Cfg.n_head, MiniGPTv03Cfg.n_layer,
            MiniGPTv03Cfg.block_size, MiniGPTv03Cfg.dropout,
        )
        return m, CHECKPOINT_DIR / 'minigpt_v03.pt'
    raise ValueError(f'未知模型: {name}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bigram', 'v01', 'v02', 'v03'], default='v03')
    parser.add_argument('--start', default='月')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    device = get_device()
    tokenizer = CharTokenizer.load(TOKENIZER_FILE)

    model, ckpt_path = build_model(args.model, tokenizer.vocab_size)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'找不到权重 {ckpt_path},先训练。')
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.to(device)

    start_ids = torch.tensor([tokenizer.encode(args.start)], dtype=torch.long, device=device)
    out_ids = model.generate(start_ids, max_new_tokens=args.tokens, temperature=args.temperature)
    print(tokenizer.decode(out_ids[0].tolist()))


if __name__ == '__main__':
    main()
