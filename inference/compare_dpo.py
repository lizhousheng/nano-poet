"""Week 7 · 同 prompt + 同随机种子,对比 v0.9(SFT)和 v0.10(SFT+DPO)。

固定 seed 是关键 —— 否则采样噪声会淹没 DPO 的真实改进。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CHECKPOINT_DIR
from evaluation.score import score_poem
from inference.generate_sft import encode_pieces, generate, load_sft_model

TEST_CASES = [
    ('春雨', '五言绝句'),
    ('观棋', '七言绝句'),
    ('独居', '五言律诗'),
    ('送别', '七言律诗'),
    ('梅花', '五言绝句'),
    ('饮酒', '七言律诗'),
]


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    v09, stoi, itos = load_sft_model(CHECKPOINT_DIR / 'v09_sft' / 'best.pt', device)
    v10, _, _       = load_sft_model(CHECKPOINT_DIR / 'v10_dpo' / 'best.pt', device)

    print('=' * 70)
    print('v09 (SFT) vs v10 (SFT + DPO) head-to-head')
    print('=' * 70)

    win, lose, tie = 0, 0, 0
    for title, style in TEST_CASES:
        prompt = encode_pieces(['<|题目|>', title, '<|风格|>', style, '<|开始|>'], stoi)

        torch.manual_seed(42)
        poem_09 = generate(v09, prompt, stoi, itos, max_new=120, temperature=0.8, top_k=40)
        s09, _ = score_poem(poem_09, style, title)

        torch.manual_seed(42)
        poem_10 = generate(v10, prompt, stoi, itos, max_new=120, temperature=0.8, top_k=40)
        s10, _ = score_poem(poem_10, style, title)

        arrow = 'UP' if s10 > s09 else ('DOWN' if s10 < s09 else 'TIE')
        win  += int(s10 > s09)
        lose += int(s10 < s09)
        tie  += int(s10 == s09)

        print(f'\n[{title} · {style}]')
        print(f'  v09 (score={s09}):')
        for line in poem_09.split('\n'):
            print(f'    {line}')
        print(f'  v10 (score={s10}) {arrow}:')
        for line in poem_10.split('\n'):
            print(f'    {line}')

    print(f'\n{"=" * 70}\nv10 vs v09:  win={win}  lose={lose}  tie={tie}  /  {len(TEST_CASES)}')


if __name__ == '__main__':
    main()
