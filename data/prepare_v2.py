"""Week 5 · 扩数据:唐诗 + 宋诗 + 宋词,重建 v2 tokenizer。

最终产物:
- data/processed/poems_v2_clean.txt
- data/processed/tokenizer_v2.pkl  (vocab 从 9563 涨到 ~14500)
- data/processed/train_v2.pt / val_v2.pt  (95/5 切分)
"""
import glob
import json
import pickle
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    CLEAN_TEXT_V2_FILE,
    POETRY_REPO_DIR,
    TOKENIZER_V2_FILE,
    TRAIN_DATA_V2_FILE,
    VAL_DATA_V2_FILE,
)
from data.clean import NOISE_CHARS
from data.download import download

DATA_SOURCES = [
    ('全唐诗/poet.tang.*.json', '唐诗'),
    ('全唐诗/poet.song.*.json', '宋诗'),
    ('宋词/ci.song.*.json',     '宋词'),
]

V2_TRAIN_RATIO = 0.95          # 数据多了,验证集 5% 已足够


def collect_poems() -> tuple[list[str], dict[str, int]]:
    poems: list[str] = []
    counts: dict[str, int] = {}
    for pattern, name in DATA_SOURCES:
        files = sorted(glob.glob(str(POETRY_REPO_DIR / pattern)))
        n_before = len(poems)
        for filepath in files:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
            for poem in data:
                paragraphs = poem.get('paragraphs', [])
                if paragraphs:
                    poems.append('\n'.join(paragraphs))
        counts[name] = len(poems) - n_before
        print(f'  {name}: {counts[name]} 首')
    return poems, counts


def main() -> None:
    download()                                     # 确保仓库在

    print('合并所有数据源...')
    poems, counts = collect_poems()
    print(f'总计 {len(poems)} 首')

    # 打乱很重要:否则验证集会全是宋词
    random.seed(42)
    random.shuffle(poems)

    raw = '\n\n'.join(poems)
    print(f'原始合并: {len(raw):,} 字符')

    # 清洗(规则同 Week 1)
    cleaned = ''.join(c for c in raw if c not in NOISE_CHARS)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    print(f'清洗后:   {len(cleaned):,} 字符')

    chars = sorted(set(cleaned))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    print(f'vocab_size: {vocab_size}  (对比 v1 唐诗-only=9563)')

    data = torch.tensor([stoi[c] for c in cleaned], dtype=torch.long)
    n = int(V2_TRAIN_RATIO * len(data))
    train, val = data[:n].clone(), data[n:].clone()       # clone() 切断 view
    print(f'train: {train.shape[0]:,} tokens, val: {val.shape[0]:,} tokens')

    CLEAN_TEXT_V2_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_TEXT_V2_FILE.write_text(cleaned, encoding='utf-8')
    with open(TOKENIZER_V2_FILE, 'wb') as f:
        pickle.dump({
            'stoi': stoi, 'itos': itos,
            'vocab_size': vocab_size,
            'sources_counts': counts,
        }, f)
    torch.save(train, TRAIN_DATA_V2_FILE)
    torch.save(val,   VAL_DATA_V2_FILE)

    print('\n[DONE] v2 数据集已保存')
    for p in [CLEAN_TEXT_V2_FILE, TOKENIZER_V2_FILE, TRAIN_DATA_V2_FILE, VAL_DATA_V2_FILE]:
        print(f'  {p.name}: {p.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
