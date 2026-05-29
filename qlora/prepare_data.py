"""Week 8 · 把 Week 6 的 SFT 数据转成 Qwen chat 格式。

两种采样策略:
- v1 (默认): 朴素随机 2000 条 —— 风格分布不均(七绝偏多)
- v2 (--balanced): 每种风格抽 1500 条 = 6000 条均衡

存成 HuggingFace Datasets,SFTTrainer 直接 load_from_disk 用。
"""
import argparse
import pickle
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    QLoRACfg,
    QWEN_SFT_TRAIN_DIR,
    QWEN_SFT_TRAIN_V2,
    QWEN_SFT_VAL_DIR,
    QWEN_SFT_VAL_V2,
    SFT_DATA_FILE,
    TOKENIZER_V3_FILE,
)

SUPPORTED_STYLES = ('五言绝句', '七言绝句', '五言律诗', '七言律诗')


def decode_sft_sample(ids: list[int], itos: dict[int, str]) -> tuple[str, str, str] | None:
    """把 Week 6 一条 token-id 样本解回 (title, style, poem)。"""
    text = ''.join(itos[i] for i in ids)
    try:
        title = text.split('<|题目|>')[1].split('<|风格|>')[0].strip()
        style = text.split('<|风格|>')[1].split('<|开始|>')[0].strip()
        poem  = text.split('<|开始|>')[1].split('<|结束|>')[0].strip()
        if title and style and poem:
            return title, style, poem
    except (IndexError, ValueError):
        pass
    return None


def make_qwen_sample(title: str, style: str, poem: str) -> dict:
    return {
        'messages': [
            {'role': 'user',
             'content': f'请以「{title}」为题,写一首{style}。'
                        f'严格遵守字数和句数,讲究意境与押韵。直接输出诗,不要解释。'},
            {'role': 'assistant', 'content': poem},
        ]
    }


def load_structured() -> list[tuple[str, str, str]]:
    with open(TOKENIZER_V3_FILE, 'rb') as f:
        tok = pickle.load(f)
    itos = tok['itos']
    sft = torch.load(SFT_DATA_FILE, weights_only=False)
    structured = []
    for ids in sft:
        parsed = decode_sft_sample(ids, itos)
        if parsed:
            structured.append(parsed)
    return structured


def build_v1(structured: list[tuple[str, str, str]]) -> None:
    """朴素采样 N 条,95/5 切分。"""
    cfg = QLoRACfg
    random.seed(42)
    random.shuffle(structured)
    chosen = structured[: cfg.samples_v1]
    samples = [make_qwen_sample(*x) for x in chosen]

    random.seed(42)
    random.shuffle(samples)
    n = int(cfg.train_val_ratio_v1 * len(samples))
    train, val = samples[:n], samples[n:]

    Dataset.from_list(train).save_to_disk(str(QWEN_SFT_TRAIN_DIR))
    Dataset.from_list(val).save_to_disk(str(QWEN_SFT_VAL_DIR))
    print(f'[v1] train={len(train)}, val={len(val)}  -> {QWEN_SFT_TRAIN_DIR.parent}')


def build_v2(structured: list[tuple[str, str, str]]) -> None:
    """按风格分组,每种抽 1500,合计 6000,97/3 切分。"""
    cfg = QLoRACfg
    by_style: dict[str, list[tuple[str, str, str]]] = {s: [] for s in SUPPORTED_STYLES}
    for t, s, p in structured:
        if s in by_style:
            by_style[s].append((t, s, p))
    print('原始分布:')
    for s, items in by_style.items():
        print(f'  {s}: {len(items)}')

    random.seed(42)
    balanced: list[tuple[str, str, str]] = []
    for s, items in by_style.items():
        k = min(cfg.samples_per_style, len(items))
        balanced.extend(random.sample(items, k))
        print(f'  采样 {s}: {k}')
    random.shuffle(balanced)

    samples = [make_qwen_sample(*x) for x in balanced]
    n = int(cfg.train_val_ratio_v2 * len(samples))
    train, val = samples[:n], samples[n:]

    Dataset.from_list(train).save_to_disk(str(QWEN_SFT_TRAIN_V2))
    Dataset.from_list(val).save_to_disk(str(QWEN_SFT_VAL_V2))
    print(f'\n[v2] train={len(train)}, val={len(val)}  -> {QWEN_SFT_TRAIN_V2.parent}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--balanced', action='store_true',
                        help='构造 v2 均衡数据集(每风格 1500 条,共 6000)')
    args = parser.parse_args()

    print('解码 Week 6 SFT 数据...')
    structured = load_structured()
    print(f'共 {len(structured)} 条')

    if args.balanced:
        build_v2(structured)
    else:
        build_v1(structured)


if __name__ == '__main__':
    main()
