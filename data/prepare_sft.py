"""Week 6 · 构造 SFT 指令数据集。

每个样本格式:
    <|题目|>静夜思<|风格|>五言绝句<|开始|>床前明月光\n...<|结束|>

构造好后保存成 list[list[int]] —— 训练时再 pad + 把 prompt 部分 mask 成 -100。
"""
import glob
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import POETRY_REPO_DIR, SFT_DATA_FILE, TOKENIZER_V3_FILE
from data.classify import STYLE_NAME, classify

MAX_TITLE_LEN = 30                          # 过长的标题大概率是噪音


def encode_pieces(pieces: list[str], stoi: dict[str, int]) -> list[int]:
    """混合编码:special token 整体一个 id,普通字符串逐字符。"""
    ids: list[int] = []
    for p in pieces:
        if p in stoi:
            ids.append(stoi[p])
        else:
            for c in p:
                if c in stoi:
                    ids.append(stoi[c])
    return ids


def main() -> None:
    with open(TOKENIZER_V3_FILE, 'rb') as f:
        tok = pickle.load(f)
    stoi = tok['stoi']
    print(f'tokenizer_v3 vocab: {len(stoi)}')

    json_files = sorted(glob.glob(str(POETRY_REPO_DIR / '全唐诗' / 'poet.tang.*.json')))
    if not json_files:
        raise FileNotFoundError('请先跑 data/prepare_v2.py(它会确保仓库 clone 好)')

    samples: list[list[int]] = []
    style_counts: dict[str, int] = {}

    for fp in json_files:
        with open(fp, encoding='utf-8') as f:
            data = json.load(f)
        for poem in data:
            title = poem.get('title', '').strip()
            paragraphs = poem.get('paragraphs', [])
            if not title or not paragraphs or len(title) > MAX_TITLE_LEN:
                continue

            style = classify(paragraphs)
            if style == 'other':
                continue

            style_counts[style] = style_counts.get(style, 0) + 1
            body = '\n'.join(paragraphs)
            ids = encode_pieces(
                ['<|题目|>', title, '<|风格|>', STYLE_NAME[style],
                 '<|开始|>', body, '<|结束|>'],
                stoi,
            )
            samples.append(ids)

    print(f'\n构造完成: {len(samples)} 条 SFT 样本')
    for s, n in sorted(style_counts.items(), key=lambda kv: -kv[1]):
        print(f'  {s} ({STYLE_NAME[s]}): {n}')

    lens = [len(s) for s in samples]
    print(f'\n长度统计: min={min(lens)} mid={sorted(lens)[len(lens)//2]} '
          f'mean={sum(lens)/len(lens):.1f} max={max(lens)}')

    SFT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, SFT_DATA_FILE)
    print(f'\n[DONE] -> {SFT_DATA_FILE}  ({SFT_DATA_FILE.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
