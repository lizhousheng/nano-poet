"""根据清洗后的文本构建 tokenizer,并把全文编码 + 切分成 train/val 张量。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    CLEAN_TEXT_FILE,
    TOKENIZER_FILE,
    TRAIN_DATA_FILE,
    TRAIN_RATIO,
    VAL_DATA_FILE,
)
from tokenizer.tokenizer import CharTokenizer


def build_and_encode() -> None:
    text = CLEAN_TEXT_FILE.read_text(encoding='utf-8')

    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(TOKENIZER_FILE)
    print(f'vocab_size = {tokenizer.vocab_size}  -> {TOKENIZER_FILE}')

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(TRAIN_RATIO * len(data))
    train, val = data[:n], data[n:]

    TRAIN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train, TRAIN_DATA_FILE)
    torch.save(val,   VAL_DATA_FILE)
    print(f'train: {train.shape} -> {TRAIN_DATA_FILE}')
    print(f'val:   {val.shape} -> {VAL_DATA_FILE}')


if __name__ == '__main__':
    build_and_encode()
