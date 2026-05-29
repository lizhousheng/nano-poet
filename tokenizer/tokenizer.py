"""字符级 tokenizer。

真实 LLM 用 BPE / SentencePiece;这里为了从零理解,先用最朴素的
"每个汉字 = 一个 token"。等理解透了再升级到 BPE 不晚。
"""
import pickle
from pathlib import Path
from typing import Iterable


class CharTokenizer:
    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def from_text(cls, text: str) -> 'CharTokenizer':
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi, itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Iterable[int]) -> str:
        return ''.join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'stoi': self.stoi, 'itos': self.itos}, f)

    @classmethod
    def load(cls, path: Path) -> 'CharTokenizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['stoi'], data['itos'])
