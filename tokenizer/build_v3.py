"""Week 6 · 在 v2 词表基础上追加 SFT 特殊 token,生成 tokenizer_v3。

特殊 token:<|题目|> <|风格|> <|开始|> <|结束|> <|pad|>
追加在 vocab 末尾,这样旧 ID 完全不动 —— Week 6 SFT 训练时
v0.8 的权重可以无损迁移过来(只在新增 5 行随机初始化)。
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import SPECIAL_TOKENS, TOKENIZER_V2_FILE, TOKENIZER_V3_FILE


def build() -> None:
    with open(TOKENIZER_V2_FILE, 'rb') as f:
        v2 = pickle.load(f)
    stoi = dict(v2['stoi'])
    itos = dict(v2['itos'])
    old_vocab = len(stoi)

    for tok in SPECIAL_TOKENS:
        if tok not in stoi:
            new_id = len(stoi)
            stoi[tok] = new_id
            itos[new_id] = tok

    new_vocab = len(stoi)
    print(f'old vocab: {old_vocab}  ->  new vocab: {new_vocab}')
    for tok in SPECIAL_TOKENS:
        print(f'  {tok}: id={stoi[tok]}')

    TOKENIZER_V3_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKENIZER_V3_FILE, 'wb') as f:
        pickle.dump({
            'stoi': stoi, 'itos': itos,
            'vocab_size': new_vocab,
            'special_tokens': SPECIAL_TOKENS,
            'old_vocab_size': old_vocab,
        }, f)
    print(f'\n[DONE] -> {TOKENIZER_V3_FILE}')


if __name__ == '__main__':
    build()
