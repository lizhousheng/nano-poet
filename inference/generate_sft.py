"""Week 6 · 给 SFT 模型一个题目 + 风格,让它生成一首诗。

用法:
    python inference/generate_sft.py --title 春雨 --style 五言绝句
    python inference/generate_sft.py --title 程序员 --style 七言绝句 --temperature 0.9 --top_k 40
    python inference/generate_sft.py --battery     # 跑 4 档 × 3 题 = 12 个测试
"""
import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CHECKPOINT_DIR, TOKENIZER_V3_FILE
from model.minigpt_v03 import MiniGPTv03
from train.utils import get_device

CKPT_DEFAULT = CHECKPOINT_DIR / 'v09_sft' / 'best.pt'

BATTERY = {
    'A · 古典正统题': [('明月', '五言绝句'), ('江雪', '七言绝句'), ('饮酒', '五言律诗')],
    'B · 古代但少见': [('观棋', '七言绝句'), ('夜读', '五言绝句'), ('卖花', '七言律诗')],
    'C · 现代题目用古典语言': [('咖啡', '五言绝句'), ('地铁', '七言绝句'), ('独居', '五言律诗')],
    'D · 完全现代 / 难题': [('程序员', '五言绝句'), ('深度学习', '七言绝句'), ('996', '七言绝句')],
}


def load_sft_model(ckpt_path: Path, device: str) -> tuple[MiniGPTv03, dict, dict]:
    with open(TOKENIZER_V3_FILE, 'rb') as f:
        tok = pickle.load(f)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg  = ckpt['config']
    model = MiniGPTv03(
        cfg['vocab_size'], cfg['n_embed'], cfg['n_head'], cfg['n_layer'],
        cfg['block_size'], cfg['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, tok['stoi'], tok['itos']


def encode_pieces(pieces: list[str], stoi: dict[str, int]) -> list[int]:
    ids: list[int] = []
    for p in pieces:
        if p in stoi:
            ids.append(stoi[p])
        else:
            for c in p:
                if c in stoi:
                    ids.append(stoi[c])
    return ids


@torch.no_grad()
def generate(
    model: MiniGPTv03,
    prompt_ids: list[int],
    stoi: dict[str, int],
    itos: dict[int, str],
    *,
    max_new: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    end_id = stoi['<|结束|>']
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out_ids: list[int] = []
    for _ in range(max_new):
        idx_cond = ids[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        nid = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nid], dim=1)
        nid_v = nid.item()
        if nid_v == end_id:
            break
        out_ids.append(nid_v)
    return ''.join(itos[i] for i in out_ids)


def write(model, stoi, itos, title: str, style: str, **kw) -> str:
    prompt = encode_pieces(['<|题目|>', title, '<|风格|>', style, '<|开始|>'], stoi)
    poem = generate(model, prompt, stoi, itos, **kw)
    print(f'\n[{title} · {style}]')
    print(poem)
    return poem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--title')
    parser.add_argument('--style', choices=list({s for _, s in
                        sum(BATTERY.values(), [])}) | {'五言绝句', '七言绝句', '五言律诗', '七言律诗'})
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--max_new', type=int, default=120)
    parser.add_argument('--ckpt', type=Path, default=CKPT_DEFAULT)
    parser.add_argument('--battery', action='store_true',
                        help='跑 4 档 × 3 题 测试套件,忽略 title/style')
    args = parser.parse_args()

    device = get_device()
    model, stoi, itos = load_sft_model(args.ckpt, device)
    print(f'模型: {args.ckpt}  (device={device})')

    if args.battery:
        for level, cases in BATTERY.items():
            print(f'\n{"=" * 60}\n{level}\n{"=" * 60}')
            for t, s in cases:
                write(model, stoi, itos, t, s,
                      max_new=args.max_new, temperature=args.temperature, top_k=args.top_k)
    else:
        assert args.title and args.style, '--title / --style 必填(或用 --battery)'
        write(model, stoi, itos, args.title, args.style,
              max_new=args.max_new, temperature=args.temperature, top_k=args.top_k)


if __name__ == '__main__':
    main()
