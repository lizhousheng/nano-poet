"""Week 7 · 用 v0.9 SFT 模型生成偏好对(自标注)。

流程:
1. 从 SFT val 抽 1000 个唯一 (题目, 风格)
2. 每个 prompt 用 v0.9 生成 N_CANDIDATES=4 首
3. 用 evaluation/score.py 打分,挑最高分作 chosen,最低分作 rejected
4. 只保留分差 >= MIN_GAP 的对,保证训练信号清晰
5. 保存成 list[dict],每个 dict 含 prompt_ids/chosen_ids/rejected_ids/scores

跑下来大约能收到 500~700 对(取决于评分分布)。
"""
import pickle
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CHECKPOINT_DIR, DPO_DATA_FILE, DPOCfg, SFT_DATA_FILE, TOKENIZER_V3_FILE
from evaluation.score import score_poem
from inference.generate_sft import encode_pieces, generate, load_sft_model

N_PROMPTS = 1000


def extract_meta(sample_ids: list[int], stoi, itos):
    start_id = stoi['<|开始|>']
    try:
        pos = sample_ids.index(start_id)
    except ValueError:
        return None
    prompt_ids = sample_ids[:pos + 1]
    text = ''.join(itos[i] for i in prompt_ids)
    try:
        title = text.split('<|题目|>')[1].split('<|风格|>')[0]
        style = text.split('<|风格|>')[1].split('<|开始|>')[0]
    except IndexError:
        return None
    return {'prompt_ids': prompt_ids, 'title': title, 'style': style}


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = DPOCfg

    model_sft, stoi, itos = load_sft_model(CHECKPOINT_DIR / 'v09_sft' / 'best.pt', device)
    end_id = stoi['<|结束|>']

    # 用 SFT val 当 prompt 池(确保 v0.9 没在这些上过拟合)
    samples = torch.load(SFT_DATA_FILE, weights_only=False)
    random.seed(42)
    random.shuffle(samples)
    n = int(0.95 * len(samples))
    val_pool = samples[n:]

    prompts = []
    seen = set()
    for s in val_pool:
        meta = extract_meta(s, stoi, itos)
        if not meta:
            continue
        key = (meta['title'], meta['style'])
        if key in seen:
            continue
        seen.add(key)
        prompts.append(meta)
        if len(prompts) >= N_PROMPTS:
            break

    print(f'采样 {len(prompts)} 个唯一 prompt')

    preferences = []
    t0 = time.time()

    for i, p in enumerate(prompts):
        cands = []
        for _ in range(cfg.n_candidates):
            ans_text = generate(
                model_sft, p['prompt_ids'], stoi, itos,
                max_new=120, temperature=1.0, top_k=40,
            )
            if not ans_text.strip():
                continue
            sc, _ = score_poem(ans_text, p['style'], p['title'])
            # full_ids 要包含 prompt + 答案 + <|结束|>(便于 DPO compute_logprobs)
            ans_ids = [stoi[c] for c in ans_text if c in stoi]
            full_ids = p['prompt_ids'] + ans_ids + [end_id]
            cands.append({'full_ids': full_ids, 'score': sc})

        if len(cands) < 2:
            continue
        cands.sort(key=lambda c: -c['score'])
        chosen, rejected = cands[0], cands[-1]
        if chosen['score'] - rejected['score'] < cfg.min_score_gap:
            continue

        preferences.append({
            'prompt_ids':     p['prompt_ids'],
            'chosen_ids':     chosen['full_ids'],
            'rejected_ids':   rejected['full_ids'],
            'chosen_score':   chosen['score'],
            'rejected_score': rejected['score'],
            'title':          p['title'],
            'style':          p['style'],
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(prompts) - i - 1)
            print(f'  [{i+1}/{len(prompts)}] 已收 {len(preferences)} 对 | {elapsed:.0f}s | ETA {eta:.0f}s')

    print(f'\n[DONE] {len(preferences)} 对偏好,用时 {(time.time()-t0)/60:.1f} 分钟')

    if preferences:
        gaps = [p['chosen_score'] - p['rejected_score'] for p in preferences]
        print(f'分差: min={min(gaps)} mid={sorted(gaps)[len(gaps)//2]} '
              f'mean={sum(gaps)/len(gaps):.1f} max={max(gaps)}')

    DPO_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(preferences, DPO_DATA_FILE)
    print(f'-> {DPO_DATA_FILE}')


if __name__ == '__main__':
    main()
