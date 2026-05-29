"""Week 7 · MiniGPT v0.10 = v0.9 + DPO 偏好对齐。

流程:
1. 加载 v0.9 SFT best 作为 policy(可训练)
2. 同样权重深拷贝一份作为 ref_model(冻结)
3. 用 prepare_dpo.py 生成的 (chosen, rejected) 对计算 DPO loss
4. 训 500 步即可,lr 极小 (1e-5)

──────────────────────────────────────────────────────────
【老师开讲:DPO 是在教模型"什么是更好的回答"】
SFT 教会了模型"怎么答",但答得"好不好"它还没概念 —— 同一个问题,可能有个有礼貌的好答案,
也有个生硬的坏答案,模型不知道该偏向哪个。

经典做法(RLHF)要先训练一个"评委模型"给答案打分,很重。DPO(Direct Preference Optimization)
聪明在:**跳过评委**,直接拿成对的数据 (chosen 好答案, rejected 坏答案) 来教 ——
让模型"把好答案的概率调高一点、把坏答案的概率调低一点",一步到位。

这里有两个模型(看 main 里):
  · **policy**:正在训练的那个,我们要让它越来越偏好"好答案";
  · **ref(参考)**:训练开始那一刻的快照,**全程冻结**。它当"锚",防止 policy 为了讨好
    偏好数据而跑偏、把好不容易练出的语言能力练崩。DPO 比较的是"policy 相对 ref 的变化"。

判断有没有学会,看 acc(选对 chosen 的比例,越接近 1 越好)和 loss(理想 << 0.69)。
DPO loss 的数学公式见 train/dpo_loss.py;完整讲解见 courses/07_dpo_preference_alignment.ipynb。
──────────────────────────────────────────────────────────
"""
import copy
import math
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.setup_device import auto_setup
auto_setup()  # 缺 GPU 后端时在 import torch 之前自动安装(见 scripts/setup_device.py)

import torch
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from configs.config import (
    CHECKPOINT_DIR,
    DPO_DATA_FILE,
    DPOCfg,
    TB_LOG_DIR,
    TOKENIZER_V3_FILE,
)
from model.minigpt_v03 import MiniGPTv03
from train.checkpoint import save_checkpoint
from train.dpo_loss import dpo_loss
from train.utils import amp_enabled, get_device

CKPT_DIR_DPO = CHECKPOINT_DIR / 'v10_dpo'
CKPT_SFT_BEST = CHECKPOINT_DIR / 'v09_sft' / 'best.pt'


def make_dpo_batch(pool, batch_size, max_len, pad_id, device):
    """抽一批偏好对。每条样本含同一个 prompt 的两种续写:chosen(更好)和 rejected(更差)。

    返回 chosen / rejected 两条序列,以及 plens(prompt 长度,后面算 loss 时
    要据此跳过 prompt 部分,只比较"答案"的 log-prob)。
    """
    indices = random.sample(range(len(pool)), batch_size)
    chosen   = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    rejected = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    plens = []
    for i, idx in enumerate(indices):
        p = pool[idx]
        c, r = p['chosen_ids'][:max_len], p['rejected_ids'][:max_len]
        chosen[i, :len(c)]   = torch.tensor(c)
        rejected[i, :len(r)] = torch.tensor(r)
        plens.append(min(len(p['prompt_ids']), max_len))   # prompt 长度(截断到 max_len)
    return chosen.to(device), rejected.to(device), plens


def main() -> None:
    device = get_device()
    cfg = DPOCfg

    # ===== tokenizer =====
    with open(TOKENIZER_V3_FILE, 'rb') as f:
        tok = pickle.load(f)
    pad_id = tok['stoi']['<|pad|>']

    # ===== 加载 SFT best 作为 policy =====
    assert CKPT_SFT_BEST.exists(), '先跑 train_v09_sft.py'
    ckpt = torch.load(CKPT_SFT_BEST, map_location='cpu', weights_only=False)
    sft_cfg = ckpt['config']

    model = MiniGPTv03(
        sft_cfg['vocab_size'], sft_cfg['n_embed'], sft_cfg['n_head'], sft_cfg['n_layer'],
        sft_cfg['block_size'], sft_cfg['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state'])

    # ===== 同权重深拷贝作为冻结 ref =====
    # ref_model 是"出发点"的快照:DPO 让 policy 在偏好对上变得更好,同时用 ref 当锚,
    # 惩罚 policy 偏离太远(防止为了讨好偏好而把语言能力练崩)。它不参与训练,永远冻结。
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False                  # 冻结:不算梯度、不更新
    print(f'policy: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    print(f'ref:    冻结同结构')

    # ===== 偏好对 =====
    preferences = torch.load(DPO_DATA_FILE, weights_only=False)
    random.seed(42)
    random.shuffle(preferences)
    n = int(0.95 * len(preferences))
    dpo_train, dpo_val = preferences[:n], preferences[n:]
    print(f'DPO train: {len(dpo_train)}, val: {len(dpo_val)}\n')

    # ===== 优化器 =====
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=amp_enabled(device))

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(cfg.num_steps - cfg.warmup_steps, 1)
        progress = min(progress, 1.0)
        return cfg.lr * 0.1 + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.lr * 0.1)

    @torch.no_grad()
    def estimate_dpo(eval_iters: int = 10) -> dict[str, dict[str, float]]:
        out = {}
        model.eval()                                 # 评估时关 Dropout
        for split, pool in (('train', dpo_train), ('val', dpo_val)):
            losses, accs, margins = [], [], []
            for _ in range(eval_iters):
                c, r, pl = make_dpo_batch(pool, cfg.batch_size, cfg.max_len, pad_id, device)
                loss, m = dpo_loss(model, ref_model, c, pl, r, pl, pad_id, beta=cfg.beta)
                losses.append(loss.item())
                accs.append(m['acc'])
                margins.append(m['reward_margin'])
            out[split] = {
                'loss': sum(losses) / len(losses),
                'acc':  sum(accs) / len(accs),
                'margin': sum(margins) / len(margins),
            }
        model.train()
        return out

    run_name = f'v10_dpo_{datetime.now().strftime("%m%d_%H%M")}'
    writer = SummaryWriter(TB_LOG_DIR / run_name)
    print(f'TB log: {TB_LOG_DIR / run_name}')
    print(f'cfg: steps={cfg.num_steps}, batch={cfg.batch_size}, lr={cfg.lr}, β={cfg.beta}\n')

    t0 = time.time()
    best_val = float('inf')

    for step in range(cfg.num_steps):
        lr = lr_at(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()                                # 训练时 Dropout 开
        c, r, pl = make_dpo_batch(dpo_train, cfg.batch_size, cfg.max_len, pad_id, device)
        loss, metrics = dpo_loss(model, ref_model, c, pl, r, pl, pad_id, beta=cfg.beta)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar('lr', lr, step)
        writer.add_scalar('train/loss_step', loss.item(), step)
        writer.add_scalar('train/acc',       metrics['acc'], step)
        writer.add_scalar('train/margin',    metrics['reward_margin'], step)
        writer.add_scalar('grad_norm', grad_norm.item(), step)

        if step % cfg.eval_interval == 0 or step == cfg.num_steps - 1:
            ev = estimate_dpo()
            writer.add_scalar('val/loss',   ev['val']['loss'],   step)
            writer.add_scalar('val/acc',    ev['val']['acc'],    step)
            writer.add_scalar('val/margin', ev['val']['margin'], step)

            tag = ''
            if ev['val']['loss'] < best_val:
                best_val = ev['val']['loss']
                save_checkpoint(
                    CKPT_DIR_DPO / 'best.pt',
                    step=step, model=model, optimizer=optimizer, scaler=scaler,
                    best_val=best_val, config=sft_cfg,
                )
                tag = ' [best]'
            elapsed = time.time() - t0
            eta = elapsed / max(step + 1, 1) * (cfg.num_steps - step - 1)
            print(
                f'step {step:4d} | lr {lr:.6f} '
                f'| train loss {loss.item():.4f} acc {metrics["acc"]:.2f} '
                f'| val loss {ev["val"]["loss"]:.4f} acc {ev["val"]["acc"]:.2f} '
                f'margin {ev["val"]["margin"]:.3f} '
                f'| {elapsed:.0f}s (ETA {eta:.0f}s){tag}'
            )

    writer.close()
    print(f'\n[DONE] DPO best val {best_val:.4f}  (理想 << 0.69:模型真的偏好 chosen)')


if __name__ == '__main__':
    main()
