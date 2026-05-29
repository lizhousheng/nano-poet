"""Week 7 · DPO loss 实现。

DPO 公式:
    L = -log σ( β · ( logπ(chosen)/πref(chosen) - logπ(rejected)/πref(rejected) ) )

直觉:让当前 policy 在 chosen 上的相对 log-prob 比 rejected 高(相对于 ref),
这样无需训练独立的 reward model 就能逼近 RLHF 的效果。
"""
import torch
import torch.nn.functional as F

from train.utils import autocast_ctx


def compute_logprobs(
    model: torch.nn.Module,
    ids: torch.Tensor,
    prompt_lens: list[int] | torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """对 (B, T) 序列,返回 (B,) 个"答案部分"的总 log-prob。

    用 autocast fp16 forward,gather/log_softmax 升回 fp32 保证数值稳定。
    """
    device = ids.device
    with autocast_ctx(device):
        logits, _ = model(ids)

    logits  = logits[:, :-1, :].float()                       # 错位:logits[t] 预测 ids[t+1]
    targets = ids[:, 1:]                                      # (B, T-1)

    log_probs = F.log_softmax(logits, dim=-1)
    per_token = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (B, T-1)

    # 构造 mask:答案部分=1,prompt 部分=0,pad 也=0
    B, Tm1 = per_token.shape
    mask = torch.zeros_like(per_token)
    for i, plen in enumerate(prompt_lens):
        start = max(int(plen) - 1, 0)
        mask[i, start:] = 1.0
    mask = mask * (targets != pad_id).float()
    return (per_token * mask).sum(dim=-1)


def dpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    chosen_ids: torch.Tensor,
    chosen_prompt_lens: list[int] | torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_prompt_lens: list[int] | torch.Tensor,
    pad_id: int,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_lp   = compute_logprobs(model, chosen_ids,   chosen_prompt_lens,   pad_id)
    rejected_lp = compute_logprobs(model, rejected_ids, rejected_prompt_lens, pad_id)
    with torch.no_grad():
        ref_chosen_lp   = compute_logprobs(ref_model, chosen_ids,   chosen_prompt_lens,   pad_id)
        ref_rejected_lp = compute_logprobs(ref_model, rejected_ids, rejected_prompt_lens, pad_id)

    chosen_ratio   = chosen_lp   - ref_chosen_lp
    rejected_ratio = rejected_lp - ref_rejected_lp

    margin = beta * (chosen_ratio - rejected_ratio)
    loss   = -F.logsigmoid(margin).mean()

    with torch.no_grad():
        metrics = {
            'reward_chosen':   (beta * chosen_ratio).mean().item(),
            'reward_rejected': (beta * rejected_ratio).mean().item(),
            'reward_margin':   margin.mean().item(),
            'acc':             (chosen_ratio > rejected_ratio).float().mean().item(),
        }
    return loss, metrics
