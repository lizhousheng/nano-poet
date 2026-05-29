"""Transformer 通用构件:多头注意力、FFN、Block。

v0.2 和 v0.3 共用这些层;v0.1 用的是教学版的 SingleHead,放在 minigpt_v01.py 里。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """合并 QKV 的多头自注意力 + causal mask。"""

    def __init__(self, n_embed: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embed % n_head == 0, 'n_embed 必须能被 n_head 整除'
        self.n_head = n_head
        self.head_size = n_embed // n_head

        self.qkv  = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    """两层 MLP,中间 4 倍扩展。"""

    def __init__(self, n_embed: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer Block:LN -> Attn -> 残差 -> LN -> FFN -> 残差。"""

    def __init__(self, n_embed: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embed)
        self.ffn  = FeedForward(n_embed, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
