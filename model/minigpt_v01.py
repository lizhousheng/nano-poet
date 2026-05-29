"""MiniGPT v0.1:Bigram + 单头 Self-Attention。

教学版本:SingleHead 单独定义在这里,Q/K/V 是三个独立的 Linear,
方便对照公式 attention(Q,K,V) = softmax(QK^T / √d) V。
真正训练用 v0.2/v0.3。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHead(nn.Module):
    """单头 Self-Attention,把 n_embed 投到 head_size。"""

    def __init__(self, n_embed: int, head_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key   = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)

        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return weights @ v


class MiniGPTv01(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, head_size: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.attention = SingleHead(n_embed, head_size, block_size)
        self.lm_head   = nn.Linear(head_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.attention(tok_emb + pos_emb)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx
