"""MiniGPT v0.2:多头 Attention + FFN + LayerNorm + 残差(单 Block)。

──────────────────────────────────────────────────────────
【老师开讲】
v0.1 失败在"信息瓶颈太窄"。v0.2 一口气把缺的零件补齐,凑成一个完整的 Transformer Block:
  · **多头**:不再是一个窄头,而是几个头分工(见 layers.py 的比喻),信息不再被压扁;
  · **FFN**:注意力负责"在字与字之间搬运信息",FFN 负责"对每个字自己再加工一遍";
  · **LayerNorm + 残差**:让训练更稳、深了也不容易崩(细节在 layers.Block)。
这一版就把所有零件装进**一个** Block 里跑通;v0.3 再把 Block 叠成多层。

结果:val loss 终于压回到比 Bigram 更低,模型开始写出"像样"的句式。
(对应 courses/02_transformer_basics.ipynb)
──────────────────────────────────────────────────────────
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Block


class MiniGPTv02(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        # 相比 v01 的裸 SingleHead,这里换成一个完整 Block(多头+FFN+LN+残差)
        self.block   = Block(n_embed, n_head, block_size, dropout)
        self.ln_f    = nn.LayerNorm(n_embed)         # 输出前的最后一层归一化(final LN)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                              # (B, T, n_embed)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.block(tok_emb + pos_emb)            # 过一个 Transformer Block
        x = self.ln_f(x)                             # final LayerNorm
        logits = self.lm_head(x)                     # → (B, T, vocab_size)

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
            idx_cond = idx[:, -self.block_size:]         # 截断到 block_size(同 v01)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature      # 取最后一步 + 温度缩放
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 采样
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx
