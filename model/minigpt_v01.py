"""MiniGPT v0.1:Bigram + 单头 Self-Attention。

教学版本:SingleHead 单独定义在这里,Q/K/V 是三个独立的 Linear,
方便对照公式 attention(Q,K,V) = softmax(QK^T / √d) V。
真正训练用 v0.2/v0.3。

──────────────────────────────────────────────────────────
【老师开讲】
这一版故意写得"最朴素",就为了让你把注意力公式看透:Q/K/V 三个 Linear 分开写,
一一对应公式里的三个矩阵。看懂这里,再去看 layers.py 的多头(把它们合成一个 Linear)。

两个第一次出现的新东西:
1) **位置编码**(position_embedding)。注意力本身是"无序"的 —— 打乱字的顺序,算出来
   一样,因为它只看 Q·K 配不配对,不看谁在前谁在后。可"白日依山"和"山日白依"意思
   天差地别!所以我们额外给每个"位置"也学一个向量,和字向量**相加**,把顺序信息注进去。
   (为什么是"加"而不是"拼"?加不增加维度、最省事,而且效果就很好,是业界惯例。)
2) **单头注意力**。只有一个头、head_size 还故意调得很小(16 维)。

⚠️ 重要:v0.1 是个"失败案例"!它的 val loss 反而比 Bigram 还高。
原因是 16 维这个"信息瓶颈"太窄,把字义压扁了。这不是 bug,是教学设计 ——
亲眼看它失败、并搞懂"为什么加了 attention 反而更差",比直接给你成功代码学得多。
(完整复盘见 courses/02_transformer_basics.ipynb 第 2.2 节)
──────────────────────────────────────────────────────────
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHead(nn.Module):
    """单头 Self-Attention,把 n_embed 投到 head_size。"""

    def __init__(self, n_embed: int, head_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        # Q/K/V 各用一个独立 Linear(教学版故意分开写,方便对照公式)
        self.query = nn.Linear(n_embed, head_size, bias=False)   # 我在找什么
        self.key   = nn.Linear(n_embed, head_size, bias=False)   # 我能提供什么
        self.value = nn.Linear(n_embed, head_size, bias=False)   # 我实际携带的信息
        # causal mask:下三角为 1,保证位置 t 只能看到 0..t
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)      # 各 (B, T, head_size)

        # QK^T / √d:每个位置对其它位置的相关度打分,(B, T, T)
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 屏蔽未来
        weights = F.softmax(scores, dim=-1)                      # 归一化成关注度
        return weights @ v                                       # 加权求和 V → (B, T, head_size)


class MiniGPTv01(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, head_size: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embed)   # 每个字 → 一个向量(字义)
        self.position_embedding = nn.Embedding(block_size, n_embed)   # 每个位置 → 一个向量(位置)
        self.attention = SingleHead(n_embed, head_size, block_size)
        self.lm_head   = nn.Linear(head_size, vocab_size)             # 把特征投回 vocab 上打分

    def forward(
        self,
        idx: torch.Tensor,                       # (B, T)
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                              # (B, T, n_embed) 字义
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embed) 位置
        # 字义 + 位置 相加(广播):自注意力本身不分先后,靠位置向量注入顺序信息
        x = self.attention(tok_emb + pos_emb)
        logits = self.lm_head(x)                                        # (B, T, vocab_size)

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
            # 关键:位置编码只到 block_size,所以上文超长时要截断只保留最近 block_size 个
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature      # 只看最后一步的预测,温度调随机性
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 按概率采样下一个字
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx
