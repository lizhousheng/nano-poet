"""Bigram baseline:一张 (vocab_size, vocab_size) 的查找表。

完全没有上下文,只看上一个字预测下一个字。作为后续所有模型的 loss 下界。

──────────────────────────────────────────────────────────
【老师开讲】
打个比方:Bigram 就像一本"词语接龙小抄"。它死记硬背"看到『春』后面最常接什么字",
但完全不看前文 —— 哪怕前面写的是"春天"还是"伤春",它的预测一模一样。

它怎么记的?就一张表 lut[i][j] = "上一个字是 i 时,下一个字是 j 的打分"。
  · 假设全世界只有 5 个字(vocab_size=5),这张表就是 5×5;
  · 真实项目里 vocab_size≈9563,所以表是 9563×9563 ≈ 9100 万个数 —— 参数量惊人,
    却笨得可怜,这正好说明"光靠背、不看上下文"是行不通的。

为什么还要写它?它是个"地板":后面任何带 attention 的模型,loss 都该比它低,
否则就是写错了。它是我们衡量进步的尺子。(详见 courses/02_transformer_basics.ipynb 第 2.1 节)
──────────────────────────────────────────────────────────
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # 查找表的每一行就是"看到第 i 个字时,下一个字的打分(logits)"。
        # 行数=列数=vocab_size,所以参数量是 vocab_size^2。
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,                       # (B, T) 一批 token id,B=batch,T=序列长度
        targets: torch.Tensor | None = None,     # (B, T) 每个位置的"正确下一个字",训练时才传
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 直接查表:idx 里每个 id 取出对应那一行 → (B, T, vocab_size) 的打分
        logits = self.token_embedding(idx)
        if targets is None:                      # 推理:只要打分,不算 loss
            return logits, None
        # cross_entropy 要求形状是 (N, C) 和 (N,),所以把 batch 和时间维拍平
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()                             # 生成不需要梯度,省显存也更快
    def generate(
        self,
        idx: torch.Tensor,                       # (B, T) 当前已有的上文
        max_new_tokens: int,                     # 要续写多少个字
        temperature: float = 1.0,                # >1 更随机,<1 更保守,=1 原样
    ) -> torch.Tensor:
        self.eval()                              # 关 Dropout 等训练专属行为
        for _ in range(max_new_tokens):
            # Bigram 只看最后一个字,所以只喂 idx 的最后一列
            logits, _ = self(idx[:, -1:])
            # 取最后一个时间步的打分,除以温度调节随机性
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)            # 打分 → 概率分布
            next_id = torch.multinomial(probs, num_samples=1)  # 按概率采样一个字
            idx = torch.cat([idx, next_id], dim=1)       # 拼到序列末尾,继续下一轮
        self.train()                             # 恢复训练模式
        return idx
