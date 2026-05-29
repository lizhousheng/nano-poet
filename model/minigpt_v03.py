"""MiniGPT v0.3:在 v0.2 基础上堆 n_layer 层 Block —— 最接近 GPT 结构的版本。

──────────────────────────────────────────────────────────
【老师开讲】
v0.2 只有一层 Block,只能学到"一步"的关系。v0.3 把 Block 像流水线工序一样叠 n_layer 层:
第 1 层先认出"这俩字搭配",第 2 层在此基础上认出"这半句的语气",越往上越抽象 ——
这种"逐层组合"正是深度网络的威力,也是 GPT 名字里 "Transformer" 的标准骨架。

唯一的新代码就两处:① 用 nn.Sequential 把 n_layer 个 Block 串起来;
② 加了 GPT-2 风格的小权重初始化(深网络对初始值敏感,见 _init_weights 注释)。
除此之外,forward 和 generate 跟 v0.2 一模一样。

结果:val loss 4.14,模型开始押得上韵、写出有意境的句子。这就是我们要长期训练的那个结构。
(对应 courses/02_transformer_basics.ipynb)
──────────────────────────────────────────────────────────
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Block


class MiniGPTv03(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        # 唯一相比 v02 的结构变化:把 n_layer 个 Block 串起来。层数越多,模型越能学到
        # 多步、组合性的依赖关系 —— 这就是 GPT 的基本骨架。
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f    = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # GPT-2 风格初始化:深层网络对初始权重更敏感,统一用 std=0.02 的小正态分布,
        # 让初始激活/梯度幅度可控,训练一开始更稳定。
        self.apply(self._init_weights)              # apply 会递归作用到每个子模块

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)         # bias 从 0 开始
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                              # (B, T, n_embed)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)           # 依次穿过 n_layer 个 Block
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
            idx_cond = idx[:, -self.block_size:]         # 截断到 block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature      # 取最后一步 + 温度缩放
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 按概率采样
            idx = torch.cat([idx, next_id], dim=1)       # 自回归:把新字接上去再续写
        self.train()
        return idx
