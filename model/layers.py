"""Transformer 通用构件:多头注意力、FFN、Block。

v0.2 和 v0.3 共用这些层;v0.1 用的是教学版的 SingleHead,放在 minigpt_v01.py 里。

──────────────────────────────────────────────────────────
【老师开讲:注意力到底在干嘛?】
想象一桌人在聊天。轮到某个字"发言"前,它会做两件事:
  · 举一块牌 Q(Query):"我想听听关于 ___ 的信息";
  · 同时每个字都举一块牌 K(Key):"我这儿有 ___ 的信息"。
谁的 Q 和谁的 K 对得上,就多听谁的 V(Value,那个字真正携带的内容)。
"自注意力"的"自",就是 Q/K/V 全都来自同一句话自己。

"多头"是什么?就是同时开好几桌,每桌只关注一个侧面 —— 一桌盯"谁和谁押韵",
一桌盯"谁是主语",最后把各桌的结论拼起来。这样比一桌包打天下学得更细。

【一遍具体数字走查】(对着下面 MultiHeadAttention.forward 看)
假设:1 句话 B=1、长度 T=4 个字、每字向量 C=8 维、开 n_head=2 个头(每头 head_size=4)。
  x          : (1, 4, 8)     输入,4 个字各 8 维
  qkv(x)     : (1, 4, 24)    一次算出 Q/K/V,切成三份各 (1,4,8)
  拆头后 q/k/v: (1, 2, 4, 4)  把 8 维劈给 2 个头,每头 4 维
  scores     : (1, 2, 4, 4)  每个头一张 4×4 "谁该关注谁"的表
  (加 mask + softmax,把"看未来"的格子清零并归一化)
  out = w@v  : (1, 2, 4, 4)  按关注度把各字的 V 加权求和
  拼回       : (1, 4, 8)     两个头拼回 8 维,形状跟输入一样,可以继续往上叠

更深入的公式推导 / 为什么 v0.1 单头会"失败",见 courses/02_transformer_basics.ipynb。
──────────────────────────────────────────────────────────
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """合并 QKV 的多头自注意力 + causal mask。"""

    def __init__(self, n_embed: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # 多头:把 n_embed 维切成 n_head 份,每个头在 head_size 维的子空间里独立做注意力,
        # 不同头可以关注不同的模式(谁押韵、谁是主语…),最后拼回来。
        assert n_embed % n_head == 0, 'n_embed 必须能被 n_head 整除'
        self.n_head = n_head
        self.head_size = n_embed // n_head

        # 一个 Linear 同时算出 Q、K、V(输出 3*n_embed,再切三份),比三个 Linear 高效
        self.qkv  = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)          # 多头拼接后再混合一次

        # 下三角矩阵(causal mask):第 t 个位置只能看 0..t,看不到未来。注册成 buffer
        # 表示它是常量、跟着模型搬设备,但不是要训练的参数。
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_dropout  = nn.Dropout(dropout)         # 随机丢弃注意力权重
        self.resid_dropout = nn.Dropout(dropout)         # 残差连接前再 dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                                # batch, 序列长度, 通道数(=n_embed)

        # 一次算出 Q/K/V,再沿最后一维切成三块,各 (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=-1)
        # 拆出多头并把"头"维提前:(B, T, C) → (B, n_head, T, head_size)
        # 这样后面的矩阵乘法会在每个头上独立并行进行。
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 注意力打分 = Q·Kᵀ,(..., T, head_size) @ (..., head_size, T) → (B, n_head, T, T)
        # 除以 √head_size 是为了让点积方差稳定,避免 softmax 进入梯度极小的饱和区。
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        # 把未来位置(上三角)填 -inf,softmax 后就变成 0 → 实现"只能看过去"
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)              # 每一行归一化成概率:对各位置的关注度
        weights = self.attn_dropout(weights)

        # 用关注度对 V 加权求和:(B, n_head, T, T) @ (B, n_head, T, head_size)
        out = weights @ v
        # 把多头拼回去:(B, n_head, T, head_size) → (B, T, C)。contiguous 是因为 transpose
        # 只改了视图,view 需要内存连续。
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    """两层 MLP,中间 4 倍扩展。

    注意力负责"在不同位置之间搬运信息",FFN 负责"对每个位置自己的特征做非线性加工"。
    中间放大到 4 倍是 Transformer 的惯例,给模型更大的表达空间。
    """

    def __init__(self, n_embed: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),     # 升维
            nn.GELU(),                           # 非线性激活
            nn.Linear(4 * n_embed, n_embed),     # 降回原维度
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer Block:LN -> Attn -> 残差 -> LN -> FFN -> 残差。

    "Pre-LN"指 LayerNorm 放在子层之前(而非之后),配合残差连接能让深层网络更稳定、更好训。
    残差 x + f(x) 让梯度有一条"高速公路"直达浅层,缓解深层退化。
    """

    def __init__(self, n_embed: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embed)        # 注意力前的归一化
        self.attn = MultiHeadAttention(n_embed, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embed)        # FFN 前的归一化
        self.ffn  = FeedForward(n_embed, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))           # 先归一化再注意力,结果加回 x(残差)
        x = x + self.ffn(self.ln2(x))            # 同理,FFN 也走残差
        return x
