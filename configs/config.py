"""项目所有超参数与路径常量。

所有脚本/模块都从这里 import,避免硬编码路径。
"""
from pathlib import Path

# ===== 路径 =====
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / 'data'
RAW_DIR        = DATA_DIR / 'raw'
PROCESSED_DIR  = DATA_DIR / 'processed'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

POETRY_REPO_URL = 'https://github.com/chinese-poetry/chinese-poetry.git'
POETRY_REPO_DIR = RAW_DIR / 'chinese-poetry'

RAW_TEXT_FILE   = RAW_DIR / 'tang_poems_raw.txt'
CLEAN_TEXT_FILE = PROCESSED_DIR / 'tang_poems_clean.txt'
TOKENIZER_FILE  = PROCESSED_DIR / 'tokenizer.pkl'
TRAIN_DATA_FILE = PROCESSED_DIR / 'train.pt'
VAL_DATA_FILE   = PROCESSED_DIR / 'val.pt'

# ===== Week 5: v2 数据(唐 + 宋诗 + 宋词)=====
CLEAN_TEXT_V2_FILE = PROCESSED_DIR / 'poems_v2_clean.txt'
TOKENIZER_V2_FILE  = PROCESSED_DIR / 'tokenizer_v2.pkl'
TRAIN_DATA_V2_FILE = PROCESSED_DIR / 'train_v2.pt'
VAL_DATA_V2_FILE   = PROCESSED_DIR / 'val_v2.pt'

# ===== Week 6: v3 tokenizer(加 SFT 特殊 token)+ SFT 数据 =====
TOKENIZER_V3_FILE = PROCESSED_DIR / 'tokenizer_v3.pkl'
SFT_DATA_FILE     = PROCESSED_DIR / 'sft_data.pt'
SPECIAL_TOKENS    = ['<|题目|>', '<|风格|>', '<|开始|>', '<|结束|>', '<|pad|>']

# ===== Week 7: DPO 偏好对 =====
DPO_DATA_FILE = PROCESSED_DIR / 'dpo_data.pt'

# ===== Week 8: QLoRA Qwen 微调 =====
QWEN_MODEL_NAME      = 'Qwen/Qwen2.5-1.5B-Instruct'
QWEN_SFT_TRAIN_DIR   = PROCESSED_DIR / 'qwen_sft_train'        # v1 朴素 2000
QWEN_SFT_VAL_DIR     = PROCESSED_DIR / 'qwen_sft_val'
QWEN_SFT_TRAIN_V2    = PROCESSED_DIR / 'qwen_sft_train_v2'     # v2 均衡 6000
QWEN_SFT_VAL_V2      = PROCESSED_DIR / 'qwen_sft_val_v2'
QWEN_LORA_V1_DIR     = CHECKPOINT_DIR / 'qwen_lora_v1'
QWEN_LORA_V2_DIR     = CHECKPOINT_DIR / 'qwen_lora_v2'

# ===== TensorBoard =====
TB_LOG_DIR = PROJECT_ROOT / 'tb_logs'

# ===== 数据切分 =====
TRAIN_RATIO = 0.9

# ===== 通用训练超参 =====
BATCH_SIZE    = 32
BLOCK_SIZE    = 128
LEARNING_RATE = 3e-4
DROPOUT       = 0.1
EVAL_ITERS    = 20
EVAL_INTERVAL = 500
NUM_STEPS     = 5000

# ===== 各模型结构超参 =====
class BigramCfg:
    """纯查找表,无任何隐层结构"""
    learning_rate = 1e-2          # Bigram 没有梯度链路问题,lr 可以大
    num_steps     = 6000


class MiniGPTv01Cfg:
    """单头 Self-Attention"""
    n_embed    = 64
    head_size  = 16
    block_size = BLOCK_SIZE


class MiniGPTv02Cfg:
    """多头 + FFN + LayerNorm + 残差(单 Block)"""
    n_embed    = 128
    n_head     = 4
    block_size = BLOCK_SIZE
    dropout    = DROPOUT


class MiniGPTv03Cfg:
    """多层 Block stack —— 最接近真实 GPT 架构的版本"""
    n_embed    = 192
    n_head     = 6
    n_layer    = 6
    block_size = BLOCK_SIZE
    dropout    = DROPOUT


class MiniGPTv08Cfg:
    """Week 5 扩规模:25M 参数 + block_size 翻倍 + 唐宋数据"""
    n_embed    = 384
    n_head     = 6
    n_layer    = 8
    block_size = 256
    dropout    = 0.1


# ===== Week 3 / 4 通用训练超参 =====
class TrainV05Cfg:
    """v04+: schedule;v05: + AMP + grad clip + weight decay"""
    num_steps     = 8000
    warmup_steps  = 200
    max_lr        = 3e-4
    min_lr        = 3e-5
    weight_decay  = 0.1
    grad_clip     = 1.0
    eval_interval = 500


# ===== Week 5 v0.8 训练超参 =====
class TrainV08Cfg:
    num_steps           = 10000
    warmup_steps        = 500
    max_lr              = 3e-4
    min_lr              = 3e-5
    weight_decay        = 0.1
    grad_clip           = 1.0
    eval_interval       = 500
    checkpoint_interval = 1000
    batch_size          = 32


# ===== Week 6 SFT 训练超参 =====
class SFTCfg:
    num_steps     = 3000
    warmup_steps  = 100
    max_lr        = 5e-5            # 微调 lr 比预训小 6 倍
    min_lr        = 5e-6
    weight_decay  = 0.1
    grad_clip     = 1.0
    batch_size    = 32
    max_len       = 128
    eval_interval = 200


# ===== Week 7 DPO 训练超参 =====
class DPOCfg:
    num_steps     = 500
    warmup_steps  = 50
    lr            = 1e-5
    beta          = 0.1             # DPO 温度
    batch_size    = 4
    max_len       = 128
    eval_interval = 50
    n_candidates  = 4               # 每个 prompt 生成几首作为候选
    min_score_gap = 5               # chosen vs rejected 至少分差


# ===== Week 8 QLoRA(对 Qwen 1.5B)=====
class QLoRACfg:
    """LoRA + 4-bit 量化基础"""
    # LoRA
    rank            = 16
    alpha           = 32
    target_modules  = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    lora_dropout    = 0.05

    # SFT 数据
    samples_v1          = 2000          # v1 朴素采样
    samples_per_style   = 1500          # v2 每种风格采样

    # Trainer (v1)
    num_train_epochs_v1   = 2
    learning_rate_v1      = 2e-4
    train_val_ratio_v1    = 0.95

    # Trainer (v2 继续训)
    num_train_epochs_v2   = 3
    learning_rate_v2      = 1e-4
    train_val_ratio_v2    = 0.97

    # 共用
    batch_size            = 2            # T4 安全 batch
    gradient_accumulation = 4            # 有效 batch = 8
    weight_decay          = 0.01
    warmup_ratio          = 0.05
    max_length            = 256
    save_steps            = 500
    eval_steps            = 200
    logging_steps         = 50
