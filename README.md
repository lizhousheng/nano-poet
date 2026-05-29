# poetry-llm — 八周从零写 LLM

从字符级 Bigram 一步步搭到能押韵的小 Transformer,再到 SFT / DPO / QLoRA。
每一周都有明确的能力跃迁和数字目标。

## 课程总览

| Week | 主题 | 规模 / 关键技术 | 目标 val | 状态 |
|---|---|---|---|---|
| 1 | 数据 + Tokenizer | vocab=9,563(字符级) | — | ✅ |
| 2 | Bigram baseline | 91M(查表) | 5.43 | ✅ |
| 2 | 单头 Attention(v0.1) | 0.79M | 6.03(故意退步,看容量瓶颈) | ✅ |
| 2 | 多头+FFN+残差(v0.2) | 2.67M | 5.10(击穿 Bigram) | ✅ |
| 2 | 6 层 Transformer(v0.3) | 6.37M + GPT-2 风格初始化 | 4.14(开始押韵) | ✅ |
| 3 | + LR Schedule(v0.4) | warmup + cosine | 4.21~4.24 | ✅ |
| 3 | + AMP/grad clip/wd(v0.5) | 训练健康度全套 | < 4.21 | ✅ |
| 4 | + 完整 Checkpoint(v0.6) | best.pt / latest.pt | 同 v0.5 | ✅ |
| 4 | + Resume + TensorBoard(v0.7) | 断点续训 + 曲线监控 | 4.21 | ✅ |
| 5 | 扩规模 + 唐宋诗词(v0.8) | 25M / vocab 14k | 4.03(出现"金句") | ✅ |
| 6 | SFT 指令调优(v0.9) | 题目+风格→正文 | 3.55(SFT loss) | ✅ |
| 7 | DPO 偏好对齐(v0.10) | 自动打分 + RLHF | DPO loss < 0.69 | ✅ |
| 8 | QLoRA 微调 Qwen 1.5B | 4-bit + LoRA(可训 4.3M / 1.5B) | 对标商业模型 | ✅ |

## 目录结构

```
.
├── configs/config.py            所有超参 + 路径,一处改全部
├── data/                        数据 pipeline
│   ├── raw/   processed/        (运行脚本后生成)
│   ├── download.py   merge.py   clean.py   prepare.py      Week 1
│   ├── prepare_v2.py                                       Week 5: 加宋诗+宋词
│   ├── classify.py                                         Week 6: 体裁分类
│   ├── prepare_sft.py                                      Week 6: 构造 SFT 样本
│   └── prepare_dpo.py                                      Week 7: 自动标偏好对
├── tokenizer/
│   ├── tokenizer.py             CharTokenizer 类
│   ├── build.py                 Week 1: v1 (唐诗 only)
│   └── build_v3.py              Week 6: 加 SFT 特殊 token
├── model/
│   ├── layers.py                MultiHeadAttention / FFN / Block
│   ├── bigram.py                Week 2 baseline
│   ├── minigpt_v01.py           Week 2: 单头
│   ├── minigpt_v02.py           Week 2: 多头 + FFN + LN + 残差
│   └── minigpt_v03.py           Week 2: 多层 + GPT-2 初始化 (后续都用这个)
├── train/
│   ├── utils.py                 get_batch / estimate_loss / 基础训练循环
│   ├── lr_schedule.py           Week 3: warmup + cosine
│   ├── checkpoint.py            Week 4: save/load full state
│   ├── dpo_loss.py              Week 7: compute_logprobs + dpo_loss
│   ├── train_bigram.py          Week 2
│   ├── train_v01.py             Week 2
│   ├── train_v02.py             Week 2
│   ├── train_v03.py             Week 2
│   ├── train_v04.py             Week 3: + LR schedule
│   ├── train_v05.py             Week 3: + AMP + clip + wd
│   ├── train_v06.py             Week 4: + checkpoint
│   ├── train_v07.py             Week 4: + TensorBoard, --resume
│   ├── train_v08.py             Week 5: 25M 模型
│   ├── train_v09_sft.py         Week 6: vocab 扩展 + SFT
│   └── train_v10_dpo.py         Week 7: DPO
├── evaluation/
│   └── score.py                 Week 6/7: pypinyin 自动评分
├── inference/
│   ├── generate.py              Week 2: 通用 CLI(bigram / v01-v03)
│   ├── generate_sft.py          Week 6: 题目+风格→诗 + 12 测试套件
│   └── compare_dpo.py           Week 7: v09 vs v10 head-to-head
├── qlora/                       Week 8: QLoRA 微调 Qwen 1.5B
│   ├── load_qwen.py             4-bit (NF4) 加载
│   ├── zero_shot_baseline.py    微调前基线(6 题 + 4 个 英→中)
│   ├── prepare_data.py          Qwen chat 格式数据(v1 朴素 / v2 均衡)
│   ├── train.py                 LoRA 训练(--v2 继续训均衡数据)
│   └── generate.py              加载 LoRA → 写诗(带 torchao patch)
├── courses/                     **8 周文字课程**(.ipynb,边读边跑)
│   ├── 01_data_and_tokenizer.ipynb
│   ├── 02_transformer_basics.ipynb
│   ├── 03_optim_engineering.ipynb
│   ├── 04_checkpoint_resume.ipynb
│   ├── 05_scale_up_song_poetry.ipynb
│   ├── 06_sft_instruction_tuning.ipynb
│   ├── 07_dpo_preference_alignment.ipynb
│   ├── 08_qlora_qwen_finetune.ipynb
│   └── week2_self_test.md       Week 2 概念自测题库(30 道 Q&A)
├── checkpoints/                 各 Week 的权重(子目录:v06/ v07/ v08/ v09_sft/ v10_dpo/ qwen_lora_v1/ qwen_lora_v2/)
├── tb_logs/                     TensorBoard 日志(运行后生成)
└── scripts/
    ├── env_check.py             CUDA / GPU 检查
    ├── _course_helpers.py       课程 .ipynb 生成的公共工具
    └── build_courses_part{1,2}.py  课程内容源,改完跑一遍重生成 .ipynb
```

## 环境(用 uv 建 venv)

[uv](https://docs.astral.sh/uv/) 是 Astral 出的快速 Python 包管理器,比 pip 快一个数量级。
本项目用 `pyproject.toml` + `.python-version` + `uv.lock` 描述环境,
**最终落地的就是一个标准 .venv**,激活之后跟普通 Python 项目一模一样。

### 一次性准备

```powershell
# 1. 装 uv(只需一次)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 装基础依赖到 .venv/ (Week 1-7)
uv sync
#   ↑ 自动:
#     - 按 .python-version 拉 Python 3.12
#     - 在项目根目录建 .venv/
#     - 按 uv.lock 装 torch (cu128) + tensorboard + pypinyin + tqdm

# 3. Week 8 QLoRA 才需要,再补一次
uv sync --extra qlora        # 加装 transformers + peft + trl + bitsandbytes + ...
```

### 每次开工:激活 venv

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows cmd
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

激活后命令行提示符前面会有 `(poetry-llm)`,这时候 `python` / `pip` /
`tensorboard` 全都走项目 venv,跟普通 venv 完全一样。要退出就 `deactivate`。

### 常用 uv 命令

| 命令 | 作用 |
|---|---|
| `uv sync` | 按 lock 装/同步依赖到 .venv/ |
| `uv sync --extra qlora` | 同时装 qlora 可选依赖 |
| `uv add <pkg>` | 加新依赖到 pyproject.toml + 更新 lock |
| `uv add --optional qlora <pkg>` | 加到 qlora 可选组 |
| `uv lock --upgrade` | 升级所有锁定版本 |
| `uv pip list` | 看 venv 里装了什么(无需激活) |
| `uv run python xxx.py` | **不激活 venv 直接跑**(适合一次性命令) |

### 注意事项

- **CUDA 版本**:`pyproject.toml` 里默认走 `cu128`(PyTorch 2.10+)。显卡老一点
  改成 `cu124` / `cu121` / `cu118`,纯 CPU 跑就删掉 `[tool.uv.sources]` 整段。
- **Windows 上的 bitsandbytes**:0.43+ 已官方支持 Windows,直接 `uv sync --extra qlora`
  就能装上。装失败可以退路 Colab/Linux(T4 12GB 显存够 Week 8)。

## 推荐的学习路径

每周配套一个 `.ipynb` 课程,边读边跑代码。建议:

```powershell
# 激活 venv 后,进 courses/ 启动 Jupyter
.venv\Scripts\Activate.ps1
jupyter notebook courses/
```

每个课程结构相同(7 段):
1. **本周目标** —— 一句话 + val/规模对照表
2. **前置知识** —— 必备 + 不必备
3. **核心概念** —— 数学 / 直觉 / 代码片段,每节带最小可跑 demo
4. **代码地图** —— 本周改动到哪几个文件,按读代码顺序列
5. **动手做** —— `subprocess.run([...])` 一键跑训练
6. **自测题** —— 不给答案,自己脑子里答(Week 2 答案在 `week2_self_test.md`)
7. **容易踩的坑** —— 我们项目里实际踩过的 5 个左右

**重新生成课程**:课程内容的源头是 `scripts/build_courses_part{1,2}.py`,
改完跑一遍重生成 `.ipynb`:

```powershell
python scripts/build_courses_part1.py    # Week 1-4
python scripts/build_courses_part2.py    # Week 5-8
```

---

## 完整跑一遍(估计耗时,以 T4 为基准)

**前提**:已经 `uv sync` 装好依赖,并 `.venv\Scripts\Activate.ps1` 激活 venv
(看到提示符前面有 `(poetry-llm)`)。下面的 `python` 都走项目 venv。

```powershell
# ===== 准备 =====
python scripts/env_check.py
python data/prepare.py                              # ~1 min(Week 1 数据)

# ===== Week 2:四个模型,Loss 一步步下降 =====
python train/train_bigram.py                        # ~30s     val ~5.43
python train/train_v01.py                           # ~1 min   val ~6.03 (容量瓶颈)
python train/train_v02.py                           # ~2 min   val ~5.10
python train/train_v03.py                           # ~5 min   val ~4.14

# ===== Week 3:工程化训练 =====
python train/train_v04.py                           # ~8 min   + LR schedule
python train/train_v05.py                           # ~6 min   + AMP/clip/wd

# ===== Week 4:Checkpoint + Resume =====
python train/train_v06.py                           # ~6 min
python train/train_v07.py                           # ~5 min   + TensorBoard
python train/train_v07.py --resume                  # ~5 min   续训

# ===== Week 5:扩规模 + 加宋诗宋词 =====
python data/prepare_v2.py                           # ~3 min   25M 字符
python train/train_v08.py                           # ~40 min  val ~4.03

# ===== Week 6:SFT 指令调优 =====
python tokenizer/build_v3.py                        # 加特殊 token
python data/prepare_sft.py                          # ~30s     ~37k 样本
python train/train_v09_sft.py                       # ~10 min

# 看看 v0.9 写诗
python inference/generate_sft.py --title 春雨 --style 五言绝句
python inference/generate_sft.py --battery          # 12 道测试题

# ===== Week 7:DPO 偏好对齐 =====
python data/prepare_dpo.py                          # ~30 min  自动生成 + 打分
python train/train_v10_dpo.py                       # ~5 min
python inference/compare_dpo.py                     # v0.9 vs v0.10

# ===== Week 8:QLoRA 微调 Qwen 1.5B(需先 uv sync --extra qlora)=====
python qlora/zero_shot_baseline.py                  # ~5 min   微调前基线
python qlora/prepare_data.py                        # v1 数据(2000 朴素)
python qlora/train.py                               # ~15 min  v1 LoRA
python qlora/prepare_data.py --balanced             # v2 数据(6000 均衡)
python qlora/train.py --v2                          # ~30 min  v2 继续训(基于 v1)

python qlora/generate.py --battery                  # 8 题套件
python qlora/generate.py --memorize                 # 看是否一字不差背静夜思
python qlora/generate.py --prompt "写一首关于 996 的七言绝句"

# ===== 全程实时看曲线 =====
tensorboard --logdir tb_logs
```

## 学习路径表(对应 index.ipynb 的迭代历史)

```
Week 1   数据 + Tokenizer       9,563 vocab       →  无模型,搭好 pipeline
Week 2   Bigram(无 Attention)  91M               →  val 5.43(纯查表上限)
Week 2   单头 Attention         0.79M             →  val 6.03(反而退步,看到容量瓶颈)
Week 2   多头+FFN+残差          2.67M             →  val 5.10(击穿 Bigram)
Week 2   6 层 Transformer       6.37M             →  val 4.14(开始押韵)
Week 3   LR 调度                6.37M             →  val 4.21~4.24(训练健康度)
Week 3   AMP + grad clip + wd  6.37M             →  显存省一半,可上更大 batch
Week 4   Checkpoint            6.37M             →  断点不丢
Week 4   Resume + TensorBoard  6.37M             →  曲线可视化、续训
Week 5   扩规模 + 唐宋诗词      25M / 14k vocab  →  val ~4.03(出现真正"金句")
Week 6   SFT 指令调优           25M               →  能按"题目+风格"生成
Week 7   DPO 偏好对齐           25M               →  RLHF 概念落地,质量再升
Week 8   QLoRA 微调 Qwen        1.5B → 4.36M 可训 → 真正能与商业模型比较
```

## 设计取舍

- 字符级 tokenizer 而非 BPE:学习成本最低,vocab 增长可控。
- v03 之后所有模型共用一个 `MiniGPTv03` 类,只改超参 —— Week 5/6/7 都靠它。
- prompt mask:SFT 把 prompt 部分的 y 设为 -100,只在答案上算 loss。
- vocab 扩展:Week 6 从 v0.8(vocab=14477)迁到 v0.9(vocab=14482),前 14477 行
  embedding/lm_head 拷贝,后 5 行随机初始化 —— 旧字能力一字不丢。
- DPO 自动标注:用规则评分器(句数/字数/押韵/题目相关/重复)给候选打分,
  代替昂贵的人工 RLHF。注意这套评分能力上限就是评分器的智力上限。
- DPO 训练时模型也开 train(Dropout 开),但 ref_model 永远 eval —— 评估时
  两个都 eval,这样 ref==policy 的 DPO loss 必须严格等于 -log(0.5)=0.6931。
- **QLoRA 的关键细节**:LoRA 补丁手动转 FP32,SFTConfig 关掉 fp16/bf16 ——
  4-bit 基础 + FP32 LoRA 的纯净混合精度。开 AMP 反而会跟 4-bit 量化撞车。
- Qwen LoRA v2 基于 v1 继续训,lr 减半(2e-4 → 1e-4)防止破坏 v1 学到的格律。
