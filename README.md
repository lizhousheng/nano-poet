# nano-poet — 八周从零写 LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://docs.astral.sh/uv/)
[![Course](https://img.shields.io/badge/Course-8%20Weeks-green.svg)](./courses/)

[English](./README.en.md) · 简体中文

> 致敬 [nanoGPT](https://github.com/karpathy/nanoGPT) 的教学型小模型流派,用全唐诗 / 全宋诗 / 宋词当语料,带你从字符级 Bigram 一路走到 QLoRA 微调 Qwen 1.5B。**8 个文字课程 notebook 配套全部代码**,边读边跑。

## 效果预览(各版本同一类 prompt)

| Week | 模型 | 输入 | 输出片段 |
|---|---|---|---|
| 2 | Bigram (91M 查表) | "月" 起始 | `月、之之、其，之之...`(单字接龙) |
| 2 | v0.2 单 Block | "月" 起始 | `月度好。芳期恨雲，蓬壺漱楊磨...`(有句式但语义乱) |
| 2 | v0.3 6 层 | "月" 起始 | `月光透碧簾，輕雲度遠峰。客舟橫水靜，孤雁過天空。`(押韵 + 意境) |
| 6 | v0.9 SFT | 题目=春雨 / 风格=五绝 | `春雨潤晴川，東風拂柳煙。新苗承露重，遠岫接雲連。` |
| 7 | v0.10 DPO | 同上 | 押韵更稳、题目相关字出现更多 |
| 8 | Qwen 1.5B + LoRA | 同上 | `春雨绵绵润万家，柳丝轻拂燕双斜。一犁新水耕烟绿，几树残红傍水涯。`(流畅、用典) |

📈 **val loss 演进**: Bigram 5.43 → v0.3 4.14 → v0.8 4.03 → v0.9 SFT 3.55 → Qwen LoRA(对照商业级)

---

## 📚 课程导航 — 从这里开始读

每周一个 `.ipynb` 课程,标准深度教学(~700 行/篇),理论 + 代码 + 动手 + 自测 + 踩坑五件套。
**推荐学习顺序就是 01 → 08**,每周收尾前给下周做铺垫。

| | 课程 | 一句话本周做什么 | 关键产出 |
|---|---|---|---|
| **Week 1** | [📓 01_data_and_tokenizer](./courses/01_data_and_tokenizer.ipynb) | 唐诗 JSON → 字符级 tokenizer → train.pt / val.pt,搭好数据 pipeline | 9,563 vocab |
| **Week 2** | [📓 02_transformer_basics](./courses/02_transformer_basics.ipynb) | Bigram → 单头 → 多头+FFN+残差 → 6 层,看 loss 一步步降 | val **4.14** |
| | [📝 week2_self_test](./courses/week2_self_test.md) | Transformer 30 道概念自测题(带答案) | — |
| **Week 3** | [📓 03_optim_engineering](./courses/03_optim_engineering.ipynb) | 加 LR schedule + AMP + grad clip + weight decay,训练健康度全面升级 | 显存省一半 |
| **Week 4** | [📓 04_checkpoint_resume](./courses/04_checkpoint_resume.ipynb) | 完整 checkpoint(model + optim + scaler + step)+ 断点续训 + TensorBoard | 工程闭环 |
| **Week 5** | [📓 05_scale_up_song_poetry](./courses/05_scale_up_song_poetry.ipynb) | 加宋诗+宋词,数据 4M→22M,模型 6.37M→25M,出现真正"金句" | val **4.03** |
| **Week 6** | [📓 06_sft_instruction_tuning](./courses/06_sft_instruction_tuning.ipynb) | SFT 指令微调,特殊 token + prompt mask + vocab 扩展时的权重迁移 | val **3.55** |
| **Week 7** | [📓 07_dpo_preference_alignment](./courses/07_dpo_preference_alignment.ipynb) | 规则评分器自动标偏好对 + DPO loss,无需 RLHF 也能做对齐 | DPO loss < 0.69 |
| **Week 8** | [📓 08_qlora_qwen_finetune](./courses/08_qlora_qwen_finetune.ipynb) | 4-bit 量化加载 Qwen 1.5B + LoRA(4.3M 可训),对标商业级 | T4 上 30 min |

每个课程的结构(7 段):

```
0. 本周目标          一句话 + val/规模对照表
1. 前置知识          必备 + 不必备
2. 核心概念          数学 / 直觉 / 代码片段,每节带最小可跑 demo
3. 代码地图          本周改动到哪几个文件,按读代码顺序
4. 动手做            subprocess.run([...]) 一键跑训练
5. 自测题            不给答案,自己脑子里答
6. 容易踩的坑        我们项目里实际踩过的 5 个左右
```

---

## ⚡ 快速开始(5 分钟跑出第一首诗)

```powershell
# 1. 装 uv(只需一次)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 装依赖 + 激活 venv
uv sync
.venv\Scripts\Activate.ps1               # Linux/Mac: source .venv/bin/activate

# 3. 一键跑数据准备 + Week 2 最终模型(T4 约 5 分钟)
python data/prepare.py
python train/train_v03.py

# 4. 让它写首诗
python inference/generate.py --model v03 --start 月 --tokens 150
```

**接下来**:打开 [`courses/01_data_and_tokenizer.ipynb`](./courses/01_data_and_tokenizer.ipynb),
跟着课程一步步推进。

```powershell
jupyter notebook courses/
```

---

## 环境(用 uv 建 venv)

[uv](https://docs.astral.sh/uv/) 是 Astral 出的快速 Python 包管理器,比 pip 快一个数量级。
本项目用 `pyproject.toml` + `.python-version` + `uv.lock` 描述环境,**最终落地的就是一个标准 .venv**,
激活之后跟普通 Python 项目一模一样。

### 一次性准备

```powershell
# 基础依赖(Week 1-7)
uv sync
#   ↑ 自动:
#     - 按 .python-version 拉 Python 3.12
#     - 在项目根目录建 .venv/
#     - 按 uv.lock 装 torch (cu128) + tensorboard + pypinyin + tqdm

# Week 8 QLoRA 才需要,再补一次
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

激活后命令行提示符前面会有 `(nano-poet)`,这时候 `python` / `pip` / `tensorboard` 全都走项目 venv。
要退出就 `deactivate`。

### 常用 uv 命令

| 命令 | 作用 |
|---|---|
| `uv sync` | 按 lock 装/同步依赖到 .venv/ |
| `uv sync --extra qlora` | 同时装 qlora 可选依赖 |
| `uv add <pkg>` | 加新依赖到 pyproject.toml + 更新 lock |
| `uv lock --upgrade` | 升级所有锁定版本 |
| `uv run python xxx.py` | **不激活 venv 直接跑**(适合一次性命令) |

### 注意事项

- **CUDA 版本**:`pyproject.toml` 里默认走 `cu128`(PyTorch 2.10+)。显卡老一点改成
  `cu124` / `cu121` / `cu118`,纯 CPU 跑就删掉 `[tool.uv.sources]` 整段。
- **Windows 上的 bitsandbytes**:0.43+ 已官方支持 Windows,直接 `uv sync --extra qlora`
  就能装上。装失败可以退路 Colab / ModelScope。

---

## 在 Colab 上跑(海外推荐,免费 T4)

[Google Colab](https://colab.research.google.com/) 提供免费 T4 GPU(15 GB 显存),
跑完前 7 周绰绰有余,Week 8 QLoRA 也能跑(batch_size 降到 1)。

```python
# ===== 1. 检查 GPU =====
!nvidia-smi --query-gpu=name,memory.total --format=csv

# ===== 2. clone 项目 =====
!git clone https://github.com/<你的用户名>/nano-poet.git
%cd nano-poet

# ===== 3. 装依赖(Colab 自带 PyTorch + CUDA,不用 uv,直接 pip)=====
!pip install -q pypinyin tqdm tensorboard
# Week 8 才装这一行(2 GB,首次 5-10 分钟)
# !pip install -q transformers>=4.45 accelerate bitsandbytes peft trl datasets

# ===== 4. 挂 Drive(强烈推荐:防止运行时断开权重全丢)=====
from google.colab import drive
drive.mount('/content/drive')

# 把 checkpoints/ 软链接到 Drive
!mkdir -p /content/drive/MyDrive/nano-poet
!rm -rf checkpoints && ln -s /content/drive/MyDrive/nano-poet checkpoints

# ===== 5. 打开课程边读边跑 =====
# 在 Colab 文件浏览器里双击 courses/01_data_and_tokenizer.ipynb 即可

# ===== 6. 或者直接跑训练 =====
!python data/prepare.py
!python train/train_v03.py

%load_ext tensorboard
%tensorboard --logdir tb_logs
```

**Colab 上的注意事项**:
- 免费版**90 分钟无操作会断开** → 长训练前一定挂 Drive 存 checkpoint
- T4 跑 v0.8(25M)约 40 分钟,v0.9 SFT 约 10 分钟
- Week 8 推荐 Pro 版的 V100/A100,T4 上 batch_size 必须降到 1

## 在 ModelScope 上跑(国内推荐,网速快)

[ModelScope](https://www.modelscope.cn/) 是阿里出的模型社区,提供**免费 GPU 在线 Notebook**
(A10 24GB / V100 32GB)+ 模型仓库托管。**国内用户访问 GitHub / HuggingFace 慢的话,推荐这个**。

```bash
# 1. 注册 https://www.modelscope.cn/ → 启动 GPU Notebook(新用户每周免费 36 小时)

# 2. clone + 装包(阿里镜像源)
git clone https://github.com/<你的用户名>/nano-poet.git
cd nano-poet
pip install -i https://mirrors.aliyun.com/pypi/simple/ pypinyin tqdm tensorboard
# Week 8: pip install -i https://mirrors.aliyun.com/pypi/simple/ transformers accelerate bitsandbytes peft trl datasets

# 3. 打开 courses/ 学习,或直接 python data/prepare.py → train/train_v03.py
```

**Qwen 模型从 ModelScope 拉(比 HuggingFace 快 10×)**:改 `qlora/load_qwen.py` 顶部为
`from modelscope import AutoModelForCausalLM, AutoTokenizer` + `MODEL_NAME = "qwen/Qwen2.5-1.5B-Instruct"`。
或者保持原代码,设环境变量 `export HF_ENDPOINT=https://hf-mirror.com` 走镜像。

**训完推回 ModelScope Hub** 让别人下载:

```bash
pip install modelscope[framework]
modelscope login --token <你的_token>
modelscope upload --model_id <你的用户名>/nano-poet-qwen-lora --local_path checkpoints/qwen_lora_v2
```

**注意事项**:
- 实例关机后数据**不保留** → 重要文件下载到本地
- 跑完 Week 8 别忘了**主动停机**省额度
- A10 24GB 比 Colab T4 大,Week 8 batch_size 可以开到 4

---

## 课程总览(技术规格速查表)

| Week | 主题 | 规模 / 关键技术 | 目标 val |
|---|---|---|---|
| 1 | 数据 + Tokenizer | vocab=9,563(字符级) | — |
| 2 | Bigram baseline | 91M(查表) | 5.43 |
| 2 | 单头 Attention(v0.1) | 0.79M | 6.03(故意退步,看容量瓶颈) |
| 2 | 多头+FFN+残差(v0.2) | 2.67M | 5.10(击穿 Bigram) |
| 2 | 6 层 Transformer(v0.3) | 6.37M + GPT-2 风格初始化 | 4.14(开始押韵) |
| 3 | + LR Schedule(v0.4) | warmup + cosine | 4.21~4.24 |
| 3 | + AMP/grad clip/wd(v0.5) | 训练健康度全套 | < 4.21 |
| 4 | + 完整 Checkpoint(v0.6) | best.pt / latest.pt | 同 v0.5 |
| 4 | + Resume + TensorBoard(v0.7) | 断点续训 + 曲线监控 | 4.21 |
| 5 | 扩规模 + 唐宋诗词(v0.8) | 25M / vocab 14k | 4.03(出现"金句") |
| 6 | SFT 指令调优(v0.9) | 题目+风格→正文 | 3.55(SFT loss) |
| 7 | DPO 偏好对齐(v0.10) | 自动打分 + RLHF | DPO loss < 0.69 |
| 8 | QLoRA 微调 Qwen 1.5B | 4-bit + LoRA(可训 4.3M / 1.5B) | 对标商业模型 |

## 完整跑一遍(命令清单,以 T4 为基准)

**前提**:已激活 venv(看到提示符前面有 `(nano-poet)`)。**只是查命令的话不必跑这个清单 —— 学习时跟着课程 notebook 走更自然**。

```powershell
# ===== 准备 =====
python scripts/env_check.py
python data/prepare.py                              # ~1 min(Week 1)

# ===== Week 2:四个模型,Loss 一步步下降 =====
python train/train_bigram.py                        # ~30s     val ~5.43
python train/train_v01.py                           # ~1 min   val ~6.03(容量瓶颈)
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
python qlora/generate.py --memorize                 # 是否一字不差背静夜思
python qlora/generate.py --prompt "写一首关于 996 的七言绝句"

# ===== 全程实时看曲线 =====
tensorboard --logdir tb_logs
```

---

## 代码目录结构

```
nano-poet/
├── courses/                     ← 8 周文字课程(.ipynb),入口在这里!
│   ├── 01_data_and_tokenizer.ipynb
│   ├── 02_transformer_basics.ipynb
│   ├── 03_optim_engineering.ipynb
│   ├── 04_checkpoint_resume.ipynb
│   ├── 05_scale_up_song_poetry.ipynb
│   ├── 06_sft_instruction_tuning.ipynb
│   ├── 07_dpo_preference_alignment.ipynb
│   ├── 08_qlora_qwen_finetune.ipynb
│   └── week2_self_test.md       Week 2 概念自测题库(30 Q&A)
│
├── configs/config.py            所有超参 + 路径,一处改全部
├── data/                        数据 pipeline
│   ├── download.py              git clone chinese-poetry
│   ├── merge.py + clean.py + prepare.py   Week 1
│   ├── prepare_v2.py            Week 5: 加宋诗+宋词
│   ├── classify.py              Week 6: 五/七绝律分类
│   ├── prepare_sft.py           Week 6: SFT 样本构造
│   └── prepare_dpo.py           Week 7: 偏好对自动标注
├── tokenizer/
│   ├── tokenizer.py             CharTokenizer 类
│   ├── build.py                 Week 1: v1 (唐诗 only)
│   └── build_v3.py              Week 6: 加 SFT 特殊 token
├── model/
│   ├── layers.py                MultiHeadAttention / FFN / Block(共享)
│   ├── bigram.py                Week 2 baseline
│   ├── minigpt_v01.py           Week 2: 单头
│   ├── minigpt_v02.py           Week 2: 多头 + FFN + LN + 残差
│   └── minigpt_v03.py           Week 2: 多层 + GPT-2 初始化(后续 Week 5-7 都用它)
├── train/
│   ├── utils.py                 get_batch / estimate_loss
│   ├── lr_schedule.py           Week 3: warmup + cosine
│   ├── checkpoint.py            Week 4: save/load full state
│   ├── dpo_loss.py              Week 7: compute_logprobs + dpo_loss
│   └── train_*.py               Week 2-7 每个版本一个训练入口
├── evaluation/score.py          Week 6/7: pypinyin 自动评分
├── inference/
│   ├── generate.py              Week 2: 通用 CLI(bigram / v01-v03)
│   ├── generate_sft.py          Week 6: 题目+风格→诗
│   └── compare_dpo.py           Week 7: v09 vs v10 head-to-head
├── qlora/                       Week 8: QLoRA 微调 Qwen 1.5B
│   ├── load_qwen.py             4-bit (NF4) 加载
│   ├── zero_shot_baseline.py    微调前基线
│   ├── prepare_data.py          Qwen chat 格式数据
│   ├── train.py                 LoRA 训练
│   └── generate.py              加载 LoRA → 写诗
├── checkpoints/                 各 Week 的权重(子目录 v06/ v07/ v08/ v09_sft/ v10_dpo/ qwen_lora_v*/)
├── tb_logs/                     TensorBoard 日志(运行后生成)
└── scripts/
    ├── env_check.py             CUDA / GPU 检查
    ├── _course_helpers.py       课程 .ipynb 生成的公共工具
    └── build_courses_part{1,2}.py    课程内容源,改完跑一遍重生成 .ipynb
```

---

## 设计取舍(为什么这么搭)

- **字符级 tokenizer 而非 BPE**:学习成本最低,vocab 增长可控。等理解透了再换 BPE 不晚。
- **v03 之后所有模型共用一个 `MiniGPTv03` 类,只改超参** —— Week 5/6/7 都靠它。架构演进到 v0.3 就定型了,后面拼数据 + LoRA。
- **prompt mask**:SFT 把 prompt 部分的 y 设为 -100,只在答案上算 loss,模型不会误学"如何续写题目"。
- **vocab 扩展**:Week 6 从 v0.8(vocab=14477)迁到 v0.9(vocab=14482),前 14477 行 embedding/lm_head 拷贝,后 5 行随机初始化 —— 旧字能力一字不丢。
- **DPO 自动标注**:用规则评分器(句数/字数/押韵/题目相关/重复)给候选打分,代替昂贵的人工 RLHF。注意这套评分能力上限就是评分器的智力上限。
- **DPO 训练时**模型也开 train(Dropout 开),但 ref_model 永远 eval —— 评估时两个都 eval,这样 ref==policy 的 DPO loss 必须严格等于 -log(0.5)=0.6931。
- **QLoRA 的关键细节**:LoRA 补丁手动转 FP32,SFTConfig 关掉 fp16/bf16 —— 4-bit 基础 + FP32 LoRA 的纯净混合精度。开 AMP 反而会跟 4-bit 量化撞车。
- **Qwen LoRA v2 基于 v1 继续训,lr 减半**(2e-4 → 1e-4)防止破坏 v1 学到的格律。

---

## 致谢

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy —— 教学型小模型的精神祖宗
- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) —— 全唐诗 / 全宋诗 / 宋词的开源语料
- [Qwen](https://github.com/QwenLM/Qwen) by 阿里通义千问 —— Week 8 的基座模型

## License

[MIT](./LICENSE)
