"""Week 8 · QLoRA 微调 Qwen2.5-1.5B-Instruct。

两种用法:
    python qlora/train.py                # v1:朴素 2000 样本,2 epoch,lr 2e-4
    python qlora/train.py --v2            # v2:均衡 6000 样本,3 epoch,lr 1e-4
                                          # 默认会基于 v1 LoRA 权重继续训

关键设计:
1. NF4 4-bit 加载基础权重 → 显存从 3GB 降到 ~1.2GB
2. LoRA 只在 q/k/v/o_proj 上加 r=16 的低秩补丁 → 可训参数 ~4.3M
3. **关键修复**:LoRA 补丁手动转 FP32,SFTConfig 关闭 fp16/bf16 ——
   这样 4-bit 基础 + FP32 LoRA 的纯净配置,避免 GradScaler 报错
4. paged_adamw_8bit 优化器 + gradient_checkpointing 进一步压显存
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    QLoRACfg,
    QWEN_LORA_V1_DIR,
    QWEN_LORA_V2_DIR,
    QWEN_SFT_TRAIN_DIR,
    QWEN_SFT_TRAIN_V2,
    QWEN_SFT_VAL_DIR,
    QWEN_SFT_VAL_V2,
    TB_LOG_DIR,
)
from qlora.load_qwen import load_qwen_4bit


def attach_lora_or_resume(model, resume_dir: Path | None):
    """全新套 LoRA,或者从已有 LoRA 目录继续训。"""
    cfg = QLoRACfg
    model = prepare_model_for_kbit_training(model)
    if resume_dir and resume_dir.exists():
        print(f'[resume] 从 {resume_dir} 加载 LoRA 继续训')
        model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
    else:
        print('[fresh] 套全新 LoRA')
        lora_cfg = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)

    # 关键修复:LoRA 补丁转 FP32,基础权重保持 4-bit
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'可训参数: {trainable / 1e6:.2f}M / 总 {total / 1e9:.2f}B  '
          f'({trainable / total * 100:.3f}%)')
    return model


def build_training_args(output_dir: Path, *, is_v2: bool) -> SFTConfig:
    cfg = QLoRACfg
    return SFTConfig(
        output_dir=str(output_dir),

        num_train_epochs=cfg.num_train_epochs_v2 if is_v2 else cfg.num_train_epochs_v1,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,

        learning_rate=cfg.learning_rate_v2 if is_v2 else cfg.learning_rate_v1,
        lr_scheduler_type='cosine',
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,

        # 关键:4-bit 基础 + FP32 LoRA 模式,关掉 AMP
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,

        logging_steps=cfg.logging_steps,
        eval_strategy='steps',
        eval_steps=cfg.eval_steps,
        save_strategy='steps',
        save_steps=cfg.save_steps,
        save_total_limit=2,

        max_length=cfg.max_length,
        packing=False,
        optim='paged_adamw_8bit',

        report_to='tensorboard',
        logging_dir=str(TB_LOG_DIR / output_dir.name),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v2', action='store_true',
                        help='用均衡数据继续训,基于 v1 LoRA')
    args = parser.parse_args()

    if args.v2:
        train_dir = QWEN_SFT_TRAIN_V2
        val_dir   = QWEN_SFT_VAL_V2
        out_dir   = QWEN_LORA_V2_DIR
        resume    = QWEN_LORA_V1_DIR
    else:
        train_dir = QWEN_SFT_TRAIN_DIR
        val_dir   = QWEN_SFT_VAL_DIR
        out_dir   = QWEN_LORA_V1_DIR
        resume    = None

    assert train_dir.exists(), f'找不到 {train_dir},先跑 qlora/prepare_data.py' + (' --balanced' if args.v2 else '')

    model, tokenizer = load_qwen_4bit()
    model = attach_lora_or_resume(model, resume)

    train_ds = load_from_disk(str(train_dir))
    val_ds   = load_from_disk(str(val_dir))
    print(f'train={len(train_ds)}, val={len(val_ds)}')

    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = build_training_args(out_dir, is_v2=args.v2)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    eff_batch  = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    epochs     = training_args.num_train_epochs
    est_steps  = len(train_ds) * epochs // eff_batch
    print(f'有效 batch={eff_batch}, 预计步数 ~{est_steps}, lr={training_args.learning_rate}\n')

    t0 = time.time()
    trainer.train()
    print(f'\n[DONE] 训练用时 {(time.time() - t0) / 60:.1f} 分钟')

    trainer.save_model(str(out_dir))
    total = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    print(f'LoRA 权重 -> {out_dir}  ({total / 1e6:.2f} MB,对比 base 3GB)')


if __name__ == '__main__':
    main()
