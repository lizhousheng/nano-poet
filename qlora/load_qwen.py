"""Week 8 · 4-bit (NF4) 量化加载 Qwen2.5-1.5B-Instruct。

3GB 模型只占 ~1.2GB 显存。后续 LoRA / SFT / 推理共用这个加载函数。

依赖:
    pip install transformers>=4.45 accelerate bitsandbytes peft trl datasets
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from configs.config import QWEN_MODEL_NAME


def load_qwen_4bit(model_name: str = QWEN_MODEL_NAME):
    """返回 (model, tokenizer),model 是 NF4 4-bit。"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
    )
    return model, tokenizer


def main() -> None:
    print(f'下载 {QWEN_MODEL_NAME}(首次约 3GB,2-5 分钟)...')
    model, tokenizer = load_qwen_4bit()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n[DONE] 模型加载完成')
    print(f'  参数量: {n_params / 1e9:.2f}B')
    print(f'  显存:   {torch.cuda.memory_allocated() / 1e9:.2f} GB')
    print(f'  vocab:  {tokenizer.vocab_size}')


if __name__ == '__main__':
    main()
