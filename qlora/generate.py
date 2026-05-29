"""Week 8 · 加载 base Qwen + LoRA adapter,写诗。

用法:
    python qlora/generate.py --prompt "写一首五言绝句,主题春天"
    python qlora/generate.py --battery     # 6 题对照套件
    python qlora/generate.py --memorize    # 续写测试(看是否记住训练集)
    python qlora/generate.py --lora v1     # 用 v1 LoRA (默认 v2)

注意:加载 LoRA 时打补丁绕过 peft 的 torchao 检测(版本兼容问题)。
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import QWEN_LORA_V1_DIR, QWEN_LORA_V2_DIR, QWEN_MODEL_NAME


def _patch_peft_torchao() -> None:
    """有些 peft 版本启动时会探测 torchao 导致报错,直接 monkey-patch 关掉。"""
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    if hasattr(peft.import_utils.is_torchao_available, 'cache_clear'):
        peft.import_utils.is_torchao_available.cache_clear()
    try:
        import peft.tuners.lora.torchao as _t
        _t.is_torchao_available = lambda: False
    except (ImportError, AttributeError):
        pass


def load_lora_model(lora_dir: Path):
    _patch_peft_torchao()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f'加载 tokenizer ({lora_dir})...')
    tokenizer = AutoTokenizer.from_pretrained(str(lora_dir))

    print(f'加载 base {QWEN_MODEL_NAME} (bf16)...')
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
    )
    print(f'加载 LoRA adapter ({lora_dir})...')
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_poem(model, tokenizer, prompt: str,
                  max_new: int = 200, temperature: float = 0.7,
                  top_p: float = 0.9) -> str:
    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new,
        temperature=temperature, top_p=top_p, do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


BATTERY = [
    '写一首五言绝句,主题春天',
    '写一首七言绝句,主题是秋夜思乡',
    '写一首关于明月的五言律诗',
    '以江南烟雨为题,作一首七言绝句',
    '写一首送别友人的诗',
    '写一首关于地铁的五言绝句',
    '以咖啡为题作一首七言绝句',
    '写一首关于程序员加班的诗',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--prompt', help='单条 prompt 直接生成')
    parser.add_argument('--battery', action='store_true', help='跑 8 题套件')
    parser.add_argument('--memorize', action='store_true',
                        help='续写《静夜思》前两句,看是否一字不差背诵(过拟合预警)')
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()

    lora_dir = QWEN_LORA_V2_DIR if args.lora == 'v2' else QWEN_LORA_V1_DIR
    assert lora_dir.exists(), f'找不到 LoRA {lora_dir},先跑 qlora/train.py' + (' --v2' if args.lora == 'v2' else '')

    model, tokenizer = load_lora_model(lora_dir)
    print(f'\nLoRA: {lora_dir}\n')

    if args.memorize:
        prompt = '续写:床前明月光,疑是地上霜。'
        print(f'[memorize 测试,T=0.3]\nprompt: {prompt}')
        print(generate_poem(model, tokenizer, prompt, temperature=0.3))
        print('\n→ 如果一字不差输出"举头望明月,低头思故乡",说明开始背诵了')
        return

    if args.battery:
        for i, p in enumerate(BATTERY, 1):
            print(f'\n【{i}】 {p}')
            print('-' * 60)
            try:
                print(generate_poem(model, tokenizer, p, temperature=args.temperature))
            except Exception as e:
                print(f'FAIL: {e}')
        return

    assert args.prompt, '需要 --prompt / --battery / --memorize 三选一'
    print(generate_poem(model, tokenizer, args.prompt, temperature=args.temperature))


if __name__ == '__main__':
    main()
