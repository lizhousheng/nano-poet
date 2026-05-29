"""Week 8 · Qwen2.5-1.5B 零样本写诗能力测试(微调前基线)。

同样的 6 道题目,用来跟 v0.9 / v0.10 / LoRA 微调后做公平对比。
另外测一下英→中诗,看 Qwen 的跨语言能力。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlora.load_qwen import load_qwen_4bit

POETRY_CASES = [
    ('春雨', '五言绝句'),
    ('观棋', '七言绝句'),
    ('独居', '五言律诗'),
    ('送别', '七言律诗'),
    ('梅花', '五言绝句'),
    ('饮酒', '七言律诗'),
]

EN_PROMPTS = [
    'A lonely traveler at sunset, missing home far away.',
    'Spring rain falls quietly on plum blossoms in a deserted garden.',
    'Two old friends parting at a riverside, knowing they may never meet again.',
    'An old man drinking wine alone under the moon, recalling his youth.',
]


@torch.no_grad()
def qwen_generate(model, tokenizer, user_msg: str, max_new: int = 200,
                  temperature: float = 0.7, top_p: float = 0.9) -> str:
    messages = [{'role': 'user', 'content': user_msg}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new,
        do_sample=True, temperature=temperature, top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def main() -> None:
    model, tokenizer = load_qwen_4bit()

    print('=' * 70)
    print('Qwen2.5-1.5B 零样本写诗(无微调)')
    print('=' * 70)
    for title, style in POETRY_CASES:
        prompt = (f'请以「{title}」为题,写一首{style}。'
                  f'严格遵守字数和句数,讲究意境与押韵。直接输出诗,不要解释。')
        print(f'\n[{title} · {style}]')
        print(qwen_generate(model, tokenizer, prompt))

    print('\n' + '=' * 70)
    print('Qwen2.5-1.5B 英→中诗(零样本)')
    print('=' * 70)
    for en in EN_PROMPTS:
        prompt = (
            'Read this English description and write a Chinese classical poem '
            '(Tang Dynasty style) that captures its essence.\n'
            'Format: 4 lines, each line either 5 or 7 characters, with rhyme.\n'
            'Output the Chinese poem only, no explanation.\n\n'
            f'Description: {en}'
        )
        print(f'\nEN: {en}')
        print(qwen_generate(model, tokenizer, prompt))


if __name__ == '__main__':
    main()
