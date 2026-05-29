"""把全唐诗的 JSON 合并成一份原始 txt。"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import POETRY_REPO_DIR, RAW_TEXT_FILE


def merge() -> None:
    pattern = str(POETRY_REPO_DIR / '全唐诗' / 'poet.tang.*.json')
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        raise FileNotFoundError(f'找不到 JSON,先运行 download.py。pattern={pattern}')

    print(f'准备处理 {len(json_files)} 个文件')
    poems: list[str] = []
    for filepath in json_files:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        for poem in data:
            paragraphs = poem.get('paragraphs', [])
            if paragraphs:
                poems.append('\n'.join(paragraphs))

    text = '\n\n'.join(poems)
    RAW_TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_TEXT_FILE.write_text(text, encoding='utf-8')

    print(f'共 {len(poems)} 首诗,{len(text):,} 字符 -> {RAW_TEXT_FILE}')


if __name__ == '__main__':
    merge()
