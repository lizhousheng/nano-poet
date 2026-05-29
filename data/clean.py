"""清洗原始唐诗文本:去掉数字/符号噪音,压缩多余换行。"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import CLEAN_TEXT_FILE, RAW_TEXT_FILE

NOISE_CHARS = set('0123456789+-/=[]{}|·…□○●⻊⿰')


def clean() -> None:
    raw = RAW_TEXT_FILE.read_text(encoding='utf-8')
    cleaned = ''.join(c for c in raw if c not in NOISE_CHARS)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    CLEAN_TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_TEXT_FILE.write_text(cleaned, encoding='utf-8')

    print(f'清洗前: {len(raw):,} 字符, vocab={len(set(raw))}')
    print(f'清洗后: {len(cleaned):,} 字符, vocab={len(set(cleaned))}')
    print(f'-> {CLEAN_TEXT_FILE}')


if __name__ == '__main__':
    clean()
