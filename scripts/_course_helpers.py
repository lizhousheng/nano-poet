"""Notebook 生成的公共工具。课程内容定义在 build_courses_part1/2.py。"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / 'courses'
OUT_DIR.mkdir(exist_ok=True)

NB_META = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'mimetype': 'text/x-python', 'file_extension': '.py'},
}


def md(*lines: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': '\n'.join(lines)}


def code(*lines: str) -> dict:
    return {
        'cell_type': 'code', 'execution_count': None,
        'metadata': {}, 'outputs': [],
        'source': '\n'.join(lines),
    }


def save(filename: str, cells: list[dict]) -> None:
    nb = {'cells': cells, 'metadata': NB_META, 'nbformat': 4, 'nbformat_minor': 5}
    out = OUT_DIR / filename
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'  {filename}: {len(cells)} cells, {out.stat().st_size / 1024:.1f} KB')


# 每周课程通用的"恢复 / 准备环境"代码,放在课程开头
PATH_SETUP = (
    "import sys, pathlib",
    "ROOT = pathlib.Path().resolve().parent  # courses/ 的上一级",
    "sys.path.insert(0, str(ROOT))",
)
