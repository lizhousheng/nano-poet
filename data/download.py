"""下载 chinese-poetry 数据集。"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import POETRY_REPO_DIR, POETRY_REPO_URL, RAW_DIR


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if POETRY_REPO_DIR.exists():
        print(f'仓库已存在,跳过 clone: {POETRY_REPO_DIR}')
        return

    print(f'clone {POETRY_REPO_URL} -> {POETRY_REPO_DIR}')
    subprocess.run(
        ['git', 'clone', '--depth', '1', POETRY_REPO_URL, str(POETRY_REPO_DIR)],
        check=True,
    )
    print('下载完成')


if __name__ == '__main__':
    download()
