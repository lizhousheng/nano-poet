"""一键跑完整数据 pipeline:download -> merge -> clean -> build tokenizer -> 编码切分。

跑完这一个脚本就能得到 train.pt / val.pt / tokenizer.pkl,后面训练直接 load。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.clean import clean
from data.download import download
from data.merge import merge
from tokenizer.build import build_and_encode


def main() -> None:
    print('=' * 60); print('[1/4] 下载数据'); print('=' * 60)
    download()

    print('\n' + '=' * 60); print('[2/4] 合并 JSON'); print('=' * 60)
    merge()

    print('\n' + '=' * 60); print('[3/4] 清洗噪音'); print('=' * 60)
    clean()

    print('\n' + '=' * 60); print('[4/4] 构建 tokenizer + 编码切分'); print('=' * 60)
    build_and_encode()

    print('\n[DONE] pipeline 全部完成')


if __name__ == '__main__':
    main()
