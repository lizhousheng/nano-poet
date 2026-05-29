"""检查 Python / PyTorch / CUDA 环境。"""
import platform
import sys

import torch


def main() -> None:
    print('=' * 50)
    print('环境检查')
    print('=' * 50)
    print(f'Python 版本: {sys.version.split()[0]}')
    print(f'PyTorch 版本: {torch.__version__}')
    print(f'操作系统: {platform.platform()}')
    print('-' * 50)

    if torch.cuda.is_available():
        print('[OK] CUDA 可用')
        print(f'GPU 型号: {torch.cuda.get_device_name(0)}')
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU 显存: {mem_gb:.2f} GB')
        print(f'CUDA 版本: {torch.version.cuda}')
    else:
        print('[!] 没有 GPU,会用 CPU 训练(慢很多)')

    print('=' * 50)


if __name__ == '__main__':
    main()
