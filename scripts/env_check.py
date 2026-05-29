"""检查 Python / PyTorch / 设备环境。"""
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.utils import amp_enabled, device_type, get_device


def main() -> None:
    print('=' * 50)
    print('环境检查')
    print('=' * 50)
    print(f'Python 版本: {sys.version.split()[0]}')
    print(f'PyTorch 版本: {torch.__version__}')
    print(f'操作系统: {platform.platform()}')
    print('-' * 50)

    dev = get_device()
    dt = device_type(dev)
    amp = '开 (fp16)' if amp_enabled(dev) else '关 (fp32)'
    print(f'选中设备: {dev}    混合精度 AMP: {amp}')

    if dt == 'cuda':
        print('[OK] CUDA 可用 (NVIDIA / Linux ROCm-AMD)')
        print(f'GPU 型号: {torch.cuda.get_device_name(0)}')
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU 显存: {mem_gb:.2f} GB')
        print(f'CUDA 版本: {torch.version.cuda}')
    elif dt == 'xpu':
        print('[OK] Intel XPU 可用')
        try:
            print(f'GPU 型号: {torch.xpu.get_device_name(0)}')
        except Exception:
            pass
    elif dt == 'mps':
        print('[OK] Apple MPS 可用')
    elif dt == 'privateuseone':
        print('[OK] DirectML 可用 (Windows 上的 AMD 卡 / Intel 核显),只能 fp32')
    else:
        print('[!] 没有可用 GPU,会用 CPU 训练(慢很多)')
        print('    Intel 独显/新核显: pip install torch --index-url https://download.pytorch.org/whl/xpu')
        print('    AMD 卡 / 老 Intel 核显: pip install torch-directml')

    print('=' * 50)


if __name__ == '__main__':
    main()
