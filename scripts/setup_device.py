"""探测显卡厂商,自动装对的 PyTorch 后端(全自动模式)。

用法:
    python scripts/setup_device.py            # 只检测 + 打印建议命令(不动环境)
    python scripts/setup_device.py --apply    # 检测 + 实际安装

也被 train_*.py 在 `import torch` 之前调用 auto_setup() 实现"训练时自动装"。

关键约束:本模块**绝不在顶层 import torch**。
安装 torch 后端(尤其 torch-directml 要替换 torch)必须在 torch 尚未加载时进行;
否则 Windows 下 torch 的 DLL 被当前进程占用,pip 无法覆盖 → 安装失败。
因此:命令行/全新进程能全自动装成;notebook 里 kernel 已加载 torch 时装不动,
会优雅回退并打印手动命令,绝不让训练崩。
"""
import argparse
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

# 装不动时写个标记,避免每次跑都重试(删掉它即可重新尝试)
_MARKER = Path(__file__).resolve().parent.parent / '.device_setup_attempted'


def _gpu_names() -> list[str]:
    """问操作系统要显卡型号列表(不依赖 torch)。"""
    sysname = platform.system()
    try:
        if sysname == 'Windows':
            out = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 '(Get-CimInstance Win32_VideoController).Name'],
                capture_output=True, text=True, encoding='utf-8', timeout=20)
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        if sysname == 'Linux':
            out = subprocess.run(['lspci'], capture_output=True, text=True, timeout=20)
            return [ln for ln in out.stdout.splitlines() if 'VGA' in ln or '3D' in ln]
        if sysname == 'Darwin':
            return ['Apple Silicon']
    except Exception:
        pass
    return []


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def detect() -> dict:
    """返回 {names, vendor, backend, install}。

    backend ∈ cuda / xpu / directml / mps / cpu。
    install 为 pip 命令(list)或 None(无需装 / 无法自动装)。
    """
    names = _gpu_names()
    j = ' '.join(names).lower()
    nvidia = any(k in j for k in ('nvidia', 'geforce', 'rtx', 'gtx', 'quadro', 'tesla'))
    amd    = any(k in j for k in ('amd', 'radeon'))
    intel  = 'intel' in j
    apple  = platform.system() == 'Darwin'

    if nvidia:   # cu 版 torch 自带 CUDA,装好驱动即可,无需额外装
        return dict(names=names, vendor='NVIDIA', backend='cuda', install=None)
    if apple:    # Apple 芯片走 MPS,torch 自带
        return dict(names=names, vendor='Apple', backend='mps', install=None)
    if amd:      # Windows 上 AMD 唯一通路是 DirectML,有干净的 pip 包
        return dict(names=names, vendor='AMD', backend='directml',
                    install=[sys.executable, '-m', 'pip', 'install', 'torch-directml'])
    if intel:    # 新 Intel(Arc/酷睿Ultra)走 XPU;替换整个 torch build,风险大,只建议不自动装
        return dict(names=names, vendor='Intel', backend='xpu',
                    install=[sys.executable, '-m', 'pip', 'install', 'torch',
                             '--index-url', 'https://download.pytorch.org/whl/xpu'])
    return dict(names=names, vendor='Unknown', backend='cpu', install=None)


def _cmd_str(cmd: list[str]) -> str:
    return ' '.join(c if c != sys.executable else 'python' for c in cmd)


def _print_manual(info: dict) -> None:
    if info['install']:
        print(f"[setup] 请手动安装 {info['backend']} 后端(建议用独立 venv,避免降级当前 torch):")
        print(f"        {_cmd_str(info['install'])}")
    else:
        print(f"[setup] {info['vendor']} 无需额外安装(backend={info['backend']})。")


def _run_install(cmd: list[str]) -> bool:
    print(f"[setup] 执行: {_cmd_str(cmd)}")
    try:
        return subprocess.run(cmd).returncode == 0
    except Exception as e:
        print(f"[setup] 安装出错: {e}")
        return False


def auto_setup(verbose: bool = True) -> None:
    """train_*.py 在 `import torch` 之前调用:缺后端就自动装,装不动就回退 CPU。

    自动安装目前仅对 DirectML(AMD / 老 Intel 核显)启用 —— 它是替换风险最小、
    有独立 pip 包的情形。NVIDIA/Apple 无需装;Intel XPU 要换整个 torch build,
    风险太大,只打印建议不自动执行。
    可用环境变量 NANO_POET_NO_AUTOSETUP=1 关闭本机制。
    """
    if os.environ.get('NANO_POET_NO_AUTOSETUP') == '1':
        return
    info = detect()
    backend = info['backend']

    if backend in ('cuda', 'mps', 'cpu'):       # 无需自动装
        return
    if backend == 'directml':
        if _has('torch_directml'):               # 已装好,直接用
            return
        if _MARKER.exists():                     # 上次装不动,别每次重试
            if verbose:
                print('[setup] 之前自动安装 DirectML 未成功(删除 .device_setup_attempted 可重试)。')
                _print_manual(info)
            return
        if verbose:
            print(f"[setup] 检测到 {info['vendor']} 显卡但缺 DirectML 后端,尝试自动安装 ...")
        ok = _run_install(info['install'])
        if ok and _has('torch_directml'):
            if verbose:
                print('[setup] DirectML 安装成功,本次训练将使用显卡。')
        else:
            _MARKER.write_text('directml install failed\n', encoding='utf-8')
            if verbose:
                print('[setup] 自动安装失败(常见原因:当前/其它进程已加载 torch,DLL 被占用)。')
                print('        将回退 CPU 继续。建议在独立 venv 手动安装:')
                _print_manual(info)
        return
    if backend == 'xpu':                          # 不自动替换 torch build,只建议
        if verbose:
            print(f"[setup] 检测到 {info['vendor']} 显卡。XPU 需替换 torch build,未自动执行:")
            _print_manual(info)


def main() -> None:
    parser = argparse.ArgumentParser(description='探测显卡并(可选)安装对应 PyTorch 后端')
    parser.add_argument('--apply', action='store_true', help='实际执行安装(默认只打印命令)')
    args = parser.parse_args()

    info = detect()
    print('=' * 50)
    print('显卡探测')
    print('=' * 50)
    print('显卡:', '; '.join(info['names']) or '(未探测到)')
    print('厂商:', info['vendor'])
    print('建议后端:', info['backend'])
    print('-' * 50)

    if not info['install']:
        _print_manual(info)
        return
    if args.apply:
        _run_install(info['install'])
    else:
        print('建议执行(加 --apply 可直接安装;建议用独立 venv 以免降级当前 torch):')
        print(f"  {_cmd_str(info['install'])}")


if __name__ == '__main__':
    main()
