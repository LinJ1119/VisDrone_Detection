"""
文件操作工具函数。

提供 Windows 原子写入、图像可读性校验、递归目录创建。
"""

import os
from pathlib import Path
from PIL import Image


def ensure_dir(path):
    """递归创建目录（等效 mkdir -p）。

    Args:
        path: 目录路径（str 或 Path），可含多级父目录
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_write(filepath, data):
    """先写临时文件 → os.replace 原子替换，防止写入中断导致文件损坏。

    Windows 上 os.replace 是同磁盘原子操作，避免 os.rename 的非原子问题
    （先删目标再重命名）。参见概要设计 §5 风险九。

    Args:
        filepath: 最终目标文件路径
        data: 写入内容（str 或 bytes）
    """
    filepath = Path(filepath)
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")

    mode = "wb" if isinstance(data, bytes) else "w"
    encoding = None if isinstance(data, bytes) else "utf-8"

    with open(tmp_path, mode, encoding=encoding) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, filepath)


def verify_image(path):
    """校验图像文件是否可读取（不加载完整像素数据，仅验证文件头）。

    Args:
        path: 图像文件路径

    Returns:
        bool: True 表示图像格式有效且可解析
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
