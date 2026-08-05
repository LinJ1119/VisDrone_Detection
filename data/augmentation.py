"""
在线数据增强管道。

接口定义参见概要设计 I-03。
通过 is_train 区分训练/验证模式，使用 Ultralytics 内置增强管道。
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import yaml
from ultralytics.data import build_yolo_dataset as _build_yolo_dataset
from ultralytics.data import build_dataloader as _build_dataloader
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml as _check_yaml

logger = logging.getLogger(__name__)


@dataclass
class AugConfig:
    """数据增强配置。

    字段与 config.yaml 中 aug 节一一对应，带 Ultralytics 8.1.0 默认值。
    对应需求 FR-2（基础增强 Must）和 AC-2.4（多尺度训练 Should）。
    """

    mosaic: bool = True               # Mosaic 增强（4 张拼接）
    flip_prob: float = 0.5            # 水平翻转概率
    hsv_h: float = 0.015              # HSV-H 色调抖动
    hsv_s: float = 0.7                # HSV-S 饱和度抖动
    hsv_v: float = 0.4                # HSV-V 明度抖动
    close_mosaic: int = 10            # 最后 N 个 epoch 关闭 Mosaic
    multiscale: bool = False          # 多尺度训练（Should 可选，默认关闭）
    multiscale_range: Tuple[float, float] = (0.5, 1.5)  # 多尺度范围因子（320~960）

    def to_ultralytics_cfg(self, imgsz: int = 640) -> IterableSimpleNamespace:
        """将 AugConfig 转为 Ultralytics 兼容的配置对象。

        以 DEFAULT_CFG 为基底，覆盖 AugConfig 中非默认项。
        """
        cfg_dict = dict(vars(DEFAULT_CFG))
        overrides = {
            "imgsz": imgsz,
            "mosaic": 1.0 if self.mosaic else 0.0,
            "fliplr": self.flip_prob,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            # 关闭其他非默认增强（保持行为可控）
            "mixup": 0.0,
            "degrees": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
        }
        cfg_dict.update(overrides)
        return IterableSimpleNamespace(**cfg_dict)


def build_dataloader(data_yaml, batch_size, num_workers=2, is_train=True,
                     aug_config=None, imgsz=640):
    """构建 PyTorch DataLoader，集成在线增强管道。

    接口 I-03。

    Args:
        data_yaml: Ultralytics 标准 data.yaml 路径
        batch_size: 批大小
        num_workers: DataLoader 子进程数
        is_train: True=训练模式（完整增强），False=验证模式（仅 LetterBox）
        aug_config: AugConfig 增强参数（None 使用默认值）
        imgsz: 输入图像尺寸（默认 640）

    Returns:
        torch.utils.data.DataLoader

    Raises:
        FileNotFoundError: data_yaml 不存在
        ValueError: YAML 解析失败 或 batch_size ≤ 0
    """
    if aug_config is None:
        aug_config = AugConfig()

    if batch_size <= 0:
        raise ValueError(f"batch_size 必须 > 0，当前: {batch_size}")

    # 校验并加载 data.yaml
    data_yaml = _check_yaml(data_yaml)
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_dict = yaml.safe_load(f)

    # 构建 ultralytics 兼容的配置对象
    cfg = aug_config.to_ultralytics_cfg(imgsz=imgsz)

    # 取图像目录路径
    if is_train:
        img_dir = data_dict.get("train", "")
        mode = "train"
    else:
        img_dir = data_dict.get("val", "")
        mode = "val"

    if not img_dir:
        raise ValueError(f"data.yaml 中缺少 {'train' if is_train else 'val'} 路径字段")

    logger.info("构建 %s DataLoader: imgsz=%d, batch=%d, workers=%d",
                "训练" if is_train else "验证", imgsz, batch_size, num_workers)

    # 构建 YOLO 数据集
    dataset = _build_yolo_dataset(
        cfg=cfg,
        img_path=img_dir,
        batch=batch_size,
        data=data_dict,
        mode=mode,
        rect=not is_train,  # 验证模式使用矩形批（提高效率）
        stride=32,
    )

    # 构建 DataLoader
    dataloader = _build_dataloader(
        dataset=dataset,
        batch=batch_size,
        workers=num_workers,
        shuffle=is_train,
    )

    logger.info("%s DataLoader 构建完成: %d 批", "训练" if is_train else "验证", len(dataloader))
    return dataloader
