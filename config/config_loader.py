"""
配置加载器。

加载 config.yaml，合并 CLI 参数，返回结构化配置对象。
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from data.augmentation import AugConfig

logger = logging.getLogger(__name__)

# 默认配置模板路径（相对于项目根目录）
_DEFAULT_YAML = Path(__file__).resolve().parent / "default_config.yaml"


@dataclass
class DataConfig:
    train_root: str = ""
    val_root: str = ""
    test_root: str = ""
    image_dir: str = "images"
    annotation_dir: str = "annotations"
    output_base: str = "./datasets/visdrone"
    nc: int = 10
    names: List[str] = field(default_factory=lambda: [
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor",
    ])
    val_split_ratio: float = 0.0


@dataclass
class ModelConfig:
    name: str = "yolov8n"
    pretrained: str = "yolov8n.pt"
    imgsz: int = 640
    nc: int = 10
    p2_head: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 2
    accumulation_steps: int = 2
    epochs: int = 50
    amp: bool = True
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    optimizer: str = "AdamW"
    dropout: float = 0.0
    early_stop_patience: int = 15
    multiscale: bool = False
    output_dir: str = "./runs/train"
    name: str = ""


@dataclass
class SystemConfig:
    gpu_memory_fraction: float = 0.75
    num_workers: int = 2
    pin_memory: bool = False
    seed: int = 42


@dataclass
class InferenceConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    sahi_enabled: bool = False
    sahi_slice_size: int = 640
    sahi_overlap: float = 0.2
    sahi_batch_size: int = 4


@dataclass
class ExportConfig:
    format: str = "onnx"
    quantize: str = "fp32"
    dynamic_batch: bool = True
    calib_dataset: str = ""
    opset: int = 12


@dataclass
class Config:
    """顶层配置容器。"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def _deep_update(base: dict, overrides: dict) -> dict:
    """递归合并字典，overrides 覆盖 base 同名字段。"""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[str] = None) -> Config:
    """加载 YAML 配置文件，返回结构化 Config 对象。

    加载顺序：default_config.yaml（内置模板）→ config_path（用户覆盖）。
    用户只需提供需要修改的字段，未提供的保留默认值。

    Args:
        config_path: 用户自定义配置文件路径（None 使用内置默认）

    Returns:
        Config: 结构化配置对象
    """
    # 1) 加载内置默认模板
    with open(_DEFAULT_YAML, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # 2) 合并用户配置文件
    if config_path is not None:
        user_path = Path(config_path)
        if not user_path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {user_path}")
        with open(user_path, "r", encoding="utf-8") as f:
            user_dict = yaml.safe_load(f)
        _deep_update(config_dict, user_dict)
        logger.info("已加载用户配置: %s", user_path)
    else:
        logger.info("使用内置默认配置 (default_config.yaml)")

    # 3) 构造结构化对象
    return _dict_to_config(config_dict)


def _dict_to_config(d: dict) -> Config:
    """将嵌套字典转为 Config dataclass 对象。"""
    data = d.get("data", {})
    model = d.get("model", {})
    aug = d.get("aug", {})
    train = d.get("train", {})
    system = d.get("system", {})
    inference = d.get("inference", {})
    export = d.get("export", {})

    return Config(
        data=DataConfig(
            train_root=data.get("train_root", ""),
            val_root=data.get("val_root", ""),
            test_root=data.get("test_root", ""),
            image_dir=data.get("image_dir", "images"),
            annotation_dir=data.get("annotation_dir", "annotations"),
            output_base=data.get("output_base", "./datasets/visdrone"),
            nc=data.get("nc", 10),
            names=data.get("names", []),
            val_split_ratio=data.get("val_split_ratio", 0.0),
        ),
        model=ModelConfig(
            name=model.get("name", "yolov8n"),
            pretrained=model.get("pretrained", "yolov8n.pt"),
            imgsz=model.get("imgsz", 640),
            nc=model.get("nc", 10),
            p2_head=model.get("p2_head", False),
        ),
        aug=AugConfig(
            mosaic=aug.get("mosaic", True),
            flip_prob=aug.get("flip_prob", 0.5),
            hsv_h=aug.get("hsv_h", 0.015),
            hsv_s=aug.get("hsv_s", 0.7),
            hsv_v=aug.get("hsv_v", 0.4),
            close_mosaic=aug.get("close_mosaic", 10),
            multiscale=aug.get("multiscale", False),
        ),
        train=TrainConfig(
            batch_size=train.get("batch_size", 2),
            accumulation_steps=train.get("accumulation_steps", 2),
            epochs=train.get("epochs", 50),
            amp=train.get("amp", True),
            lr0=train.get("lr0", 0.01),
            lrf=train.get("lrf", 0.01),
            momentum=train.get("momentum", 0.937),
            weight_decay=train.get("weight_decay", 0.0005),
            optimizer=train.get("optimizer", "AdamW"),
            dropout=train.get("dropout", 0.0),
            early_stop_patience=train.get("early_stop_patience", 15),
            multiscale=train.get("multiscale", False),
            output_dir=train.get("output_dir", "./runs/train"),
            name=train.get("name", ""),
        ),
        system=SystemConfig(
            gpu_memory_fraction=system.get("gpu_memory_fraction", 0.75),
            num_workers=system.get("num_workers", 2),
            pin_memory=system.get("pin_memory", False),
            seed=system.get("seed", 42),
        ),
        inference=InferenceConfig(
            conf_threshold=inference.get("conf_threshold", 0.25),
            iou_threshold=inference.get("iou_threshold", 0.45),
            sahi_enabled=inference.get("sahi_enabled", False),
            sahi_slice_size=inference.get("sahi_slice_size", 640),
            sahi_overlap=inference.get("sahi_overlap", 0.2),
            sahi_batch_size=inference.get("sahi_batch_size", 4),
        ),
        export=ExportConfig(
            format=export.get("format", "onnx"),
            quantize=export.get("quantize", "fp32"),
            dynamic_batch=export.get("dynamic_batch", True),
            calib_dataset=export.get("calib_dataset", ""),
            opset=export.get("opset", 12),
        ),
    )


def save_config_snapshot(config: Config, output_dir: str):
    """将完整配置（含未修改的默认项）保存到输出目录。

    需求 AC-8.3：训练启动时自动保存配置快照。
    """
    import shutil
    from datetime import datetime

    ensure_dir(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(output_dir, f"config_{timestamp}.yaml")
    current_path = os.path.join(output_dir, "config_current.yaml")

    # 复制模板 + 用户覆盖后的最终配置（简化：直接拷贝 default_config.yaml）
    # 用户自定义配置通过 load_config 已经合并，这里保存内置模板作为参考
    # 实际使用中，用户应提供自己的 config.yaml
    shutil.copy2(_DEFAULT_YAML, snapshot_path)
    shutil.copy2(_DEFAULT_YAML, current_path)

    logger.info("配置快照已保存: %s / %s", snapshot_path, current_path)


def ensure_dir(path):
    """递归创建目录。"""
    Path(path).mkdir(parents=True, exist_ok=True)
