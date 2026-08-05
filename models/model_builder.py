"""
YOLOv8n 模型构建与预训练权重加载。

接口定义参见概要设计 I-04。
"""

import logging
from pathlib import Path
from typing import Literal

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# 合法模型名称
VALID_MODEL_NAMES = ("yolov8n", "yolov8n-p2")

# yaml 配置文件 → 本地副本路径（相对于 models/model_configs/）
_MODEL_YAML_MAP = {
    "yolov8n": "yolov8n.yaml",
    "yolov8n-p2": "yolov8n-p2.yaml",
}


def _resolve_yaml_path(model_name: str) -> str:
    """解析模型 yaml 配置文件的绝对路径。"""
    yaml_file = _MODEL_YAML_MAP[model_name]
    local = Path(__file__).resolve().parent / "model_configs" / yaml_file
    if local.is_file():
        return str(local)
    raise FileNotFoundError(
        f"模型配置文件不存在: {local}\n"
        f"请确保 models/model_configs/{yaml_file} 存在（nc 已改为 10）"
    )


def build_model(model_name: Literal["yolov8n", "yolov8n-p2"], pretrained_path: str, nc: int = 10):
    """构建 YOLOv8n 模型对象，加载预训练权重。

    接口 I-04。

    Args:
        model_name: 模型名称，合法值 "yolov8n" | "yolov8n-p2"
        pretrained_path: COCO 预训练权重路径或名称（如 "yolov8n.pt" 或本地路径）
        nc: 类别数（默认 10）

    Returns:
        ultralytics.YOLO — 可直接调用 model.train() / model.val() / model.predict()

    Raises:
        ValueError: model_name 不在合法值列表中
        FileNotFoundError: P2 yaml 配置文件不存在
    """
    if model_name not in VALID_MODEL_NAMES:
        raise ValueError(
            f"非法的 model_name '{model_name}'，"
            f"合法值: {', '.join(VALID_MODEL_NAMES)}"
        )

    yaml_path = _resolve_yaml_path(model_name)

    logger.info("构建模型: %s (yaml=%s, nc=%d)", model_name, yaml_path, nc)

    # 构建 YOLO 模型
    try:
        model = YOLO(yaml_path)
    except Exception as e:
        raise RuntimeError(f"模型构建失败 ({model_name}): {e}") from e

    # 加载预训练权重
    try:
        model.load(pretrained_path)
        logger.info("预训练权重加载成功: %s", pretrained_path)
    except Exception:
        # 下载失败或文件不存在 → 降级为随机初始化
        logger.warning(
            "预训练权重加载失败 (%s)，降级为 kaiming_uniform 随机初始化。"
            "预期影响：收敛速度变慢，最终 mAP 可能降低 1~3 个百分点。"
            "建议：检查网络连接或将 yolov8n.pt 放到项目根目录。",
            pretrained_path
        )

    # Ultralytics 内置行为：仅加载匹配层的权重，新增层（如 P2 头）自动随机初始化
    # 当 model_name="yolov8n-p2" 时，P2 相关层在 yaml 中存在但预训练权重中无对应 key，
    # Ultralytics 自动跳过这些层并使用 kaiming_uniform 初始化。

    return model
