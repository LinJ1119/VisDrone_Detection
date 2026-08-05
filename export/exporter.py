"""
模型导出——ONNX / TensorRT FP32。

接口定义参见概要设计 I-10。
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_FORMATS = ("onnx", "engine")


def export_model(model_path, format="onnx", imgsz=640, opset=12):  # pylint: disable=redefined-builtin
    """导出训练好的 PyTorch 模型为 ONNX 或 TensorRT FP32 格式。

    接口 I-10。

    Args:
        model_path: checkpoint 路径
        format: 导出格式 "onnx" | "engine"
        imgsz: 输入图像尺寸
        opset: ONNX opset 版本（仅 onnx 格式使用）

    Returns:
        exported_file_path: 导出文件路径

    Raises:
        FileNotFoundError: model_path 不存在
        ValueError: format 非法值
        RuntimeError: 导出失败
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if format not in VALID_FORMATS:
        raise ValueError(
            f"非法的导出格式 '{format}'，合法值: {', '.join(VALID_FORMATS)}"
        )

    from ultralytics import YOLO

    model = YOLO(model_path)
    logger.info("导出模型: %s → %s (imgsz=%d, opset=%d)", model_path, format, imgsz, opset)

    try:
        if format == "onnx":
            export_kwargs = dict(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
        else:
            export_kwargs = dict(format="engine", imgsz=imgsz, half=False)  # FP32 only

        exported_path = model.export(**export_kwargs)

        if isinstance(exported_path, str):
            logger.info("导出成功: %s", exported_path)
            return exported_path

        # 某些版本返回路径
        result_path = str(Path(model_path).with_suffix(f".{format}"))
        logger.info("导出成功: %s", result_path)
        return result_path

    except Exception as e:
        raise RuntimeError(f"模型导出失败 ({format}): {e}") from e
