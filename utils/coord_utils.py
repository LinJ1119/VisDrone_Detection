"""
坐标转换工具函数。

边界框格式互转、归一化/反归一化、越界裁剪。
坐标写入精度统一为 f"{x:.8f}"（8 位小数，保证归一化还原误差 < 1e-6）。
"""

import numpy as np


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) → 中心点 (cx, cy, w, h)

    Args:
        boxes: shape (N, 4)，格式 [x1, y1, x2, y2]

    Returns:
        shape (N, 4)，格式 [cx, cy, w, h]
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    result = np.empty_like(boxes)
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0  # cx
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0  # cy
    result[:, 2] = boxes[:, 2] - boxes[:, 0]           # w
    result[:, 3] = boxes[:, 3] - boxes[:, 1]           # h
    return result


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """中心点 (cx, cy, w, h) → (x1, y1, x2, y2)

    Args:
        boxes: shape (N, 4)，格式 [cx, cy, w, h]

    Returns:
        shape (N, 4)，格式 [x1, y1, x2, y2]
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    result = np.empty_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # x1
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0  # y1
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0  # x2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0  # y2
    return result


def normalize_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """像素坐标 → 归一化 [0, 1]

    Args:
        boxes: shape (N, 4)，像素坐标（任意格式 xyxy 或 xywh）
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）

    Returns:
        shape (N, 4)，归一化坐标
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    result = np.empty_like(boxes)
    result[:, 0] = boxes[:, 0] / float(img_w)
    result[:, 1] = boxes[:, 1] / float(img_h)
    result[:, 2] = boxes[:, 2] / float(img_w)
    result[:, 3] = boxes[:, 3] / float(img_h)
    return result


def denormalize_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """归一化 [0, 1] → 像素坐标

    Args:
        boxes: shape (N, 4)，归一化坐标
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）

    Returns:
        shape (N, 4)，像素坐标
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    result = np.empty_like(boxes)
    result[:, 0] = boxes[:, 0] * float(img_w)
    result[:, 1] = boxes[:, 1] * float(img_h)
    result[:, 2] = boxes[:, 2] * float(img_w)
    result[:, 3] = boxes[:, 3] * float(img_h)
    return result


def clip_boxes(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """裁剪越界坐标到图像边界内。

    适用于 xyxy 格式的像素坐标框。

    Args:
        boxes: shape (N, 4)，格式 [x1, y1, x2, y2]（像素坐标）
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）

    Returns:
        shape (N, 4)，裁剪后的框
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    result = boxes.copy()
    # x1, y1 裁剪下界 → 0
    result[:, 0] = np.clip(result[:, 0], 0.0, float(img_w))
    result[:, 1] = np.clip(result[:, 1], 0.0, float(img_h))
    # x2, y2 裁剪上界 → img_w, img_h
    result[:, 2] = np.clip(result[:, 2], 0.0, float(img_w))
    result[:, 3] = np.clip(result[:, 3], 0.0, float(img_h))
    return result


def format_coord(value: float) -> str:
    """单个坐标值格式化为 8 位小数精度字符串。

    Args:
        value: 单个浮点坐标值

    Returns:
        "{:.8f}".format(value)
    """
    return f"{value:.8f}"


def format_yolo_line(class_id: int, cx: float, cy: float, w: float, h: float) -> str:
    """生成一行 YOLO 格式的标注文本。

    精度约定 f"{x:.8f}"，保证归一化→写入→读取→还原误差 < 1e-6。

    Args:
        class_id: 类别索引 [0, nc-1]
        cx: 归一化中心 x
        cy: 归一化中心 y
        w: 归一化宽度
        h: 归一化高度

    Returns:
        YOLO 标注行，如 "0 0.12345678 0.23456789 0.01234567 0.03456789"
    """
    return f"{int(class_id)} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"
