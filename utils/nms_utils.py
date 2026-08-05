"""
非极大值抑制 (Non-Maximum Suppression)。

标准贪心 NMS 实现，基于 NumPy，不依赖 PyTorch/CUDA。
"""

import numpy as np


def _box_area(boxes: np.ndarray) -> np.ndarray:
    """计算每个框的面积。"""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return w * h


def _box_iou(boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
    """计算 boxes (N,4) 与单个 box (4,) 的 IoU，返回 shape (N,)。"""
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_boxes = _box_area(boxes)
    area_box = _box_area(box.reshape(1, 4))[0]
    union_area = area_boxes + area_box - inter_area

    return np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area > 0)


def nms(boxes, scores, iou_threshold=0.45):
    """标准贪心 NMS。

    按 score 降序排列，依次选取最高分框，抑制与之 IoU > threshold 的框。

    Args:
        boxes: shape (N, 4)，格式 [[x1, y1, x2, y2], ...]（列表或 ndarray）
        scores: shape (N,)，置信度分数（列表或 ndarray）
        iou_threshold: IoU 阈值，超过该值的框被抑制（默认 0.45）

    Returns:
        keep: ndarray，保留框的索引列表，按原始顺序排列
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # 按 score 降序排序
    order = np.argsort(scores)[::-1]

    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        # 计算当前框与所有未处理框的 IoU
        ious = _box_iou(boxes[~suppressed], boxes[i])
        # 找到对应的原始索引
        remaining_indices = np.where(~suppressed)[0]
        for j, idx in enumerate(remaining_indices):
            if ious[j] > iou_threshold:
                suppressed[idx] = True

    return np.array(keep, dtype=np.int64)
