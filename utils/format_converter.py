"""
VisDrone 原始标注 → YOLO 格式转换器（纯函数）。

接口定义参见概要设计 I-02。这是项目最高风险模块——
原 ID 0 是 ignored regions，不是 pedestrian，映射错误会导致静默 bug。
"""
from .coord_utils import format_yolo_line

# ── 默认类别映射：原始 ID → 新 ID ──────────────────────────────
# 原始 12 类 → YOLO 10 类：
#   0 (ignored) → 过滤    1 (pedestrian) → 0      2 (people) → 1
#   3 (bicycle) → 2        4 (car) → 3             5 (van) → 4
#   6 (truck) → 5          7 (tricycle) → 6        8 (awning-tricycle) → 7
#   9 (bus) → 8            10 (motor) → 9          11 (others) → 过滤
DEFAULT_CLASS_MAPPING = {
    0: None,   # ignored  → 过滤
    1: 0,      # pedestrian，人群
    2: 1,      # people，人
    3: 2,      # bicycle，自行车
    4: 3,      # car，小汽车
    5: 4,      # van，厢式车
    6: 5,      # truck，卡车
    7: 6,      # tricycle，三轮车
    8: 7,      # awning-tricycle，篷布三轮车
    9: 8,      # bus，公交车
    10: 9,     # motor，摩托车
    11: None,  # others → 过滤
}

# 默认类别数
DEFAULT_NC = 10


def _make_filter_stats():
    """创建空过滤统计字典（供 data_loader 等模块复用）。"""
    return {
        "total_boxes": 0,
        "filtered_score_zero": 0,
        "filtered_class_ignored": 0,
        "filtered_class_others": 0,
        "filtered_invalid_size": 0,
        "filtered_invalid_fields": 0,
        "clipped_boxes": 0,
    }


def _validate_class_mapping(mapping: dict, nc: int):
    """校验类别映射表：所有输出值必须在 [0, nc-1] 范围内。

    Raises:
        ValueError: 映射表包含越界类别 ID 时
    """
    for src_id, dst_id in mapping.items():
        if dst_id is not None and not (0 <= dst_id < nc):
            raise ValueError(
                f"类别映射表中原 ID {src_id} → 新 ID {dst_id} 越界，"
                f"期望范围 [0, {nc - 1}]"
            )


def _parse_line(line: str, line_no: int, logger):
    """解析单行原始标注，成功返回 8 元素元组，失败返回 None。"""
    line = line.strip()
    # 处理行尾多余逗号（VisDrone 部分标注文件存在此问题）
    line = line.rstrip(",")
    if not line:
        return None

    fields = line.split(",")
    if len(fields) != 8:
        if logger:
            logger.warning(
                "标注行 %d 字段数异常（期望 8，实际 %d），跳过：%s",
                line_no, len(fields), line[:80]
            )
        return None

    try:
        values = tuple(int(f.strip()) for f in fields)
    except ValueError:
        if logger:
            logger.warning(
                "标注行 %d 含非整数字段，跳过：%s",
                line_no, line[:80]
            )
        return None

    return values


def convert_to_yolo(annotation_lines, img_w, img_h, class_mapping=None, nc=DEFAULT_NC, logger=None):
    """将 VisDrone 原始 8 字段标注行转换为 YOLO 格式。

    Args:
        annotation_lines: 原始标注行列表，每行 "left,top,width,height,score,category,trunc,occlusion"
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）
        class_mapping: 类别映射表 {原ID: 新ID | None}（None 使用 DEFAULT_CLASS_MAPPING）
        nc: 类别总数（默认 10，用于校验映射表）
        logger: 可选的 logging.Logger（None 表示静默，便于单元测试）

    Returns:
        (yolo_lines, filter_stats)
        - yolo_lines: YOLO 格式标注行 ["class_id cx cy w h", ...]
        - filter_stats: {"total_boxes", "filtered_score_zero", "filtered_class_ignored",
            "filtered_class_others", "filtered_invalid_size", "filtered_invalid_fields",
            "clipped_boxes"}
    """
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING

    _validate_class_mapping(class_mapping, nc)

    stats = _make_filter_stats()

    yolo_lines = []

    for line_no, line in enumerate(annotation_lines, start=1):
        values = _parse_line(line, line_no, logger)
        if values is None:
            # 空行跳过不统计，字段异常才统计
            if line.strip():
                stats["filtered_invalid_fields"] += 1
            continue

        bbox_left, bbox_top, bbox_width, bbox_height, score, obj_category, _trunc, _occlusion = values

        stats["total_boxes"] += 1

        # ── 过滤规则 ──────────────────────────────────────────────

        # 规则 1: score == 0
        if score == 0:
            stats["filtered_score_zero"] += 1
            continue

        # 规则 2: 原 ID 0（ignored）
        if obj_category == 0:
            stats["filtered_class_ignored"] += 1
            continue

        # 规则 3: 原 ID 11（others）
        if obj_category == 11:
            stats["filtered_class_others"] += 1
            continue

        # 规则 4: width ≤ 1 或 height ≤ 1
        if bbox_width <= 1 or bbox_height <= 1:
            stats["filtered_invalid_size"] += 1
            continue

        # ── 类别映射 ──────────────────────────────────────────────
        if obj_category not in class_mapping:
            if logger:
                logger.warning(
                    "标注行 %d 类别 ID %d 不在映射表中，跳过",
                    line_no, obj_category
                )
            stats["filtered_class_others"] += 1
            continue

        new_class_id = class_mapping[obj_category]
        if new_class_id is None:
            # 显式标记为 None = 过滤（ignored / others 走这里）
            # 但上面已经单独处理了 0 和 11，这里是用自定义映射时的兜底
            stats["filtered_class_others"] += 1
            continue

        # ── 坐标裁剪 ──────────────────────────────────────────────
        clipped = False
        # 左上角越界：收缩 left/top，同步缩小宽高
        if bbox_left < 0:
            bbox_width += bbox_left   # bbox_left 为负，等价 width -= |left|
            bbox_left = 0
            clipped = True
        if bbox_top < 0:
            bbox_height += bbox_top   # bbox_top 为负
            bbox_top = 0
            clipped = True
        # 右下角越界：裁剪宽高至图像边界
        if bbox_left + bbox_width > img_w:
            bbox_width = img_w - bbox_left
            clipped = True
        if bbox_top + bbox_height > img_h:
            bbox_height = img_h - bbox_top
            clipped = True
        if clipped:
            stats["clipped_boxes"] += 1
            if logger:
                logger.info(
                    "标注行 %d 坐标越界已裁剪 (left=%d, top=%d, w=%d, h=%d, img=%d×%d)",
                    line_no, bbox_left, bbox_top, bbox_width, bbox_height, img_w, img_h
                )

        # 裁剪后再次校验宽高
        if bbox_width <= 1 or bbox_height <= 1:
            stats["filtered_invalid_size"] += 1
            continue

        # ── 坐标归一化 ────────────────────────────────────────────
        cx = (bbox_left + bbox_width / 2.0) / img_w
        cy = (bbox_top + bbox_height / 2.0) / img_h
        w = bbox_width / img_w
        h = bbox_height / img_h

        # ── 写入 YOLO 行（8 位小数精度） ──────────────────────────
        yolo_lines.append(format_yolo_line(new_class_id, cx, cy, w, h))

    return yolo_lines, stats
