"""
format_converter 单元测试。

覆盖 I-02 接口全部异常约定：
- 正常 8 字段标注行转为 YOLO 格式
- 12 个原始类别全部映射正确
- 原 ID 0 和 11 被过滤
- score=0 的框被过滤
- width≤1 或 height≤1 的框被过滤
- 坐标归一化→还原误差 < 1e-6
- 字段数≠8 的异常行跳过
- 非整数字段跳过
- 坐标越界自动裁剪
"""

import logging

import pytest

from utils.format_converter import convert_to_yolo


# ── fixture ─────────────────────────────────────────────────────

@pytest.fixture
def test_logger():
    """返回用于测试的 logger，方便 caplog 捕获日志。"""
    return logging.getLogger("test_format_converter")


# ── 辅助函数 ────────────────────────────────────────────────────

def make_line(bbox_left=100, bbox_top=200, bbox_width=50, bbox_height=60,
              score=1, obj_category=1, truncation=0, occlusion=0):
    """构造单行 VisDrone 原始标注。"""
    return f"{bbox_left},{bbox_top},{bbox_width},{bbox_height},{score},{obj_category},{truncation},{occlusion}"


# ══════════════════════════════════════════════════════════════════
# 1. 正常转换
# ══════════════════════════════════════════════════════════════════

def test_normal_conversion():
    """8 字段标注行正常转为 YOLO 格式（原 ID 1→新 0 pedestrian）。"""
    lines = [make_line(bbox_left=100, bbox_top=200, bbox_width=50, bbox_height=60,
                       score=1, obj_category=1)]
    yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480)

    assert len(yolo_lines) == 1
    parts = yolo_lines[0].split()
    assert len(parts) == 5
    class_id = int(parts[0])
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    assert class_id == 0  # 原 ID 1 → 新 0
    assert abs(cx - (100 + 25) / 640) < 1e-8
    assert abs(cy - (200 + 30) / 480) < 1e-8
    assert abs(w - 50 / 640) < 1e-8
    assert abs(h - 60 / 480) < 1e-8


# ══════════════════════════════════════════════════════════════════
# 2. 12 个原始类别全部映射正确
# ══════════════════════════════════════════════════════════════════

def test_class_mapping_all_12():
    """覆盖 12 个原始类别的完整转换路径。"""
    # 每行一个类别（ID 0~11），但 0 和 11 的 score 故意设为 0 方便单独测试
    # 实际上 0 和 11 被映射过滤，不需要 score=0
    lines = [
        make_line(obj_category=0, score=1),   # → 过滤（ignored）
        make_line(obj_category=1, score=1),   # → 新 0 pedestrian
        make_line(obj_category=2, score=1),   # → 新 1 people
        make_line(obj_category=3, score=1),   # → 新 2 bicycle
        make_line(obj_category=4, score=1),   # → 新 3 car
        make_line(obj_category=5, score=1),   # → 新 4 van
        make_line(obj_category=6, score=1),   # → 新 5 truck
        make_line(obj_category=7, score=1),   # → 新 6 tricycle
        make_line(obj_category=8, score=1),   # → 新 7 awning-tricycle
        make_line(obj_category=9, score=1),   # → 新 8 bus
        make_line(obj_category=10, score=1),  # → 新 9 motor
        make_line(obj_category=11, score=1),  # → 过滤（others）
    ]
    yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480)

    assert len(yolo_lines) == 10  # 12 - 2 过滤
    expected_new_ids = list(range(10))  # 0..9
    for i, line in enumerate(yolo_lines):
        new_id = int(line.split()[0])
        assert new_id == expected_new_ids[i], f"line {i}: expected class {expected_new_ids[i]}, got {new_id}"

    assert stats["total_boxes"] == 12
    assert stats["filtered_class_ignored"] == 1  # ID 0
    assert stats["filtered_class_others"] == 1   # ID 11


# ══════════════════════════════════════════════════════════════════
# 3. 过滤规则：score=0
# ══════════════════════════════════════════════════════════════════

def test_filter_score_zero():
    lines = [
        make_line(score=0, obj_category=1),  # 过滤
        make_line(score=1, obj_category=1),  # 保留
    ]
    yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480)

    assert len(yolo_lines) == 1
    assert stats["filtered_score_zero"] == 1


# ══════════════════════════════════════════════════════════════════
# 4. 过滤规则：width≤1 或 height≤1
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("w,h,should_filter", [
    (0, 50, True),
    (1, 50, True),
    (50, 0, True),
    (50, 1, True),
    (2, 2, False),
])
def test_filter_invalid_size(w, h, should_filter):
    lines = [make_line(bbox_width=w, bbox_height=h, obj_category=1)]
    yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480)

    if should_filter:
        assert len(yolo_lines) == 0
        assert stats["filtered_invalid_size"] == 1
    else:
        assert len(yolo_lines) == 1
        assert stats["filtered_invalid_size"] == 0


# ══════════════════════════════════════════════════════════════════
# 5. 坐标归一化 → 还原误差 < 1e-6
# ══════════════════════════════════════════════════════════════════

def test_coordinate_roundtrip_precision():
    """归一化→写入→读取→还原 误差 < 1e-6（对应需求 AC-1.2）。"""
    boxes_original = [
        (100, 200, 50, 60),
        (0, 0, 640, 480),
        (37, 83, 15, 22),  # 小目标
    ]
    img_w, img_h = 640, 480

    lines = [make_line(bbox_left=l, bbox_top=t, bbox_width=w, bbox_height=h, obj_category=1)
             for l, t, w, h in boxes_original]
    yolo_lines, _ = convert_to_yolo(lines, img_w=img_w, img_h=img_h)

    for i, line in enumerate(yolo_lines):
        parts = line.split()
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # 还原像素坐标
        bx = (cx * img_w) - (w * img_w) / 2.0
        by = (cy * img_h) - (h * img_h) / 2.0
        bw = w * img_w
        bh = h * img_h

        orig_x, orig_y, orig_w, orig_h = boxes_original[i]
        # 8 位小数归一化→文本→还原，像素误差 < 1e-5（对应 img=2000px 场景）
        assert abs(bx - orig_x) < 1e-5, f"x 还原误差过大: {bx} vs {orig_x}"
        assert abs(by - orig_y) < 1e-5, f"y 还原误差过大: {by} vs {orig_y}"
        assert abs(bw - orig_w) < 1e-5, f"w 还原误差过大: {bw} vs {orig_w}"
        assert abs(bh - orig_h) < 1e-5, f"h 还原误差过大: {bh} vs {orig_h}"


# ══════════════════════════════════════════════════════════════════
# 6. 字段数 ≠ 8 的异常行
# ══════════════════════════════════════════════════════════════════

def test_skip_invalid_field_count(test_logger, caplog):
    lines = [
        "1,2,3",                          # 3 字段 → 跳过
        make_line(obj_category=1),        # 正常
        "1,2,3,4,5,6,7,8,9,10",         # 10 字段 → 跳过
    ]
    with caplog.at_level(logging.WARNING, logger="test_format_converter"):
        yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480, logger=test_logger)

    assert len(yolo_lines) == 1
    assert stats["filtered_invalid_fields"] == 2
    # 验证 WARNING 日志：2 条字段数异常
    field_warnings = [r for r in caplog.records if "字段数异常" in r.message]
    assert len(field_warnings) == 2
    assert all("期望 8" in r.message for r in field_warnings)


# ══════════════════════════════════════════════════════════════════
# 7. 非整数字段
# ══════════════════════════════════════════════════════════════════

def test_skip_non_integer_fields(test_logger, caplog):
    lines = [
        "abc,200,50,60,1,1,0,0",         # 非整数
        make_line(obj_category=1),        # 正常
    ]
    with caplog.at_level(logging.WARNING, logger="test_format_converter"):
        yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480, logger=test_logger)

    assert len(yolo_lines) == 1
    assert stats["filtered_invalid_fields"] == 1
    # 验证 WARNING 日志：1 条非整数
    non_int = [r for r in caplog.records if "非整数字段" in r.message]
    assert len(non_int) == 1


# ══════════════════════════════════════════════════════════════════
# 8. 坐标越界自动裁剪
# ══════════════════════════════════════════════════════════════════

def test_clip_out_of_bounds(test_logger, caplog):
    lines = [
        make_line(bbox_left=-10, bbox_top=-5, bbox_width=100, bbox_height=80, obj_category=1),
        make_line(bbox_left=600, bbox_top=400, bbox_width=100, bbox_height=200, obj_category=1),
    ]
    with caplog.at_level(logging.INFO, logger="test_format_converter"):
        yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480, logger=test_logger)

    assert len(yolo_lines) == 2
    assert stats["clipped_boxes"] == 2

    # 第 1 个框：left 裁剪到 0，top 裁剪到 0
    parts = yolo_lines[0].split()
    cx, cy = float(parts[1]), float(parts[2])
    assert abs(cx - (0 + 90/2) / 640) < 1e-8  # width 被裁为 90 (原 -10+100=90)

    # 第 2 个框：right 超出 → width 裁剪到 40
    parts = yolo_lines[1].split()
    w = float(parts[3])
    assert abs(w - 40/640) < 1e-8  # width 裁剪为 img_w - left = 40

    # 验证 INFO 日志：2 条坐标越界裁剪
    clip_logs = [r for r in caplog.records if "坐标越界已裁剪" in r.message]
    assert len(clip_logs) == 2


# ══════════════════════════════════════════════════════════════════
# 9. 全部框被过滤 → 返回空列表
# ══════════════════════════════════════════════════════════════════

def test_all_filtered():
    lines = [
        make_line(score=0, obj_category=1),     # score=0
        make_line(obj_category=0, score=1),     # ignored
        make_line(bbox_width=0, obj_category=1), # invalid size
    ]
    yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480)

    assert yolo_lines == []
    assert stats["total_boxes"] == 3


# ══════════════════════════════════════════════════════════════════
# 10. 空输入
# ══════════════════════════════════════════════════════════════════

def test_empty_input():
    yolo_lines, stats = convert_to_yolo([], img_w=640, img_h=480)
    assert yolo_lines == []
    assert stats["total_boxes"] == 0


# ══════════════════════════════════════════════════════════════════
# 11. 自定义类别映射
# ══════════════════════════════════════════════════════════════════

def test_custom_class_mapping():
    """自定义映射表：只保留 pedestrian（1→0）和 car（4→1），过滤其余。"""
    custom = {1: 0, 4: 1}  # 只有这两个类别
    lines = [
        make_line(obj_category=1),   # → 新 0
        make_line(obj_category=4),   # → 新 1
        make_line(obj_category=2),   # → 不在映射 → 过滤
    ]
    yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480, class_mapping=custom, nc=2)

    assert len(yolo_lines) == 2
    assert yolo_lines[0].startswith("0 ")
    assert yolo_lines[1].startswith("1 ")


# ══════════════════════════════════════════════════════════════════
# 12. 映射表校验：越界 ID
# ══════════════════════════════════════════════════════════════════

def test_invalid_class_mapping_raises():
    """输出类别越界应抛出 ValueError。"""
    bad_mapping = {1: 99}  # 超出 0..9
    with pytest.raises(ValueError):
        convert_to_yolo([make_line(obj_category=1)], img_w=640, img_h=480,
                        class_mapping=bad_mapping, nc=10)


# ══════════════════════════════════════════════════════════════════
# 13. 空行不统计为异常字段
# ══════════════════════════════════════════════════════════════════

def test_blank_lines_not_counted():
    lines = [
        "",
        "   ",
        make_line(obj_category=1),
    ]
    yolo_lines, stats = convert_to_yolo(lines, img_w=640, img_h=480)

    assert len(yolo_lines) == 1
    assert stats["filtered_invalid_fields"] == 0
    assert stats["total_boxes"] == 1
