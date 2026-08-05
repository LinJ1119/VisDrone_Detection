"""
VisDrone 数据集加载与格式转换。

接口定义参见概要设计 I-01。
"""

import logging
import random
import shutil
from pathlib import Path

import yaml

from utils.file_utils import ensure_dir, verify_image
from utils.format_converter import convert_to_yolo, _make_filter_stats

logger = logging.getLogger(__name__)

# VisDrone 10 类别（按 YOLO ID 0~9，对应原始 ID 1~10）
VISDRONE_CLASS_NAMES = [
    "pedestrian",       # 0
    "people",           # 1
    "bicycle",          # 2
    "car",              # 3
    "van",              # 4
    "truck",            # 5
    "tricycle",         # 6
    "awning-tricycle",  # 7
    "bus",              # 8
    "motor",            # 9
]


def load_dataset(root_dir, image_dir="images", annotation_dir="annotations"):
    """读取 VisDrone 数据集目录，配对图像与标注文件。

    接口 I-01。扫描 root_dir 下的 images/ 和 annotations/ 子目录，
    按文件名（不含扩展名）配对。

    Args:
        root_dir: 数据集根目录路径（如 "D:/Data/VisDrone/train"）
        image_dir: 图像子目录名（默认 "images"）
        annotation_dir: 标注子目录名（默认 "annotations"）

    Returns:
        (paired_list, stats)
        - paired_list: [(image_abs_path, annotation_abs_path), ...]
        - stats: {"total_images", "paired", "skipped_no_annotation",
                   "skipped_no_image", "skipped_files": [...]}

    Raises:
        FileNotFoundError: root_dir 或其子目录不存在
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    img_dir = root / image_dir
    ann_dir = root / annotation_dir

    if not img_dir.is_dir():
        raise FileNotFoundError(f"图像子目录不存在: {img_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"标注子目录不存在: {ann_dir}")

    # 收集图像文件（支持常见扩展名）
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = {}
    for f in img_dir.iterdir():
        if f.suffix.lower() in image_extensions:
            image_files[f.stem] = str(f.resolve())

    total_images = len(image_files)
    logger.info("发现 %d 张图像（%s）", total_images, img_dir)

    stats = {
        "total_images": total_images,
        "paired": 0,
        "skipped_no_annotation": 0,
        "skipped_no_image": 0,
        "skipped_files": [],
    }

    paired_list = []

    # 先按图像找标注 → 配对
    for stem, img_path in sorted(image_files.items()):
        ann_file = ann_dir / f"{stem}.txt"
        if ann_file.is_file():
            paired_list.append((img_path, str(ann_file.resolve())))
        else:
            stats["skipped_no_annotation"] += 1
            stats["skipped_files"].append(f"{stem} (无对应标注文件)")
            logger.warning("图像 %s 无对应标注，跳过", stem)

    # 检查有标注但无图像的孤儿文件
    paired_stems = {Path(p[0]).stem for p in paired_list}
    for f in ann_dir.iterdir():
        if f.suffix.lower() == ".txt" and f.stem not in paired_stems:
            stats["skipped_no_image"] += 1
            stats["skipped_files"].append(f"{f.stem} (无对应图像文件)")
            logger.warning("标注 %s 无对应图像，跳过", f.stem)

    stats["paired"] = len(paired_list)
    logger.info("配对完成: 成功 %d, 无标注 %d, 无图像 %d",
                stats["paired"], stats["skipped_no_annotation"], stats["skipped_no_image"])

    return paired_list, stats


def convert_and_save(root_dir, output_dir, image_dir="images", annotation_dir="annotations",
                     label_dir="labels", class_mapping=None, nc=10):
    """加载 VisDrone 数据集 → 转换为 YOLO 格式 → 保存到 output_dir。

    整体流程（对应概要设计 §3.1 步骤 2）：
    1. load_dataset 配对图像与标注
    2. 逐张读取原始标注 → convert_to_yolo 转换
    3. 过滤空标注图像（所有框均被过滤的）
    4. 转换后的 YOLO label 写入 output_dir/label_dir/

    Args:
        root_dir: 数据集根目录
        output_dir: 转换输出根目录
        image_dir: 图像子目录名
        annotation_dir: 标注子目录名
        label_dir: 输出 YOLO 标注子目录名
        class_mapping: 类别映射表（None 使用默认）
        nc: 类别数

    Returns:
        (output_image_dir, output_label_dir, overall_stats)
        - output_image_dir: 转换后图像所在目录路径
        - output_label_dir: YOLO 标注输出目录路径
        - overall_stats: 汇总统计 {"paired", "converted", "skipped_empty",
                                    "total_boxes", ...}
    """
    root = Path(root_dir)
    out = Path(output_dir)

    paired_list, load_stats = load_dataset(
        str(root), image_dir=image_dir, annotation_dir=annotation_dir
    )

    output_img_dir = out / "images"
    output_lbl_dir = out / label_dir
    ensure_dir(output_img_dir)
    ensure_dir(output_lbl_dir)

    overall_stats = {
        **_make_filter_stats(),
        "paired": load_stats["paired"],
        "converted": 0,
        "skipped_empty": 0,
        "total_boxes_raw": 0,
        "skipped_corrupt_image": 0,
        "skipped_files": [],
    }

    for img_path, ann_path in paired_list:
        stem = Path(img_path).stem

        # 校验图像可读
        if not verify_image(img_path):
            overall_stats["skipped_corrupt_image"] += 1
            overall_stats["skipped_files"].append(f"{stem} (图像损坏)")
            logger.warning("图像 %s 损坏，跳过", stem)
            continue

        # 复制图像到输出目录
        dst_img_path = output_img_dir / f"{stem}{Path(img_path).suffix}"
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)

        # 读取原始标注
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
        except Exception:
            logger.warning("读取标注文件失败: %s", ann_path)
            continue

        # 获取图像尺寸（用于归一化）
        from PIL import Image
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            overall_stats["skipped_corrupt_image"] += 1
            overall_stats["skipped_files"].append(f"{stem} (无法读取尺寸)")
            logger.warning("无法读取图像尺寸: %s", img_path)
            continue

        # 转换
        yolo_lines, box_stats = convert_to_yolo(
            raw_lines, img_w, img_h,
            class_mapping=class_mapping, nc=nc, logger=logger
        )

        # 汇总统计
        overall_stats["total_boxes_raw"] += box_stats["total_boxes"]
        for key in ["filtered_score_zero", "filtered_class_ignored",
                    "filtered_class_others", "filtered_invalid_size",
                    "filtered_invalid_fields", "clipped_boxes"]:
            overall_stats[key] += box_stats[key]

        # 如果全部框被过滤，跳过该图像
        if not yolo_lines:
            overall_stats["skipped_empty"] += 1
            overall_stats["skipped_files"].append(f"{stem} (所有标注框被过滤)")
            logger.warning("图像 %s 所有标注框被过滤，跳过", stem)
            continue

        # 写入 YOLO 标注文件
        output_lbl_path = output_lbl_dir / f"{stem}.txt"
        with open(output_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        overall_stats["converted"] += 1

    logger.info(
        "转换完成: 成功 %d/%d, 跳过空标注 %d, 图像损坏 %d",
        overall_stats["converted"], overall_stats["paired"],
        overall_stats["skipped_empty"], overall_stats["skipped_corrupt_image"]
    )

    return str(output_img_dir), str(output_lbl_dir), overall_stats


def generate_data_yaml(train_path, val_path, test_path, nc, names, output_path):
    """生成 Ultralytics 标准的 data.yaml。

    参见需求 AC-1.5。所有路径使用正斜杠。

    Args:
        train_path: 训练集路径（图像所在目录）
        val_path: 验证集路径
        test_path: 测试集路径
        nc: 类别数
        names: 类别名列表
        output_path: 输出 yaml 文件路径

    Returns:
        data_yaml 的绝对路径
    """
    data = {
        "path": str(Path(train_path).parent.parent.resolve()).replace("\\", "/"),
        "train": str(Path(train_path).resolve()).replace("\\", "/"),
        "val": str(Path(val_path).resolve()).replace("\\", "/"),
        "test": str(Path(test_path).resolve()).replace("\\", "/"),
        "nc": nc,
        "names": {i: name for i, name in enumerate(names)},
    }

    output = Path(output_path)
    ensure_dir(output.parent)
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info("data.yaml 已生成: %s", output)
    return str(output.resolve())


def split_dataset(root_dir, val_ratio=0.1, seed=42, image_dir="images", annotation_dir="annotations"):
    """按随机比例从训练集中切分验证集。

    使用固定随机种子 seed=42 保证可复现（需求 AC-1.4）。

    Args:
        root_dir: 数据集根目录
        val_ratio: 验证集比例（0.0 ~ 1.0）
        seed: 随机种子
        image_dir: 图像子目录名
        annotation_dir: 标注子目录名

    Returns:
        (train_pairs, val_pairs)
        每个元素为 [(image_path, annotation_path), ...]
    """
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio 必须在 [0, 1] 范围内，当前: {val_ratio}")

    paired_list, _ = load_dataset(
        str(root_dir), image_dir=image_dir, annotation_dir=annotation_dir
    )

    if val_ratio == 0.0:
        return paired_list, []

    random.seed(seed)
    indices = list(range(len(paired_list)))
    random.shuffle(indices)

    split_point = int(len(indices) * (1.0 - val_ratio))
    train_indices = set(indices[:split_point])
    val_indices = set(indices[split_point:])

    train_pairs = [paired_list[i] for i in sorted(train_indices)]
    val_pairs = [paired_list[i] for i in sorted(val_indices)]

    logger.info(
        "数据集划分 (seed=%d, ratio=%.2f): 训练 %d, 验证 %d",
        seed, val_ratio, len(train_pairs), len(val_pairs)
    )

    return train_pairs, val_pairs


def prepare_dataset(train_root, val_root=None, test_root=None, output_base="./datasets/visdrone",
                    val_split_ratio=0.0, seed=42, class_mapping=None, nc=10, names=None):
    """数据准备一站式入口。

    对应概要设计 §3.1 步骤 2 的完整数据集准备流程：
    1. 加载/切分训练集和验证集
    2. 转换标注格式（VisDrone → YOLO）
    3. 生成 data.yaml

    Args:
        train_root: 训练集原始目录
        val_root: 验证集原始目录（None 时从训练集切分）
        test_root: 测试集原始目录
        output_base: 输出根目录
        val_split_ratio: 从训练集切分验证集比例（val_root 为 None 时生效）
        seed: 随机种子
        class_mapping: 类别映射表
        nc: 类别数
        names: 类别名列表（None 使用 VISDRONE_CLASS_NAMES）

    Returns:
        data_yaml_path: 生成的 data.yaml 绝对路径
    """
    if names is None:
        names = VISDRONE_CLASS_NAMES

    out_base = Path(output_base)

    # ── 处理训练集 / 验证集 ─────────────────────────────────
    if val_root is not None:
        # 模式 1：官方划分
        train_output, _, train_stats = convert_and_save(
            train_root, str(out_base / "train"),
            class_mapping=class_mapping, nc=nc
        )
        val_output, _, val_stats = convert_and_save(
            val_root, str(out_base / "val"),
            class_mapping=class_mapping, nc=nc
        )
    else:
        # 模式 2：从训练集随机切分
        if not (0.0 < val_split_ratio < 1.0):
            raise ValueError(
                "val_root 为 None 时 val_split_ratio 必须 > 0 且 < 1，"
                f"当前: {val_split_ratio}"
            )
        train_pairs, val_pairs = split_dataset(
            train_root, val_ratio=val_split_ratio, seed=seed
        )
        # 转换训练集
        train_output, _, train_stats = _convert_pairs(
            train_pairs, str(out_base / "train"), class_mapping, nc
        )
        # 转换验证集
        val_output, _, val_stats = _convert_pairs(
            val_pairs, str(out_base / "val"), class_mapping, nc
        )

    logger.info("训练集: %d 张", train_stats["converted"])
    logger.info("验证集: %d 张", val_stats["converted"])

    # ── 处理测试集 ──────────────────────────────────────────
    test_output = None
    if test_root is not None and Path(test_root).is_dir():
        test_output, _, test_stats = convert_and_save(
            test_root, str(out_base / "test"),
            class_mapping=class_mapping, nc=nc
        )
        logger.info("测试集: %d 张", test_stats["converted"])
    else:
        logger.info("未提供测试集路径，data.yaml 中 test 字段留空")

    # ── 生成 data.yaml ──────────────────────────────────────
    yaml_path = str(out_base / "data.yaml")
    generate_data_yaml(
        train_path=train_output,
        val_path=val_output,
        test_path=test_output or "",
        nc=nc,
        names=names,
        output_path=yaml_path,
    )

    return yaml_path


def _convert_pairs(pairs, output_dir, class_mapping, nc):
    """内部函数：将 (img, ann) 配对列表转换为 YOLO 格式并输出。"""
    out = Path(output_dir)
    out_img = out / "images"
    out_lbl = out / "labels"
    ensure_dir(out_img)
    ensure_dir(out_lbl)

    stats = {
        "converted": 0,
        "skipped_empty": 0,
        "total_boxes_raw": 0,
        "filtered_score_zero": 0,
        "filtered_class_ignored": 0,
        "filtered_class_others": 0,
        "filtered_invalid_size": 0,
        "filtered_invalid_fields": 0,
        "clipped_boxes": 0,
        "skipped_corrupt_image": 0,
        "skipped_files": [],
    }

    for img_path, ann_path in pairs:
        stem = Path(img_path).stem

        if not verify_image(img_path):
            stats["skipped_corrupt_image"] += 1
            stats["skipped_files"].append(f"{stem} (图像损坏)")
            continue

        # 复制图像到输出目录
        dst_img_path = out_img / f"{stem}{Path(img_path).suffix}"
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()
        except Exception:
            continue

        from PIL import Image
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            stats["skipped_corrupt_image"] += 1
            stats["skipped_files"].append(f"{stem} (无法读取尺寸)")
            continue

        yolo_lines, box_stats = convert_to_yolo(
            raw_lines, img_w, img_h,
            class_mapping=class_mapping, nc=nc, logger=logger
        )

        for key in ["total_boxes_raw", "filtered_score_zero", "filtered_class_ignored",
                    "filtered_class_others", "filtered_invalid_size",
                    "filtered_invalid_fields", "clipped_boxes"]:
            stats[key] = stats.get(key, 0) + box_stats.get(key, 0)

        if not yolo_lines:
            stats["skipped_empty"] += 1
            stats["skipped_files"].append(f"{stem} (所有标注框被过滤)")
            continue

        with open(out_lbl / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        stats["converted"] += 1

    return str(out_img), str(out_lbl), stats
