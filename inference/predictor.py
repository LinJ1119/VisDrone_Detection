"""
模型推理——直接缩放与 SAHI 切片推理。

接口定义参见概要设计 I-09。
"""

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from data.data_loader import VISDRONE_CLASS_NAMES
from utils.file_utils import verify_image as _verify_image

logger = logging.getLogger(__name__)

# 超尺寸阈值（最长边超过此值先降采样）
MAX_EDGE_PX = 2500
# SAHI 切片数告警阈值
MAX_SLICES_WARNING = 20


def run_predict(model_path, source, conf=0.25, conf_per_class=None, iou=0.45,
                sahi_config=None, imgsz=640):
    """对单张或批量图像执行目标检测推理。

    接口 I-09。

    Args:
        model_path: checkpoint 路径
        source: 图像路径或目录
        conf: 全局置信度阈值（默认 0.25）
        conf_per_class: 按类别置信度阈值 dict（None 使用全局 conf）
        iou: NMS IoU 阈值（默认 0.45）
        sahi_config: SAHI 配置 dict（None 表示直接缩放模式）
            {"enabled": True, "slice_size": 640, "overlap": 0.2, "batch_size": 4}
        imgsz: 输入图像尺寸

    Returns:
        list[DetectionResult]: 每张图的检测结果
    """
    from ultralytics import YOLO

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 收集图像列表
    source_path = Path(source)
    if source_path.is_dir():
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.extend(source_path.glob(ext))
        image_files = sorted(image_files)
        if not image_files:
            logger.info("目录 %s 中无图像文件", source)
            return []
    elif source_path.is_file():
        image_files = [source_path]
    else:
        raise FileNotFoundError(f"source 不存在: {source}")

    # 确定模式
    use_sahi = sahi_config is not None and sahi_config.get("enabled", False)
    mode = "sahi" if use_sahi else "direct"

    logger.info("推理模式: %s, 图像数: %d, conf=%.2f, iou=%.2f",
                mode, len(image_files), conf, iou)

    # 加载模型
    model = YOLO(model_path)

    # 预热（首次推理含 CUDA kernel 编译，不计入指标）
    _warmup(model, imgsz)

    # 逐张推理
    results = []
    skip_files = []
    total_start = time.perf_counter()

    for img_path in image_files:
        try:
            det_result = _predict_single(
                model, str(img_path), conf, conf_per_class, iou,
                sahi_config, imgsz, use_sahi
            )
            results.append(det_result)
        except Exception as e:
            logger.warning("图像 %s 推理失败: %s", img_path.name, e)
            skip_files.append(f"{img_path.name} ({e})")

    total_time = time.perf_counter() - total_start

    # 汇总报告
    total_boxes = sum(len(r.get("boxes", [])) for r in results)
    logger.info("=" * 50)
    logger.info("推理完成")
    logger.info("  模式:     %s", mode)
    logger.info("  总图像:   %d", len(image_files))
    logger.info("  成功:     %d", len(results))
    logger.info("  失败:     %d", len(skip_files))
    logger.info("  总检出框: %d", total_boxes)
    logger.info("  总耗时:   %.1f 秒", total_time)
    if skip_files:
        logger.info("  跳过文件: %s", ", ".join(skip_files))
    logger.info("=" * 50)

    return results


def _warmup(model, imgsz):
    """预热推理 10 次。"""
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy, imgsz=imgsz, verbose=False, device=model.device)


def _predict_single(model, img_path, conf, conf_per_class, iou,
                    sahi_config, imgsz, use_sahi):
    """单张图像推理。"""
    stem = Path(img_path).stem

    # PIL 预检
    if not _verify_image(img_path):
        raise ValueError("图像损坏或格式不支持")

    # 超尺寸降采样检查
    img = Image.open(img_path)
    orig_w, orig_h = img.size
    downsample_ratio = 1.0
    max_edge = max(orig_w, orig_h)
    if max_edge > MAX_EDGE_PX:
        downsample_ratio = MAX_EDGE_PX / max_edge
        new_w = int(orig_w * downsample_ratio)
        new_h = int(orig_h * downsample_ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        logger.info("图像 %s 超尺寸 (%d×%d → %d×%d)",
                    stem, orig_w, orig_h, new_w, new_h)
        # 保存临时文件给 SAHI 使用
        tmp_path = Path(img_path).with_name(f"{stem}_downsampled{Path(img_path).suffix}")
        img.save(tmp_path)
        actual_path = str(tmp_path)
    else:
        actual_path = img_path

    # 推理
    t0 = time.perf_counter()
    if use_sahi:
        slice_size = sahi_config.get("slice_size", 640)
        overlap = sahi_config.get("overlap", 0.2)
        sahi_batch = sahi_config.get("batch_size", 4)

        # 切片数预估
        n_slices_x = int(np.ceil(img.width / (slice_size * (1 - overlap))))
        n_slices_y = int(np.ceil(img.height / (slice_size * (1 - overlap))))
        total_slices = n_slices_x * n_slices_y
        if total_slices > MAX_SLICES_WARNING:
            logger.warning(
                "图像 %s 切片数 %d > %d，建议降采样或减小 overlap",
                stem, total_slices, MAX_SLICES_WARNING
            )

        from inference.sahi_adapter import sahi_predict
        sahi_result = sahi_predict(
            model, actual_path,
            slice_size=slice_size, overlap=overlap, batch_size=sahi_batch,
            conf_threshold=conf,
        )
        # 提取框
        boxes = []
        for obj in sahi_result.object_prediction_list:
            cls_name = obj.category.name
            cls_id = obj.category.id
            box_conf = obj.score.value
            # SAHI 坐标 → xyxy
            bbox = obj.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            if box_conf >= conf:
                boxes.append({
                    "class_id": cls_id, "class_name": cls_name,
                    "conf": round(box_conf, 4), "xyxy": [x1, y1, x2, y2],
                })
        mode_tag = "sahi"
    else:
        preds = model.predict(actual_path, imgsz=imgsz, conf=conf, iou=iou,
                              verbose=False)
        boxes = []
        for pred in preds:
            for i in range(len(pred.boxes)):
                xyxy = pred.boxes.xyxy[i].cpu().tolist()
                cls_id = int(pred.boxes.cls[i])
                box_conf = float(pred.boxes.conf[i])
                boxes.append({
                    "class_id": cls_id,
                    "class_name": VISDRONE_CLASS_NAMES[cls_id],
                    "conf": round(box_conf, 4),
                    "xyxy": xyxy,
                })
        mode_tag = "direct"

    t_ms = (time.perf_counter() - t0) * 1000

    # 清理临时降采样文件
    if downsample_ratio < 1.0:
        tmp_path = Path(actual_path)
        if tmp_path.exists() and "_downsampled" in tmp_path.name:
            tmp_path.unlink(missing_ok=True)

    return {
        "image_name": Path(img_path).name,
        "boxes": boxes,
        "inference_time_ms": round(t_ms, 2),
        "mode": mode_tag,
    }
