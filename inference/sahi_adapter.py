"""
SAHI 切片推理适配器。

封装 obss/sahi 库的切片推理调用。
shapely 不可用时直接报错退出，不提供自研 fallback（概要设计 V1.3）。
"""

import logging

logger = logging.getLogger(__name__)


def sahi_predict(model, image_path, slice_size=640, overlap=0.2, batch_size=4,
                 conf_threshold=0.25, device="cuda:0"):
    """SAHI 切片推理：切片 → 子图独立推理 → NMS 合并 → 返回全图检测结果。

    Args:
        model: ultralytics.YOLO 模型对象（或模型路径 .pt 文件）
        image_path: 输入图像路径
        slice_size: 切片尺寸（默认 640）
        overlap: 切片重叠率（默认 0.2）
        batch_size: 切片批处理大小（默认 4）
        conf_threshold: 置信度阈值（默认 0.25）
        device: 推理设备（默认 "cuda:0"）

    Returns:
        sahi.prediction.PredictionResult — 含全图检测框列表
            .object_prediction_list: 检测框列表
            .image: 图像 (PIL Image)

    Raises:
        ImportError: shapely 不可用时
    """
    try:
        import shapely  # noqa: F401,W0611
    except ImportError as exc:
        raise ImportError(
            "SAHI 推理需要 shapely 库，请运行：\n"
            "  conda install -c conda-forge shapely"
        ) from exc

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # 构建 SAHI 检测模型
    if isinstance(model, str):
        model_path = model
    else:
        # ultralytics.YOLO 对象 → 取 checkpoint 路径
        model_path = getattr(model, "ckpt_path", None) or getattr(model, "trainer", {})

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device,
    )

    logger.info("SAHI 切片推理: slice=%d, overlap=%.2f, batch=%d",
                slice_size, overlap, batch_size)

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=False,   # 不做全图推理，仅切片
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
    )

    n_detections = len(result.object_prediction_list)
    logger.info("SAHI 推理完成: 检测到 %d 个目标", n_detections)
    return result
