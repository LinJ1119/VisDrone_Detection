"""Quick test of visualizer with pre-computed eval result."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

import numpy as np

if __name__ == '__main__':
    from ultralytics import YOLO
    m = YOLO('runs/train/visdrone_baseline/weights/best.pt')
    r = m.val(data='datasets/visdrone/data.yaml', batch=4, workers=0, verbose=False)
    box = r.box

    from eval.eval_result import EvalResult
    from eval.evaluator import _load_class_names, _get_class_instance_count

    class_names = _load_class_names('datasets/visdrone/data.yaml', m)
    cls_indices = box.ap_class_index.tolist()

    per_class_AP, per_class_P, per_class_R = {}, {}, {}
    tp_fp_fn, fn_rate_per_class, high_fn_classes = {}, {}, []

    for idx, cls_id in enumerate(cls_indices):
        name = class_names[cls_id]
        per_class_AP[name] = float(box.ap50[idx])
        per_class_P[name] = float(box.p[idx])
        per_class_R[name] = float(box.r[idx])
        rc = float(box.r[idx])
        fn_rate_per_class[name] = 1.0 - rc
        n_gt = _get_class_instance_count('datasets/visdrone/data.yaml', name)
        tp = int(rc * n_gt)
        tp_fp_fn[name] = {'tp': tp, 'fp': max(0, int(tp / max(float(box.p[idx]), 1e-6)) - tp), 'fn': n_gt - tp}
        if fn_rate_per_class[name] > 0.4:
            high_fn_classes.append(name)

    eval_result = EvalResult(
        mAP50=float(box.map50), mAP50_95=float(box.map),
        precision=float(box.mp), recall=float(box.mr),
        per_class_AP=per_class_AP, per_class_P=per_class_P, per_class_R=per_class_R,
        tp_fp_fn=tp_fp_fn, fn_rate_per_class=fn_rate_per_class,
        high_fn_classes=high_fn_classes,
    )

    # Plot curves
    from eval.visualizer import plot_curves
    paths = plot_curves(
        eval_result, output_dir='./runs/eval/visdrone_baseline/curves',
        raw_p_curve=box.p_curve,
        raw_r_curve=box.r_curve,
        raw_f1_curve=box.f1_curve,
    )
    for p in paths:
        print(f'  {p} ({os.path.getsize(p)/1024:.0f} KB)')

    # Test draw_detections
    from eval.visualizer import draw_detections
    import cv2
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dets = [
        {'conf': 0.85, 'class_id': 3, 'xyxy': [100, 100, 300, 350]},
        {'conf': 0.45, 'class_id': 0, 'xyxy': [400, 50, 550, 200]},
    ]
    out = draw_detections(dummy, dets, conf_threshold=0.3)
    cv2.imwrite('./runs/eval/visdrone_baseline/curves/test_detections.png', out)
    print(f'  test_detections.png ({os.path.getsize("./runs/eval/visdrone_baseline/curves/test_detections.png")/1024:.0f} KB)')
    print('Done.')
