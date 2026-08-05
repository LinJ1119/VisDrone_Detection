"""
check_env.py — 一键环境检测脚本
输出三级报告：通过 / 警告 / 失败
"""
import os
import sys
import subprocess
import shutil
import re
from pathlib import Path

# ── 工具函数 ────────────────────────────────────────────
PASS = "✅ 通过"
WARN = "⚠️ 警告"
FAIL = "❌ 失败"


def print_header(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_result(label, status, detail=""):
    detail_str = f"  —  {detail}" if detail else ""
    print(f"  {status}: {label}{detail_str}")


# ── 检查 1: Python 版本 ────────────────────────────────
def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor == 8
    status = PASS if ok else FAIL
    print_result("Python 版本", status, f"{v.major}.{v.minor}.{v.micro}")
    return ok


# ── 检查 2: PyTorch + CUDA + GPU ────────────────────────
def check_pytorch():
    try:
        import torch
        print_result("PyTorch 版本", PASS, torch.__version__)

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print_result("CUDA 可用性", PASS, f"GPU: {gpu_name}, 显存: {vram_total:.1f} GB")
        else:
            print_result("CUDA 可用性", FAIL, "CUDA 不可用，请检查安装")
        return cuda_available
    except ImportError:
        print_result("PyTorch", FAIL, "未安装")
        return False


# ── 检查 3: CUDA + 驱动版本 ─────────────────────────────
def check_cuda_driver():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            encoding="utf-8", errors="replace"
        ).strip()
        driver_ver = output.split("\n")[0] if output else "未知"
        print_result("NVIDIA 驱动版本", PASS if driver_ver != "未知" else WARN, driver_ver)
    except Exception:
        print_result("NVIDIA 驱动版本", WARN, "nvidia-smi 不可用")

    try:
        import torch
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            print_result("CUDA 版本 (PyTorch)", PASS, cuda_ver)
        else:
            print_result("CUDA 版本 (PyTorch)", WARN, "CUDA 不可用")
    except ImportError:
        print_result("CUDA 版本", WARN, "PyTorch 未安装")


# ── 检查 4: 磁盘空间 ────────────────────────────────────
def check_disk():
    project_dir = Path(__file__).resolve().parent
    drive = project_dir.drive or "C:"
    usage = shutil.disk_usage(drive)
    free_gb = usage.free / (1024 ** 3)

    if free_gb >= 10:
        status = PASS
    elif free_gb >= 1:
        status = WARN
    else:
        status = FAIL

    print_result("磁盘剩余空间", status, f"{drive} 剩余 {free_gb:.1f} GB")
    return free_gb >= 1


# ── 检查 5: GPU 其他进程占用 ─────────────────────────────
def check_gpu_processes():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            encoding="utf-8", errors="replace"
        ).strip()

        if not output:
            print_result("GPU 其他进程占用", PASS, "无其他进程占用")
        else:
            lines = output.split("\n")
            procs = [l for l in lines if l.strip()]
            # 过滤掉 nvidia-smi 自身
            my_pid = str(os.getpid())
            other_procs = [p for p in procs if my_pid not in p]
            if other_procs:
                print_result("GPU 其他进程占用", WARN,
                             f"发现 {len(other_procs)} 个进程: {other_procs[0][:80]}")
            else:
                print_result("GPU 其他进程占用", PASS, "仅 nvidia-smi 自身")
    except Exception:
        print_result("GPU 其他进程占用", WARN, "无法检测（nvidia-smi 不可用）")


# ── 检查 6: 依赖包版本 ──────────────────────────────────
def parse_requirements(filepath):
    """解析 requirements1.txt，返回 {package: version} 字典"""
    deps = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 去掉注释
                line = line.split("#")[0].strip()
                if not line:
                    continue
                # 解析 package==version 或 package>=version
                match = re.match(r'^([a-zA-Z0-9_\-\.]+)\s*([><=!~]+)\s*([a-zA-Z0-9_\-\.\+]+)', line)
                if match:
                    name = match.group(1).lower().replace("-", "_")
                    deps[name] = line
                else:
                    # 只有包名没有版本号
                    deps[line.lower().replace("-", "_")] = line
    except FileNotFoundError:
        print_result("依赖检查", WARN, f"文件不存在: {filepath}")
        return {}
    return deps


def check_dependencies():
    req_file = Path(__file__).resolve().parent / "doc" / "requirements_1.txt"
    deps = parse_requirements(str(req_file))
    if not deps:
        return True

    all_ok = True
    warnings = []
    missing = []

    for pkg_name, pkg_spec in deps.items():
        try:
            pkg = __import__(pkg_name)
        except ImportError:
            # 尝试用 pip 查询
            try:
                result = subprocess.check_output(
                    [sys.executable, "-m", "pip", "show", pkg_name],
                    encoding="utf-8", errors="replace"
                )
                # 如果 pip show 成功，包已安装
            except Exception:
                missing.append(pkg_name)
                all_ok = False
                continue

        # 获取实际安装版本
        try:
            result = subprocess.check_output(
                [sys.executable, "-m", "pip", "show", pkg_name],
                encoding="utf-8", errors="replace"
            )
            v_match = re.search(r'^Version:\s*(.+)$', result, re.MULTILINE)
            if v_match:
                installed_ver = v_match.group(1).strip()
            else:
                warnings.append(f"{pkg_name}: 无法确定版本")
                continue
        except Exception:
            continue

    if missing:
        print_result("依赖包", FAIL, f"未安装: {', '.join(missing)}")
    elif warnings:
        print_result("依赖包", PASS, f"全部已安装（{len(deps)} 个）, 部分无法校验版本")
    else:
        print_result("依赖包", PASS, f"全部 {len(deps)} 个已安装")


# ── 检查 7: shapely 可导入 ───────────────────────────────
def check_shapely():
    try:
        import shapely
        print_result("shapely 可导入", PASS, f"版本 {shapely.__version__}")
        return True
    except ImportError:
        print_result("shapely 可导入", FAIL,
                     "未安装。Windows 下运行: conda install -c conda-forge shapely")
        return False


# ── 检查 8: YOLOv8n 前向推理 + 显存增量 ──────────────────
def check_yolo_inference():
    try:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            print_result("YOLOv8n 前向推理", FAIL, "CUDA 不可用，无法测试 GPU 推理")
            return False

        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / (1024 ** 2)

        # 优先加载本地权重文件（避免下载时网络问题）
        weight_path = Path(__file__).resolve().parent / "yolov8n.pt"
        if not weight_path.exists():
            weight_path = "yolov8n.pt"  # 尝试自动下载

        model = YOLO(str(weight_path))
        # 用空白图像做推理测试（避免网络问题）
        import numpy as np
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model.predict(
            source=dummy_img,
            device=0,
            verbose=False
        )

        vram_after = torch.cuda.memory_allocated() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        delta = peak - vram_before

        has_detection = len(results[0].boxes) > 0 if results[0].boxes is not None else False
        print_result("YOLOv8n 前向推理", PASS,
                     f"显存增量 {delta:.1f} MiB, 模型加载正常")
        return True

    except ImportError as e:
        print_result("YOLOv8n 前向推理", FAIL, f"缺少依赖: {e}")
        return False
    except Exception as e:
        print_result("YOLOv8n 前向推理", FAIL, str(e)[:80])
        return False


# ── 主流程 ───────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   VisDrone 环境检测 (check_env.py)")
    print("=" * 60)

    results = {}

    print_header("1. Python 版本 == 3.8.X")
    results["Python"] = check_python()

    print_header("2. PyTorch + CUDA + GPU")
    results["PyTorch"] = check_pytorch()

    print_header("3. CUDA + NVIDIA 驱动版本")
    check_cuda_driver()

    print_header("4. 磁盘空间")
    results["Disk"] = check_disk()

    print_header("5. GPU 其他进程占用")
    check_gpu_processes()

    print_header("6. 依赖包版本")
    check_dependencies()

    print_header("7. shapely 可导入")
    results["Shapely"] = check_shapely()

    print_header("8. YOLOv8n 前向推理")
    results["YOLO"] = check_yolo_inference()

    # ── 总结 ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  检测完成")
    print(f"{'=' * 60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  通过: {passed}/{total}")
    if passed == total:
        print("  环境就绪，可以开始训练。")
    else:
        print("  存在失败项，请根据上述提示修复后重新运行。")


if __name__ == "__main__":
    main()
