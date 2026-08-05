"""Batch prediction test — sends multiple images to the API."""
import os, sys, time, requests

IMG_DIR = "D:/Data/VisDrone/test-dev/images"
API_URL = "http://localhost:8000/predict"
N = 10  # how many images to test

sys.stdout.reconfigure(encoding='utf-8')

imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png')))[:N]
total_time = 0
total_dets = 0

for img_name in imgs:
    path = os.path.join(IMG_DIR, img_name)
    with open(path, 'rb') as f:
        resp = requests.post(API_URL, files={'file': f})
    data = resp.json()
    t = data['inference_time_ms']
    n = data['num_detections']
    total_time += t
    total_dets += n
    print(f"{img_name:45s} {n:3d} dets  {t:7.1f} ms")

print(f"\n---")
print(f"Total: {len(imgs)} images, {total_dets} detections, {total_time:.0f} ms")
print(f"Average: {total_time/len(imgs):.1f} ms/image, {total_dets/len(imgs):.1f} dets/image")
