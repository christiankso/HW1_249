"""Measure peak GPU memory per detector (fp32, batch 1) on one KITTI image.

Run from HW1249/DeepDataMiningLearning:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 ../.venv/Scripts/python ../scripts/gpu_memory.py

Note: "weights_mb" for YOLO is near zero because Ultralytics moves weights to the
GPU lazily on the first predict(); use "peak_mb" to compare models.
"""
import gc
import json
import sys

import torch
from PIL import Image

from DeepDataMiningLearning.ngdet.detectors.base import build_detector
from DeepDataMiningLearning.ngdet.taxonomy import Taxonomy

MODELS = [
    "torchvision:fasterrcnn_resnet50_fpn_v2",
    "torchvision:fcos_resnet50_fpn",
    "yolo:yolo11x.pt",
    "hf_detr:PekingU/rtdetr_r50vd",
    "gdino:IDEA-Research/grounding-dino-tiny",
]
IMAGE = "../data/kitti/training/image_2/000010.png"
OUT = "../experiments/latency/gpu_memory.json"

tax = Taxonomy.from_preset("driving3")
img = Image.open(IMAGE).convert("RGB")
rows = []
for spec in MODELS:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    det = build_detector(spec, tax, device="cuda", score_thr=0.3)
    weights_mb = torch.cuda.memory_allocated() / 2**20
    for _ in range(5):
        det.predict(img)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    rows.append({"model": spec, "weights_mb": round(weights_mb, 1), "peak_mb": round(peak_mb, 1)})
    print(f"{spec:42s} weights {weights_mb:8.1f} MB   peak {peak_mb:8.1f} MB", flush=True)
    del det
    gc.collect()

json.dump({"device": torch.cuda.get_device_name(0), "image": IMAGE, "rows": rows}, open(OUT, "w"), indent=2)
print("saved", OUT, file=sys.stderr)
