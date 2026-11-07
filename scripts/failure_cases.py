"""Draw failure cases on KITTI: GT (green, dashed) vs predictions (solid, per-class color).

Also measures the cyclist box mismatch: for every KITTI cyclist GT box, the best IoU
achieved by a predicted "cyclist" (COCO bicycle) box vs. the union of the best
bicycle + person boxes.

Run from HW1249/DeepDataMiningLearning:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 ../.venv/Scripts/python ../scripts/failure_cases.py
"""
import json
import os

import numpy as np
from PIL import ImageDraw, ImageFont

from DeepDataMiningLearning.ngdet.datasets import EvalDataset
from DeepDataMiningLearning.ngdet.detectors.base import build_detector
from DeepDataMiningLearning.ngdet.taxonomy import Taxonomy

OUT = "../experiments/failures"
os.makedirs(OUT, exist_ok=True)
MODELS = {"YOLO11x": "yolo:yolo11x.pt", "G-DINO-T": "gdino:IDEA-Research/grounding-dino-tiny"}
COLORS = {0: (255, 140, 0), 1: (220, 20, 60), 2: (30, 144, 255)}  # vehicle, person, cyclist
N_SCAN = 600        # images scanned for cyclist statistics
N_DRAW = 4          # cyclist images drawn per model

tax = Taxonomy.from_preset("driving3")
ds = EvalDataset("kitti", "../data/kitti", tax)
CYC, PER = tax.classes.index("cyclist"), tax.classes.index("person")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def draw(sample, det, path, title):
    im = sample.image.copy()
    d = ImageDraw.Draw(im)
    for b in sample.gt_boxes:
        d.rectangle(list(map(float, b)), outline=(0, 255, 0), width=3)
    for b, l, s in zip(det.boxes, det.labels, det.scores):
        d.rectangle(list(map(float, b)), outline=COLORS[int(l)], width=2)
        d.text((float(b[0]) + 2, float(b[1]) + 1), f"{tax.classes[int(l)]} {s:.2f}", fill=COLORS[int(l)])
    d.rectangle([0, 0, 460, 16], fill=(0, 0, 0))
    d.text((4, 2), title + "   GT=green  vehicle=orange person=red cyclist=blue", fill=(255, 255, 255))
    im.save(path)


stats = {}
for name, spec in MODELS.items():
    det_model = build_detector(spec, tax, device="cuda", score_thr=0.3)
    best_cyc, best_union, drawn = [], [], 0
    for i in range(min(N_SCAN, len(ds))):
        s = ds[i]
        cyc_gt = s.gt_boxes[s.gt_labels == CYC]
        if len(cyc_gt) == 0:
            continue
        det = det_model.predict(s.image)
        pb = det.boxes[det.labels == CYC]
        pp = det.boxes[det.labels == PER]
        for g in cyc_gt:
            bc = max([iou(g, p) for p in pb], default=0.0)
            best_cyc.append(bc)
            # Union of best-matching bicycle box and best-matching person box.
            if len(pb) and len(pp):
                b = pb[np.argmax([iou(g, p) for p in pb])]
                p = pp[np.argmax([iou(g, q) for q in pp])]
                u = [min(b[0], p[0]), min(b[1], p[1]), max(b[2], p[2]), max(b[3], p[3])]
                best_union.append(iou(g, u))
        if drawn < N_DRAW:
            draw(s, det, os.path.join(OUT, f"{name}_cyclist_{s.image_id:06d}.png"), f"{name} | KITTI #{s.image_id}")
            drawn += 1
    bc, bu = np.array(best_cyc), np.array(best_union)
    stats[name] = {
        "cyclist_gt_boxes": int(len(bc)),
        "mean_best_iou_cyclist_pred": float(bc.mean()),
        "frac_cyclist_pred_iou>=0.5": float((bc >= 0.5).mean()),
        "frac_no_cyclist_pred_overlap": float((bc == 0).mean()),
        "n_with_bike_and_person": int(len(bu)),
        "mean_iou_union_bike_person": float(bu.mean()) if len(bu) else None,
        "frac_union_iou>=0.5": float((bu >= 0.5).mean()) if len(bu) else None,
    }
    print(name, json.dumps(stats[name], indent=1), flush=True)
    del det_model

json.dump({"scanned_images": N_SCAN, "stats": stats}, open(os.path.join(OUT, "cyclist_iou_stats.json"), "w"), indent=2)
