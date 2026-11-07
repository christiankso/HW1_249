"""Combine all experiment outputs into report tables and figures.

Run from HW1249:
    .venv/Scripts/python scripts/analyze.py
Writes experiments/analysis/{summary.md, summary.json, *.png}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "experiments"
OUT = os.path.join(EXP, "analysis")
os.makedirs(OUT, exist_ok=True)

SHORT = {
    "torchvision:fasterrcnn_resnet50_fpn_v2": "Faster R-CNN",
    "torchvision:fcos_resnet50_fpn": "FCOS",
    "yolo:yolo11x.pt": "YOLO11x",
    "hf_detr:PekingU/rtdetr_r50vd": "RT-DETR",
    "gdino:IDEA-Research/grounding-dino-tiny": "G-DINO-T",
}
ORDER = list(SHORT.values())


def load(*files):
    rows = {}
    for f in files:
        for r in json.load(open(os.path.join(EXP, f))):
            rows[SHORT[r["_meta"]["model"]]] = r
    return rows


coco = load("coco_full/metrics.json")
kitti = load("kitti_full/metrics.json", "kitti_cnn/metrics.json")
lat = {SHORT[r["spec"]]: r for r in json.load(open(os.path.join(EXP, "latency/latency.json")))
       if r.get("accel") == "fp32"} if os.path.exists(os.path.join(EXP, "latency/latency.json")) else {}
mem = {SHORT[r["model"]]: r for r in json.load(open(os.path.join(EXP, "latency/gpu_memory.json")))["rows"]}

KEYS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_100"]
lines = []


def table(title, rows):
    lines.append(f"\n### {title}\n")
    lines.append("| Model | " + " | ".join(KEYS) + " | vehicle | person | cyclist | images | sec |")
    lines.append("|---" * (len(KEYS) + 6) + "|")
    for m in ORDER:
        r = rows[m]
        pc = r["per_class"]
        lines.append(f"| {m} | " + " | ".join(f"{r[k]:.3f}" for k in KEYS)
                     + f" | {pc['vehicle']:.3f} | {pc['person']:.3f} | {pc['cyclist']:.3f}"
                     + f" | {r['_meta']['num_images']} | {r['_meta']['seconds']:.0f} |")


table("COCO val2017", coco)
table("KITTI (zero-shot)", kitti)

lines.append("\n### Relative drop COCO -> KITTI (%)\n")
cols = ["mAP", "AP_small", "AP_medium", "AP_large", "vehicle", "person", "cyclist"]
lines.append("| Model | " + " | ".join(cols) + " |")
lines.append("|---" * (len(cols) + 1) + "|")


def val(r, k):
    return r["per_class"][k] if k in r["per_class"] else r[k]


drops = {}
for m in ORDER:
    d = {k: 100 * (1 - val(kitti[m], k) / val(coco[m], k)) if val(coco[m], k) > 0 else float("nan") for k in cols}
    drops[m] = d
    lines.append(f"| {m} | " + " | ".join(f"{d[k]:.0f}" for k in cols) + " |")

lines.append("\n### Speed / memory (RTX 4070 SUPER, batch 1, fp32, KITTI image 1242x375)\n")
lines.append("| Model | mean ms | FPS | peak GPU MB | COCO mAP | KITTI mAP |")
lines.append("|---|---|---|---|---|---|")
for m in ORDER:
    l = lat.get(m, {})
    lines.append(f"| {m} | {l.get('mean_ms', float('nan')):.1f} | {l.get('fps', float('nan')):.1f} | "
                 f"{mem[m]['peak_mb']:.0f} | {coco[m]['mAP']:.3f} | {kitti[m]['mAP']:.3f} |")

open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines))
json.dump({"coco": coco, "kitti": kitti, "drops": drops, "latency": lat, "memory": mem},
          open(os.path.join(OUT, "summary.json"), "w"), indent=1)

# Figure 1: COCO vs KITTI mAP by object scale
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
x = range(len(ORDER))
for ax, k, t in zip(axes, ["AP_small", "AP_medium", "AP_large"], ["small", "medium", "large"]):
    ax.bar([i - 0.2 for i in x], [coco[m][k] for m in ORDER], 0.4, label="COCO", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], [kitti[m][k] for m in ORDER], 0.4, label="KITTI", color="#DD8452")
    ax.set_xticks(list(x)); ax.set_xticklabels(ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"AP ({t} objects)")
axes[0].set_ylabel("AP@[.5:.95]"); axes[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ap_by_scale.png"), dpi=200)

# Figure 2: per-class COCO vs KITTI
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, c in zip(axes, ["vehicle", "person", "cyclist"]):
    ax.bar([i - 0.2 for i in x], [coco[m]["per_class"][c] for m in ORDER], 0.4, label="COCO", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], [kitti[m]["per_class"][c] for m in ORDER], 0.4, label="KITTI", color="#DD8452")
    ax.set_xticks(list(x)); ax.set_xticklabels(ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"class: {c}")
axes[0].set_ylabel("AP@[.5:.95]"); axes[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "ap_by_class.png"), dpi=200)

# Figure 3: speed-accuracy
if lat:
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ORDER:
        ax.scatter(lat[m]["mean_ms"], coco[m]["mAP"], s=mem[m]["peak_mb"] / 4, color="#4C72B0", alpha=.7)
        ax.scatter(lat[m]["mean_ms"], kitti[m]["mAP"], s=mem[m]["peak_mb"] / 4, color="#DD8452", alpha=.7)
        ax.annotate(m, (lat[m]["mean_ms"], coco[m]["mAP"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("latency per image, ms (log scale)"); ax.set_ylabel("mAP@[.5:.95]")
    ax.scatter([], [], color="#4C72B0", label="COCO"); ax.scatter([], [], color="#DD8452", label="KITTI")
    ax.legend(title="marker size = peak GPU memory", fontsize=8)
    ax.set_title("Speed vs accuracy (RTX 4070 SUPER, fp32)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "speed_accuracy.png"), dpi=200)

print("\n".join(lines))
