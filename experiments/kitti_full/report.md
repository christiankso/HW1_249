# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo11x.pt | 0.229 |
| hf_detr:PekingU/rtdetr_r50vd | 0.239 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.179 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo11x.pt | 0.418 |
| hf_detr:PekingU/rtdetr_r50vd | 0.437 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.343 |

## Full COCO tables (per run)

### yolo:yolo11x.pt  ×  kitti  (5984 imgs, 218.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.229 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.418 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.230 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.129 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.217 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.304 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.106 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.295 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.299 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.224 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.296 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.366 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.459 |
| person | 0.229 |
| cyclist | 0.001 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (5984 imgs, 290.9s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.239 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.437 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.234 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.129 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.227 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.314 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.105 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.299 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.309 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.245 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.305 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.371 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.482 |
| person | 0.232 |
| cyclist | 0.002 |

### gdino:IDEA-Research/grounding-dino-tiny  ×  kitti  (5984 imgs, 1351.5s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.179 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.343 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.172 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.104 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.180 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.257 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.107 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.342 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.350 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.305 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.345 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.411 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.319 |
| person | 0.201 |
| cyclist | 0.018 |
