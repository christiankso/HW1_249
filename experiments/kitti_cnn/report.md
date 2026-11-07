# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.234 |
| torchvision:fcos_resnet50_fpn | 0.232 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.432 |
| torchvision:fcos_resnet50_fpn | 0.433 |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (5984 imgs, 330.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.234 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.432 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.233 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.119 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.227 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.308 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.109 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.312 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.328 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.304 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.320 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.368 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.447 |
| person | 0.254 |
| cyclist | 0.001 |

### torchvision:fcos_resnet50_fpn  ×  kitti  (5984 imgs, 248.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.232 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.433 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.227 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.121 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.218 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.312 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.110 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.320 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.342 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.277 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.348 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.388 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.455 |
| person | 0.237 |
| cyclist | 0.004 |
