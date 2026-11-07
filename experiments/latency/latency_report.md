# ngdet latency benchmark

Device: **NVIDIA GeForce RTX 4070 SUPER**. End-to-end `predict()` latency (includes pre/post-processing).

| model | accel | mean ms | p50 ms | p90 ms | FPS | speedup |
|---|---|---|---|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | fp32 | 39.4 | 39.1 | 43.1 | 25.4 | x1.00 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | fp16 | 25.8 | 25.2 | 27.7 | 38.8 | x1.53 |
| torchvision:fcos_resnet50_fpn | fp32 | 26.4 | 25.2 | 29.9 | 37.9 | x1.00 |
| torchvision:fcos_resnet50_fpn | fp16 | 25.6 | 24.6 | 29.3 | 39.1 | x1.03 |
| yolo:yolo11x.pt | fp32 | 18.2 | 17.4 | 22.0 | 54.9 | x1.00 |
| yolo:yolo11x.pt | fp16 | 20.7 | 19.6 | 24.3 | 48.4 | x0.88 |
| hf_detr:rtdetr_r50vd | fp32 | 32.8 | 32.1 | 35.7 | 30.4 | x1.00 |
| hf_detr:rtdetr_r50vd | fp16 | 35.2 | 33.7 | 40.1 | 28.4 | x0.93 |
| gdino:grounding-dino-tiny | fp32 | 172.9 | 170.8 | 184.7 | 5.8 | x1.00 |
| gdino:grounding-dino-tiny | fp16 | 171.5 | 169.5 | 179.2 | 5.8 | x1.01 |