# Training

All models are trained with Ultralytics 8.4.166 (patched) on TT100K45 for 300 epochs without early stopping; the checkpoint with the highest validation mAP@0.5:0.95 (`best.pt`) is evaluated.

```
yolo detect train data=configs/TT100K45.yaml model=<model> pretrained=<weights> imgsz=<640|896|1024|1280> seed=<0-4> \
  epochs=300 patience=0 batch=32 optimizer=SGD lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=0.0005 \
  warmup_epochs=3 warmup_momentum=0.8 warmup_bias_lr=0.1 cos_lr=False close_mosaic=10 deterministic=True amp=True \
  box=7.5 cls=0.5 dfl=1.5 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=0.0 translate=0.1 scale=0.5 shear=0.0 \
  perspective=0.0 flipud=0.0 fliplr=0.0 mosaic=1.0 mixup=0.0 copy_paste=0.0 erasing=0.4 iou=0.7 max_det=300 \
  cache=ram [nwd_alpha=0.5 nwd_c=14.26]
```

| Model | `model` | `pretrained` | NWD arguments |
|---|---|---|---|
| YOLO11n | `yolo11n.pt` | (COCO weights) | – |
| EdgeTS-YOLO | `configs/models/yolo11n-p2p4.yaml` | `yolo11n-p2p4_init.pt` | `nwd_alpha=0.5 nwd_c=14.26` |
| P2–P4 (ablation) | `configs/models/yolo11n-p2p4.yaml` | `yolo11n-p2p4_init.pt` | `nwd_alpha=0` |
| YOLO11s | `yolo11s.pt` | (COCO weights) | – |

- `<name>_init.pt`: COCO-pretrained YOLO11n weights remapped to the modified network with `make_init_weights.py` (layers with matching shapes are copied; new layers keep their random initialization from the run's seed).
- `fliplr=0.0`: horizontal flipping changes the meaning of direction-specific signs.
- Input-resolution runs use the same arguments with a different `imgsz`; that size is also used for evaluation and for the TensorRT engine. The 1280-pixel YOLO11n runs and the 896-pixel EdgeTS-YOLO runs used `cache=False` (memory limit of the training instance); the preprocessing is otherwise identical. `nwd_c` is kept at 14.26 for all input sizes.
- The α sweep (Table 7) uses `epochs=150` and seed 0.
- The comparison models of Table 4 (YOLOv5nu, YOLOv8n, YOLOv10n, YOLOv12n, YOLO26n) use their COCO weights and the same arguments; RT-DETR-L uses the Ultralytics RT-DETR defaults (AdamW, lr0 1e-4, batch 16, 72 epochs).
