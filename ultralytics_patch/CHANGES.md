# Changes against Ultralytics 8.4.166

Apply `edgets_ultralytics-8.4.166.diff` to an Ultralytics 8.4.166 source tree (`patch -p1`), or copy `modified_files/ultralytics/` over the installed package.

| File | Change |
|---|---|
| `utils/loss.py` | `wasserstein_similarity()` (NWD) and `BboxLoss(reg_max, nwd_alpha=0, nwd_c=14.26)`: L_box = α(1 − NWD) + (1 − α)(1 − CIoU), with NWD computed in pixels of the network input (predicted and target boxes are multiplied by the stride of their feature level). `nwd_alpha=0` runs the unchanged CIoU expression. |
| `cfg/default.yaml`, `cfg/__init__.py` | New training arguments `nwd_alpha` (default 0.0) and `nwd_c` (default 14.26). |
| `nn/modules/coordatt.py` (new) | `CoordAtt` (coordinate attention) and `CoordAttRes` (residual form y = x + γ·CA(x), per-channel γ initialized to zero). Evaluated in the ablation only; not part of EdgeTS-YOLO. |
| `nn/modules/__init__.py`, `nn/tasks.py` | Registration of the attention modules in the model parser. |

**NWD constant.** C = 14.26 pixels is the mean √(wh) of the 14,019 TT100K45 training instances at the 640 × 640 network input (median 11.41).

**Model configurations** (`configs/models/`, nc = 45; the backbone is unchanged in all of them).

| Config | Heads | Params (M) | GFLOPs (640) |
|---|---|---|---|
| `yolo11n.pt` (official) | P3–P5 | 2.599 | 6.55 |
| `yolo11n-p2.yaml` | P2–P5 | 2.686 | 10.71 |
| `yolo11n-p2p4.yaml` (EdgeTS-YOLO) | P2–P4 | 1.952 | 10.12 |
| `yolo11n-ca.yaml`, `yolo11n-cares.yaml` | P3–P5 + CA | 2.610 | 6.56 |
| `yolo11n-p2p4-cares.yaml` | P2–P4 + residual CA | 1.957 | 10.13 |

**Pretrained initialization** (`training/make_init_weights.py`). Ultralytics transfers pretrained weights by layer index and shape. Because the added P2 branch and attention layers shift the indices of the neck and head, the script maps every official YOLO11n layer to its structurally identical counterpart and writes `<config>_init.pt`. Layers without a counterpart are initialized randomly by the training run.

**Checks.** NWD agrees with an independent NumPy implementation (maximum difference 6e-8 on 200 random box pairs); with `nwd_alpha=0` the box loss is bit-identical to the original CIoU loss; with the attention layers replaced by identity, the remapped initial weights reproduce the outputs of the official YOLO11n exactly.
