# EdgeTS-YOLO

Code, configurations and data splits for the paper

> Keyuan Tang, Zhengcong Du, Chuanli Liu, Lianghua Wen, Fengping Sun, Rui Song, Chunlan Luo.
> *Input resolution versus a stride-4 detection head: Accuracy, latency and energy of YOLO11n-based traffic sign detection on the NVIDIA Jetson Xavier NX.* (submitted to PLOS ONE)

EdgeTS-YOLO is YOLO11n with a stride-4 (P2) detection branch, without the stride-32 head (P2–P4 head), trained with a hybrid NWD–CIoU box loss (α = 0.5, C = 14.26). The paper compares it on a Jetson Xavier NX (JetPack 5.1.6, TensorRT 8.5.2) with the alternative of enlarging the input of the unmodified YOLO11n.

## Main results (TT100K, 45 classes, official test set)
| Model | Input | mAP@0.5:0.95 (%, 5 seeds) | Jetson FP16 latency (ms) | Energy / inference (mJ) |
|---|---|---|---|---|
| YOLO11n | 640 | 60.00 ± 0.12 | 7.98 | 76 |
| EdgeTS-YOLO | 640 | 61.57 ± 0.78 | 13.00 | 139 |
| YOLO11n | 896 | 68.83 ± 0.12 | 13.81 | 156 |
| YOLO11n | 1024 | 71.02 ± 0.28 | 17.17 | 206 |
| EdgeTS-YOLO | 896 | 69.99 ± 0.40 | 23.54 | 293 |
| YOLO11n | 1280 | 73.68 ± 0.18 | 25.95 | 346 |

TensorRT FP16 engine, batch 1, 20 W mode with jetson_clocks, mean of three trtexec runs (latency of a single request). Per-run results are given in S5 Table of the paper.

## Contents
| Folder | Content |
|---|---|
| `ultralytics_patch/` | Patch against Ultralytics 8.4.166 and the modified files: NWD–CIoU loss (`utils/loss.py`), coordinate attention modules evaluated in the ablation, model parser and the training arguments `nwd_alpha`, `nwd_c` (see `CHANGES.md`). |
| `configs/` | `models/yolo11n-p2p4.yaml` (EdgeTS-YOLO; also the P2–P4 ablation with `nwd_alpha=0`), ablation variants, dataset YAML, INT8 calibration YAML. |
| `data/` | Split lists (train / val / test image IDs, the six duplicates removed from training, the 500 INT8 calibration images) and preparation scripts for TT100K and CCTSDB 2021. Images are not redistributed. |
| `training/` | Training commands and the remapping of COCO-pretrained weights to the modified networks. |
| `evaluation/` | COCO evaluation with pycocotools and the CCTSDB 2021 cross-dataset evaluation. |
| `deployment/` | Jetson scripts: engine building (FP32/FP16/INT8, calibrators, per-group FP16 constraints), on-device accuracy, trtexec + tegrastats benchmark, INT8 sensitivity analysis, end-to-end video timing, PyTorch latency. |
| `docs/` | Software environments and measurement protocol. |

## Reproducing
1. **Environment**: Python 3.12, PyTorch 2.8.0 + CUDA 12.8, `pip install ultralytics==8.4.166 pycocotools`; copy `ultralytics_patch/modified_files/ultralytics/` over the installed package, or apply `edgets_ultralytics-8.4.166.diff` to a source checkout (`patch -p1`). See `docs/ENVIRONMENT.md`.
2. **Data**: download TT100K (2016) and CCTSDB 2021 (`data/DOWNLOAD.md`) and run `data/scripts/prepare_tt100k.py`; the resulting splits must match `data/splits/*.txt`.
3. **Training**: `training/README.md`.
4. **Evaluation**: `evaluation/make_coco_gt.py`, then `yolo detect val ... save_json=True batch=1 conf=0.001 iou=0.7` and `evaluation/coco_eval.py`.
5. **Jetson**: `docs/MEASUREMENT_PROTOCOL.md`. Engines must be built on the target device. The shell scripts source `$HOME/edgets_env.sh`, which only sets `PATH` (trtexec) and the Python environment.

## Trained weights
`edgets-yolo_640_s0.pt` (EdgeTS-YOLO, 640 pixels, seed 0; the model used for all Jetson measurements of EdgeTS-YOLO at 640 pixels) is attached to the GitHub release.

## Data not included
- TT100K and CCTSDB 2021 images and annotations: obtain them from the official sources; the licences of the datasets apply.
- The two dashcam videos used for end-to-end timing contain identifiable license plates and people and are not shared publicly; the per-frame timing results are in S2 File of the paper, and the videos can be requested from the first author.

## Licence
The code is derived from Ultralytics (AGPL-3.0) and is released under the GNU AGPL-3.0 (`LICENSE`).

## Citation
See `CITATION.cff`.
