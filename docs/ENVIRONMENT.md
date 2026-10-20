# Software environments

## Training and evaluation
- NVIDIA GeForce RTX 4080 SUPER (32 GB variant), Ubuntu 22.04.5, Python 3.12.3
- PyTorch 2.8.0 + CUDA 12.8, torchvision 0.23.0, numpy 2.3.2, pycocotools 2.0.11
- Ultralytics 8.4.166 with `ultralytics_patch/` applied

## Deployment (NVIDIA Jetson Xavier NX Developer Kit, 8 GB)
- JetPack 5.1.6 (Jetson Linux R35.6.5, Ubuntu 20.04.6), CUDA 11.4.315, cuDNN 8.6.0.166, TensorRT 8.5.2.2
- Python 3.8.10, PyTorch 2.1.0 (NVIDIA build nv23.06), torchvision 0.16.1, numpy 1.24.4, onnx 1.17.0
- Ultralytics 8.4.166 with the same patch
- The TensorRT 8.5 Python bindings use the removed alias `np.bool`; `deployment/val_engine.py` and `deployment/benchmark_video.py` restore it (`np.bool = np.bool_`), which does not change any numerical result.
