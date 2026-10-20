"""`yolo detect val` for a TensorRT engine on the Jetson, with a compatibility shim:
TensorRT 8.5's Python binding (tensorrt.nptype) uses the alias np.bool, which NumPy >= 1.24 removed (the NX has numpy 1.24.4 in
~/.local). Setting np.bool = np.bool_ restores the alias without changing any numerical behaviour. Everything else is the
standard Ultralytics validator (same arguments as the val/test evaluation on the cloud: imgsz 640 or the optional 6th argument, batch 1, save_json).
Usage: python3 val_engine.py <engine_ul.engine> <data.yaml> <split> <project> <name> [imgsz]"""
import sys
import numpy as np
if not hasattr(np, "bool") or "bool" not in np.__dict__:
    np.bool = np.bool_
from ultralytics import YOLO
eng, data, split, project, name = sys.argv[1:6]
imgsz = int(sys.argv[6]) if len(sys.argv) > 6 else 640
m = YOLO(eng, task="detect")
r = m.val(data=data, split=split, imgsz=imgsz, batch=1, save_json=True, plots=False, project=project, name=name, exist_ok=True)
print("ULTRALYTICS_mAP50-95", round(float(r.box.map), 4), "mAP50", round(float(r.box.map50), 4))
