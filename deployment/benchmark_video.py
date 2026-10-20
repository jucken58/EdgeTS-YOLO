"""Table 9: end-to-end latency on the Jetson for an offline dashcam video (1920x1080, 29.97 fps, H.264), batch 1.
Per frame: decode (cv2.VideoCapture.read, CPU), then the Ultralytics predictor on the TensorRT engine, which reports
preprocess (letterbox + to GPU tensor), inference (TensorRT) and postprocess (NMS + box rescaling) in ms (CUDA-synchronised).
The wall-clock time of the whole loop is also recorded, so total throughput includes every overhead.
Deployment thresholds: conf 0.25, iou 0.7, max_det 300 (Ultralytics predict defaults). The first --warmup frames are excluded.
Output: per-frame CSV + JSON summary (mean / median / p99 of each stage, end-to-end FPS, mean detections per frame).
Usage: python3 benchmark_video.py --engine model_ul.engine --video v.mp4 --out prefix [--warmup 100]
"""
import argparse, csv, json, time
import numpy as np, cv2
if "bool" not in np.__dict__:
    np.bool = np.bool_  # TensorRT 8.5 tensorrt.nptype uses np.bool, removed in NumPy >= 1.24 (NX has 1.24.4)
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True); ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--iou", type=float, default=0.7)
a = ap.parse_args()
model = YOLO(a.engine, task="detect")
cap = cv2.VideoCapture(a.video); assert cap.isOpened(), a.video
info = dict(width=cap.get(3), height=cap.get(4), fps=cap.get(5), frames=int(cap.get(7)))
rows, i = [], 0
t_start = None
while True:
    t0 = time.perf_counter(); ok, im = cap.read(); t1 = time.perf_counter()
    if not ok: break
    r = model.predict(im, imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=300, verbose=False)[0]
    t2 = time.perf_counter()
    if i == a.warmup: t_start = t0
    if i >= a.warmup:
        s = r.speed
        rows.append([i, (t1 - t0) * 1e3, s["preprocess"], s["inference"], s["postprocess"], (t2 - t0) * 1e3, len(r.boxes)])
    i += 1
t_end = time.perf_counter()
with open(a.out + "_frames.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "wall_ms", "n_det"]); w.writerows(rows)
A = np.array([r[1:] for r in rows], dtype=float)
st = lambda x: dict(mean=round(float(x.mean()), 3), median=round(float(np.median(x)), 3), p99=round(float(np.percentile(x, 99)), 3))
summ = dict(engine=a.engine, video=a.video, video_info=info, frames_total=i, frames_measured=len(rows), warmup=a.warmup, conf=a.conf, iou=a.iou,
            decode_ms=st(A[:, 0]), preprocess_ms=st(A[:, 1]), inference_ms=st(A[:, 2]), postprocess_ms=st(A[:, 3]),
            sum_of_stages_ms=st(A[:, 0] + A[:, 1] + A[:, 2] + A[:, 3]), wall_ms_per_frame=st(A[:, 4]),
            end_to_end_fps=round(len(rows) / (t_end - t_start), 2), mean_detections=round(float(A[:, 5].mean()), 3))
json.dump(summ, open(a.out + "_summary.json", "w"), indent=1)
print(json.dumps({k: summ[k] for k in ["frames_measured", "decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "wall_ms_per_frame", "end_to_end_fps", "mean_detections"]}))
