# PyTorch FP32 latency on the Jetson, batch 1, 640x640, same scope as trtexec "Latency" (H2D copy + forward + D2H copy).
import sys, time, json, numpy as np, torch
from ultralytics import YOLO
pt, out, n_warm, n_it = sys.argv[1], sys.argv[2], 100, 1000
m = YOLO(pt).model.float().cuda().eval()
for p in m.parameters(): p.requires_grad_(False)
x = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
ts = []
with torch.inference_mode():
    for i in range(n_warm + n_it):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y = m(torch.from_numpy(x).cuda())
        y = y[0] if isinstance(y, (list, tuple)) else y
        y = y.cpu(); torch.cuda.synchronize(); t1 = time.perf_counter()
        if i >= n_warm: ts.append((t1 - t0) * 1000)
a = np.array(ts); r = dict(pt=pt, n=len(a), mean=float(a.mean()), median=float(np.median(a)), p99=float(np.percentile(a, 99)), fps=float(1000 / a.mean()), torch=torch.__version__)
json.dump(r, open(out, "w"), indent=1); print(json.dumps(r))
