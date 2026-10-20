"""Build a TensorRT 8.5 engine from a static 1x3xNxN ONNX on the Jetson (N = environment variable IMGSZ, default 640) (Table 8 / S2 INT8 sensitivity).
Precisions:
  fp32               no reduced precision
  fp16               FP16 allowed everywhere (equivalent to trtexec --fp16)
  int8               INT8 + FP16 allowed everywhere, no per-layer constraints ("INT8-full")
  int8 --ul-head     Ultralytics 8.4.166 default INT8 scheme: head Sigmoid in FP32, head output convs (*.2/Conv) and DFL in FP16
  int8 --fp16-groups "<regex>,<regex>"   force every layer whose name matches one of the regexes to FP16 (sensitivity / mixed precision)
Calibration: --calib-algo minmax (IInt8MinMaxCalibrator, the Ultralytics default) or entropy2 (IInt8EntropyCalibrator2; the algorithm is chosen on the validation set), batch 1, images from --calib-list (training images only),
preprocessed exactly like Ultralytics validation (LetterBox to N, BGR->RGB, /255). The calibration cache is written to /
read from --cache so all INT8 variants of one model share identical tensor ranges.
Output: <out>.engine (raw, for trtexec timing) and <out>_ul.engine (same bytes with the Ultralytics metadata header, for yolo val).
The metadata is copied from the ONNX file written by Ultralytics export.
"""
import argparse, ast, json, re, os, sys, time
import numpy as np, cv2, onnx, torch, tensorrt as trt  # torch provides the CUDA context and device buffers (no pycuda on the NX)

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
ap.add_argument("--calib-list"); ap.add_argument("--cache"); ap.add_argument("--calib-algo", choices=["minmax", "entropy2"], default="minmax"); ap.add_argument("--workspace-mib", type=int, default=2048)
ap.add_argument("--ul-head", action="store_true"); ap.add_argument("--fp16-groups", default="")
ap.add_argument("--list-layers", action="store_true", help="only print layer names/types and exit")
a = ap.parse_args()
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger); config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_mib << 20)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
assert parser.parse_from_file(a.onnx), [parser.get_error(i) for i in range(parser.num_errors)]
if a.list_layers:
    for i in range(network.num_layers):
        l = network.get_layer(i); print(f"{i}\t{l.type}\t{l.name}")
    sys.exit(0)
inp = network.get_input(0); shape = tuple(inp.shape); assert shape == (1, 3, int(os.environ.get('IMGSZ', 640)), int(os.environ.get('IMGSZ', 640))), shape

class _CalibMixin:
    def _init(self, files, cache):
        self.files, self.cache, self.i = files, cache, 0
        self.dev = torch.empty(shape, dtype=torch.float32, device="cuda")
    def get_batch_size(self): return 1
    def get_batch(self, names):
        if self.i >= len(self.files): return None
        im = cv2.imread(self.files[self.i]); self.i += 1
        h, w = im.shape[:2]; Z = int(os.environ.get('IMGSZ', 640)); r = min(Z / h, Z / w); nh, nw = round(h * r), round(w * r)
        if (nh, nw) != (h, w): im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (Z - nh) // 2, (Z - nw) // 2
        im = cv2.copyMakeBorder(im, top, Z - nh - top, left, Z - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        self.dev.copy_(torch.from_numpy(x)); torch.cuda.synchronize(); return [int(self.dev.data_ptr())]
    def read_calibration_cache(self):
        return open(self.cache, "rb").read() if self.cache and os.path.exists(self.cache) else None
    def write_calibration_cache(self, c):
        if self.cache: open(self.cache, "wb").write(c)

class CalibMinMax(_CalibMixin, trt.IInt8MinMaxCalibrator):
    def __init__(self, files, cache):
        trt.IInt8MinMaxCalibrator.__init__(self); self._init(files, cache)

class CalibEntropy2(_CalibMixin, trt.IInt8EntropyCalibrator2):
    def __init__(self, files, cache):
        trt.IInt8EntropyCalibrator2.__init__(self); self._init(files, cache)

constrained = []
if a.precision in ("fp16", "int8"): config.set_flag(trt.BuilderFlag.FP16)
if a.precision == "int8":
    config.set_flag(trt.BuilderFlag.INT8)
    files = [l.strip() for l in open(a.calib_list) if l.strip()]
    config.int8_calibrator = (CalibMinMax if a.calib_algo == "minmax" else CalibEntropy2)(files, a.cache)
    pats = [re.compile(p) for p in a.fp16_groups.split(",") if p]
    names = [network.get_layer(i).name for i in range(network.num_layers)]
    idx = [int(m.group(1)) for n in names if (m := re.match(r"/model\.(\d+)/", n))]
    head = f"/model.{max(idx)}/" if idx else None
    for i in range(network.num_layers):
        l = network.get_layer(i); dt = None
        if a.ul_head and head and l.name.startswith(head):
            if l.type == trt.LayerType.ACTIVATION and "sigmoid" in l.name.lower(): dt = trt.float32
            elif l.type == trt.LayerType.CONVOLUTION and (l.name.endswith(".2/Conv") or "/dfl/" in l.name): dt = trt.float16
        if dt is None and any(p.search(l.name) for p in pats) and l.type not in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT, trt.LayerType.IDENTITY):
            dt = trt.float16
        if dt is not None:
            l.precision = dt
            for j in range(l.num_outputs): l.set_output_type(j, dt)
            constrained.append(l.name)
    if constrained: config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
t0 = time.time(); eng = builder.build_serialized_network(network, config); dt = time.time() - t0
print("CONSTRAINED_LAYERS", len(constrained), flush=True)
assert eng is not None, "build failed"
open(a.out + ".engine", "wb").write(eng)
meta = {p.key: p.value for p in onnx.load(a.onnx, load_external_data=False).metadata_props}
for k, v in list(meta.items()):  # Ultralytics stores python literals as strings in the ONNX metadata
    try: meta[k] = ast.literal_eval(v)
    except Exception: pass
m = json.dumps(meta)
with open(a.out + "_ul.engine", "wb") as f:
    f.write(len(m).to_bytes(4, byteorder="little", signed=True)); f.write(m.encode()); f.write(eng)
json.dump({"onnx": a.onnx, "precision": a.precision, "ul_head": a.ul_head, "fp16_groups": a.fp16_groups, "n_constrained_layers": len(constrained),
           "constrained_layers": constrained, "build_s": round(dt, 1), "engine_bytes": eng.nbytes, "calib_list": a.calib_list, "cache": a.cache, "calib_algo": a.calib_algo,
           "trt": trt.__version__}, open(a.out + "_build.json", "w"), indent=1)
print(f"BUILT {a.out}.engine precision={a.precision} constrained={len(constrained)} build_s={dt:.0f} bytes={eng.nbytes}")
