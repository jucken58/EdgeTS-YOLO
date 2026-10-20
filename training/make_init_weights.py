
"""EdgeTS-YOLO: build COCO-initialised checkpoints for the ablation configs by remapping layer indices.

Ultralytics transfers pretrained weights by (key name, shape). Adding the P2 branch or CoordAtt layers shifts the layer
indices of the neck/Detect head, so a plain `pretrained=yolo11n.pt` silently drops those weights. This script copies every
official layer to its structurally identical counterpart in the modified network and saves an init checkpoint.
Usage: python make_init_weights.py <ultralytics_src_dir> <official.pt> <config.yaml> <out.pt>
"""
import re, sys
sys.path.insert(0, sys.argv[1])
import torch
from ultralytics.nn.tasks import DetectionModel, load_checkpoint

src_pt, cfg, out = sys.argv[2], sys.argv[3], sys.argv[4]
name = cfg.split("/")[-1]
p2, p5, ca = "-p2" in name, "p2p4" not in name, "-ca" in name
# base layout (YOLO11n / YOLO26n): 0-10 backbone, 11-16 top-down, 17-19 P4 out, 20-22 P5 out, 23 Detect(P3,P4,P5)
if not p2:
    layer = {i: i for i in range(23)}
    det_new, lvl = (26 if ca else 23), {0: 0, 1: 1, 2: 2}
else:
    layer = {i: i for i in range(17)}
    layer.update({17: 23, 18: 24, 19: 25})          # P3->P4 downsample, concat, P4 output block
    if p5:
        layer.update({20: 26, 21: 27, 22: 28})      # P4->P5 downsample, concat, P5 output block
        det_new, lvl = (33 if ca else 29), {0: 1, 1: 2, 2: 3}
    else:
        det_new, lvl = (29 if ca else 26), {0: 1, 1: 2}  # base P5 output block and P5 head have no counterpart
src_model, _ = load_checkpoint(src_pt, device="cpu")
sd_src = src_model.float().state_dict()
new = DetectionModel(cfg, nc=src_model.yaml.get("nc", 80), verbose=False)   # keep nc=80: cls convs re-init later exactly like the baseline
sd_new = new.state_dict()
mapped, n_ok = {}, 0
for k, v in sd_src.items():
    m = re.match(r"model\.(\d+)\.(.*)", k)
    if not m: continue
    i, rest = int(m.group(1)), m.group(2)
    if i == 23:
        mm = re.match(r"((?:one2one_)?cv[23])\.(\d+)\.(.*)", rest)
        if mm:
            L = int(mm.group(2))
            if L not in lvl: continue
            rest = f"{mm.group(1)}.{lvl[L]}.{mm.group(3)}"
        nk = f"model.{det_new}.{rest}"
    elif i in layer:
        nk = f"model.{layer[i]}.{rest}"
    else:
        continue
    if nk in sd_new and sd_new[nk].shape == v.shape:
        mapped[nk] = v; n_ok += 1
sd_new.update(mapped); new.load_state_dict(sd_new)
# Tensors with no official counterpart (P2 branch, CoordAtt, ...) are emptied so that Ultralytics' (name, shape)
# matching skips them and they are initialised by the training run itself (i.e. they follow the run's seed).
n_new = 0
for nm, t in list(new.named_parameters()) + list(new.named_buffers()):
    if nm not in mapped:
        t.data = torch.empty(0, dtype=t.dtype); n_new += 1
new.args = getattr(src_model, "args", {})
torch.save({"model": new, "ema": None, "epoch": -1, "date": None, "version": "8.4.166",
            "note": f"EdgeTS-YOLO init for {name}: {n_ok}/{len(sd_src)} official tensors remapped from {src_pt.split('/')[-1]}; {n_new} new tensors left to the training seed"}, out)
print(f"{name}: remapped {n_ok}/{len(sd_src)} official tensors, {n_new} new tensors left to seed -> {out}")
