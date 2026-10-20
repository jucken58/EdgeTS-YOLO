"""Table 5: cross-dataset evaluation of TT100K45-trained models on the CCTSDB 2021 positive test set (no retraining).
Predictions of the 45 TT100K classes are collapsed to the 3 CCTSDB super-classes by prefix (i->mandatory, p->prohibitory,
w->warning; table in 01_dataset/cctsdb/tt100k45_to_cctsdb3.csv). Because several TT100K classes map to one super-class,
duplicate boxes can appear on one sign after collapsing; the primary result therefore re-applies per-super-class NMS
(IoU 0.7, same as Ultralytics val) after mapping ('renms'); the result without re-NMS ('raw') is also saved.
Inference settings follow Ultralytics val: imgsz 640, conf 0.001, iou 0.7, max_det 300, batch 1 (letterbox, minimal padding).
COCO metrics (pycocotools) on the full test set and on every weather/lighting subset.
Usage: python cctsdb_eval.py --model best.pt --gtdir <dir with gt_cctsdb_test_*.json> --imgdir <test_img> --out <out_dir> [--limit N]
"""
import argparse, json, os, glob
import torch, torchvision
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--gtdir", required=True); ap.add_argument("--imgdir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
m = YOLO(a.model); names = m.names
assert len(names) == 45, names
SUPER = {"i": 1, "p": 2, "w": 3}
cmap = {k: SUPER[v[0]] for k, v in names.items()}
imgs = sorted(glob.glob(os.path.join(a.imgdir, "*.jpg")))
if a.limit: imgs = imgs[:a.limit]
raw, renms = [], []
for p in imgs:
    iid = int(os.path.basename(p)[:-4])
    r = m.predict(p, imgsz=640, conf=0.001, iou=0.7, max_det=300, verbose=False, device=a.device)[0]
    b = r.boxes; xyxy = b.xyxy.cpu(); sc = b.conf.cpu(); cl = b.cls.cpu().long()
    sup = torch.tensor([cmap[int(c)] for c in cl], dtype=torch.long) if len(cl) else torch.zeros(0, dtype=torch.long)
    for lst, keep in ((raw, torch.arange(len(sc))), (renms, torchvision.ops.batched_nms(xyxy.float(), sc.float(), sup, 0.7) if len(sc) else torch.zeros(0, dtype=torch.long))):
        for i in keep.tolist()[:300]:
            x0, y0, x1, y1 = xyxy[i].tolist()
            lst.append({"image_id": iid, "category_id": int(sup[i]), "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)], "score": round(float(sc[i]), 5), "tt100k_class": names[int(cl[i])]})
done = {int(os.path.basename(p)[:-4]) for p in imgs}
res = {}
for tag, dets in (("renms", renms), ("raw", raw)):
    json.dump(dets, open(os.path.join(a.out, f"predictions_cctsdb_{tag}.json"), "w"))
    for gtf in sorted(glob.glob(os.path.join(a.gtdir, "gt_cctsdb_test_*.json"))):
        sub = os.path.basename(gtf)[len("gt_cctsdb_test_"):-5]
        gt = COCO(gtf); ids = [i for i in gt.getImgIds() if i in done]
        if not ids: continue
        d = [x for x in dets if x["image_id"] in set(ids)]
        if not d: continue
        E = COCOeval(gt, gt.loadRes(d), "bbox"); E.params.imgIds = ids; E.evaluate(); E.accumulate(); E.summarize()
        res[f"{tag}/{sub}"] = dict(zip(["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"], [round(float(v) * 100, 2) for v in E.stats[:6]]), n_images=len(ids))
json.dump({"model": a.model, "n_images": len(imgs), "results": res}, open(os.path.join(a.out, "cctsdb_coco.json"), "w"), indent=1)
print(json.dumps({k: (v["AP"], v["AP50"]) for k, v in res.items()}))
