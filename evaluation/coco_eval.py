"""COCO-style evaluation (AP, AP50, AP75, AP_S, AP_M, AP_L) from Ultralytics predictions.json.
Requires: pip install pycocotools
gt: COCO-format json of the test split whose image "id" values match the predictions' image_id
(Ultralytics uses the image file stem as image_id; make the GT ids match).
"""
import argparse, json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
gt = COCO(a.gt); dt = gt.loadRes(a.pred)
E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
keys = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]
json.dump({k: round(float(v) * 100, 2) for k, v in zip(keys, E.stats[:6])}, open(a.out, "w"), indent=1)
