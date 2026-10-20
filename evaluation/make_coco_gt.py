"""Build a COCO-format GT json from a YOLO-format split (TT100K45). image id = int(file stem), category id = class + 1
(matches Ultralytics predictions.json for non-COCO data); area = w*h in ORIGINAL image pixels (same size bins as Table 1)."""
import json, sys, os, glob
from PIL import Image
root, split, names_yaml, out = sys.argv[1:5]
import yaml
names = yaml.safe_load(open(names_yaml))['names']
imgs, anns, aid = [], [], 1
for p in sorted(glob.glob(f"{root}/images/{split}/*.jpg")):
    stem = os.path.splitext(os.path.basename(p))[0]; iid = int(stem)
    W, H = Image.open(p).size
    imgs.append(dict(id=iid, file_name=os.path.basename(p), width=W, height=H))
    lp = f"{root}/labels/{split}/{stem}.txt"
    for line in (open(lp).read().split("\n") if os.path.exists(lp) else []):
        if not line.strip(): continue
        c, x, y, w, h = map(float, line.split())
        bw, bh = w * W, h * H
        anns.append(dict(id=aid, image_id=iid, category_id=int(c) + 1, bbox=[x * W - bw / 2, y * H - bh / 2, bw, bh], area=bw * bh, iscrowd=0)); aid += 1
cats = [dict(id=k + 1, name=v) for k, v in sorted(names.items())]
json.dump(dict(images=imgs, annotations=anns, categories=cats), open(out, "w"))
print(split, len(imgs), "images", len(anns), "annotations")
