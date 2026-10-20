"""Prepare the CCTSDB 2021 positive test set (1,500 images, 3,228 signs) for the Table 5 cross-dataset evaluation.
No training on CCTSDB. Steps: extract test_img / test_labels / weather-subset XMLs from the official zips,
cross-check the TXT labels against the XML labels (count, class, coordinates), write COCO ground truth
(categories 1 mandatory, 2 prohibitory, 3 warning; image id = int(file stem)) for the full test set and for each weather/lighting subset.
TXT class ids: 0 mandatory, 1 prohibitory, 2 warning (verified against XML, 3228/3228 agree).
Usage: python prepare_cctsdb_test.py <CCTSDB 2021 zip dir> <output dir>
"""
import sys, os, glob, json, zipfile, hashlib, collections
import xml.etree.ElementTree as ET
from PIL import Image
SRC, OUT = sys.argv[1], sys.argv[2]
RAW = os.path.join(OUT, "raw")
ZIPS = {"test_img.zip": "test_img", "test_labels.zip": "test_labels",
        "Classification based on weather and environment.zip": "weather",
        "Classification based on size of traffic signs.zip": "size"}
def extract(zname, sub):
    with zipfile.ZipFile(os.path.join(SRC, zname)) as z:
        for info in z.infolist():
            name = info.filename
            if not (info.flag_bits & 0x800):
                try: name = name.encode("cp437").decode("gbk")   # Chinese folder names stored in GBK
                except Exception: pass
            out = os.path.join(RAW, sub, name)
            if name.endswith("/"): os.makedirs(out, exist_ok=True); continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(info) as f, open(out, "wb") as g: g.write(f.read())
zip_sha = {}
for zn, sub in ZIPS.items():
    zip_sha[zn] = hashlib.sha256(open(os.path.join(SRC, zn), "rb").read()).hexdigest()
    if not os.path.isdir(os.path.join(RAW, sub)): extract(zn, sub)
CLS = {0: "mandatory", 1: "prohibitory", 2: "warning"}
cats = [{"id": i + 1, "name": n, "supercategory": "traffic_sign"} for i, n in CLS.items()]
weather = {os.path.basename(f)[:-4]: f.split(os.sep)[-2] for f in glob.glob(os.path.join(RAW, "weather", "*", "*", "*.xml"))}
sizecls = {os.path.basename(f)[:-4]: f.split(os.sep)[-2] for f in glob.glob(os.path.join(RAW, "size", "*", "*", "*.xml"))}
images, anns, aid, check = [], [], 1, collections.Counter()
for p in sorted(glob.glob(os.path.join(RAW, "test_img", "*.jpg"))):
    stem = os.path.basename(p)[:-4]; iid = int(stem); W, H = Image.open(p).size   # image id = int(file stem), as for TT100K
    images.append({"id": iid, "file_name": os.path.basename(p), "width": W, "height": H,
                   "weather": weather.get(stem), "size_class": sizecls.get(stem)})
    lab = [l.split() for l in open(os.path.join(RAW, "test_labels", stem + ".txt")).read().splitlines() if l.strip()]
    xml = [f for f in glob.glob(os.path.join(RAW, "weather", "*", "*", stem + ".xml"))][0]
    xo = [(o.find("name").text.strip(), [float(o.find("bndbox").find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]) for o in ET.parse(xml).getroot().iter("object")]
    check["count_match"] += len(xo) == len(lab)
    for c, x, y, w, h in lab:
        c = int(c); w, h = float(w) * W, float(h) * H; x0, y0 = float(x) * W - w / 2, float(y) * H - h / 2
        nm = min(xo, key=lambda o: abs(o[1][0] - x0) + abs(o[1][1] - y0))
        check["class_match"] += nm[0] == CLS[c]; check["boxes"] += 1
        anns.append({"id": aid, "image_id": iid, "category_id": c + 1, "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                     "area": round(w * h, 2), "iscrowd": 0}); aid += 1
assert check["count_match"] == len(images) and check["class_match"] == check["boxes"], check
os.makedirs(os.path.join(OUT, "coco"), exist_ok=True)
def dump(name, keep):
    ids = {im["id"] for im in images if keep(im)}
    js = {"images": [im for im in images if im["id"] in ids], "annotations": [a for a in anns if a["image_id"] in ids], "categories": cats}
    json.dump(js, open(os.path.join(OUT, "coco", f"gt_cctsdb_test_{name}.json"), "w"))
    return len(js["images"]), len(js["annotations"])
summary = {"all": dump("all", lambda im: True)}
for w in sorted(set(weather.values())): summary[w] = dump(w, lambda im, w=w: im["weather"] == w)
json.dump({"zip_sha256": zip_sha, "check": dict(check), "subsets_images_boxes": summary,
           "image_sizes": collections.Counter(f"{im['width']}x{im['height']}" for im in images).most_common()},
          open(os.path.join(OUT, "prepare_summary.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary)); print(dict(check))
