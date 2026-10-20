"""Prepend the Ultralytics metadata header (taken from the ONNX file) to a raw trtexec engine so that `yolo val` can load it.
The engine bytes are unchanged (the header is 4-byte length + JSON). Usage: python3 wrap_engine.py model.onnx raw.engine out_ul.engine"""
import ast, json, sys, hashlib, onnx
onx, raw, out = sys.argv[1:4]
meta = {p.key: p.value for p in onnx.load(onx, load_external_data=False).metadata_props}
for k, v in list(meta.items()):
    try: meta[k] = ast.literal_eval(v)
    except Exception: pass
eng = open(raw, "rb").read(); m = json.dumps(meta)
with open(out, "wb") as f:
    f.write(len(m).to_bytes(4, byteorder="little", signed=True)); f.write(m.encode()); f.write(eng)
print(out, "engine_sha256", hashlib.sha256(eng).hexdigest())
