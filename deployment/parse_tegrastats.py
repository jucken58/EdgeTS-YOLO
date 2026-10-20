"""Average module power (VDD_IN) and peak GPU temperature from a tegrastats log.
Handles 'VDD_IN 5123mW/5010mW' (JetPack 5) and 'POM_5V_IN 5123/5010' (older) formats.
"""
import re, sys
import numpy as np
txt = open(sys.argv[1]).read().splitlines()
pw, tg = [], []
for l in txt:
    m = re.search(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)(?:mW)?/", l)
    if m: pw.append(int(m.group(1)) / 1000)
    t = re.search(r"GPU@(-?[\d.]+)C", l)
    if t: tg.append(float(t.group(1)))
assert pw, "no VDD_IN readings found - check the rail name in the log"
print(f"samples={len(pw)} avg_power_W={np.mean(pw):.2f} max_power_W={np.max(pw):.2f} peak_GPU_temp_C={max(tg) if tg else float('nan'):.1f}")
