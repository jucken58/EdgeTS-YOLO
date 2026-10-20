"""Summarise bench_fp16_formal.sh output: per model the 3 trtexec repetitions (mean/median/p99 latency, GPU compute, throughput)
and tegrastats (mean VDD_IN power, mean VDD_CPU_GPU_CV power, max GPU temperature) over each repetition.
Prints CSV: run_id, rep, latency_mean_ms, latency_median_ms, latency_p99_ms, gpu_compute_mean_ms, throughput_qps,
vdd_in_mean_W, vdd_cpu_gpu_cv_mean_W, gpu_temp_max_C, engine_MB, build_s(n/a here)."""
import sys, os, re, glob, statistics as st
out = sys.argv[1]
def lat(txt, key):
    m = re.search(key + r": min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, percentile\(99%\) = ([\d.]+) ms", txt)
    return [float(x) for x in m.groups()] if m else None
print("run_id,rep,latency_mean_ms,latency_median_ms,latency_p99_ms,gpu_compute_mean_ms,gpu_compute_median_ms,throughput_qps,vdd_in_mean_W,vdd_cpu_gpu_cv_mean_W,gpu_temp_max_C,n_tegrastats,engine_MB")
for d in sorted(glob.glob(os.path.join(out, "*", ""))):
    rid = os.path.basename(os.path.dirname(d)); eng = os.path.join(d, "fp16.engine")
    emb = round(os.path.getsize(eng) / 1e6, 2) if os.path.exists(eng) else ""
    for k in (1, 2, 3):
        f = os.path.join(d, f"run{k}.log")
        if not os.path.exists(f): continue
        t = open(f, errors="ignore").read(); L = lat(t, "Latency"); G = lat(t, "GPU Compute Time")
        q = re.search(r"Throughput: ([\d.]+) qps", t)
        vin, vcg, tg = [], [], []
        tf = os.path.join(d, f"tegrastats_run{k}.log")
        if os.path.exists(tf):
            for line in open(tf, errors="ignore"):
                a = re.search(r"VDD_IN (\d+)mW", line); b = re.search(r"VDD_CPU_GPU_CV (\d+)mW", line); g = re.search(r"GPU@([\d.]+)C", line)
                if a: vin.append(int(a.group(1)))
                if b: vcg.append(int(b.group(1)))
                if g: tg.append(float(g.group(1)))
        f2 = lambda v: round(v, 3)
        print(",".join(str(x) for x in [rid, k, f2(L[2]) if L else "", f2(L[3]) if L else "", f2(L[6]) if L else "", f2(G[2]) if G else "", f2(G[3]) if G else "",
              q.group(1) if q else "", round(st.mean(vin) / 1000, 3) if vin else "", round(st.mean(vcg) / 1000, 3) if vcg else "", max(tg) if tg else "", len(vin), emb]))
