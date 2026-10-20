#!/bin/bash
# Jetson Xavier NX FP16 engine latency and power benchmark (Tables 4, 6 and 11).
# Protocol (identical for every model):
#   - nvpmodel 8 (MODE_20W_6CORE), jetson_clocks on, desktop (gdm3) stopped; JetPack 5.1.6 / L4T R35.6.5 / TensorRT 8.5.2.2
#   - ONNX export on the NX: Ultralytics 8.4.166 (patched tree), opset 17, simplify, static 1x3xNxN (N = IMGSZ, default 640)
#   - engine: trtexec --fp16 --memPoolSize=workspace:2048 (fixed workspace for all models)
#   - timing: trtexec --loadEngine --warmUp=3000 --iterations=1000 --avgRuns=1000 --useSpinWait, repeated 3x,
#     60 s idle before each repetition; tegrastats logged at 100 ms during every repetition (power, temperature)
#   - engine latency includes H2D + GPU compute + D2H (trtexec "Latency"); pre/post-processing are measured separately (Table 9)
# Usage: [IMGSZ=896] bash bench_fp16_formal.sh <models_dir with <run_id>.pt> <out_dir> <run_id> [<run_id> ...]
set -u
source $HOME/edgets_env.sh
OPSET=${OPSET:-17}   # override only where TensorRT 8.5 cannot parse opset 17 (RT-DETR-L LayerNormalization -> OPSET=16)
MD=$1; OUT=$2; shift 2; mkdir -p $OUT
trap 'sudo -n tegrastats --stop >/dev/null 2>&1' EXIT
sudo -n nvpmodel -m 8 >/dev/null 2>&1; sudo -n jetson_clocks
{ date '+%F %T'; sudo -n nvpmodel -q; systemctl is-active gdm3; cat /etc/nv_tegra_release | head -1; free -m | sed -n 2p; } > $OUT/env_at_start.txt 2>&1
for R in "$@"; do
  W=$OUT/$R; mkdir -p $W; cp $MD/$R.pt $W/model.pt; sha256sum $W/model.pt > $W/model.sha256
  ( cd $W && yolo export model=model.pt format=onnx imgsz=${IMGSZ:-640} opset=$OPSET simplify=True dynamic=False batch=1 > export.log 2>&1 ); echo "EXPORT $R opset=$OPSET rc=$?"
  sha256sum $W/model.onnx > $W/onnx.sha256 2>/dev/null
  t0=$(date +%s); trtexec --onnx=$W/model.onnx --saveEngine=$W/fp16.engine --fp16 --memPoolSize=workspace:2048 > $W/build_fp16.log 2>&1; rc=$?
  echo "BUILD $R rc=$rc build_s=$(( $(date +%s)-t0 )) engine_bytes=$(stat -c %s $W/fp16.engine 2>/dev/null)"
  [ $rc -ne 0 ] && continue
  for k in 1 2 3; do
    sleep 60
    sudo -n tegrastats --interval 100 --logfile $W/tegrastats_run$k.log --start
    trtexec --loadEngine=$W/fp16.engine --warmUp=3000 --iterations=1000 --avgRuns=1000 --useSpinWait > $W/run$k.log 2>&1; echo "RUN $R #$k rc=$?"
    sudo -n tegrastats --stop; sleep 1
  done
done
date '+%F %T' > $OUT/end_time.txt
python3 $(dirname $0)/parse_nx_bench.py $OUT > $OUT/summary_fp16.csv; cat $OUT/summary_fp16.csv
