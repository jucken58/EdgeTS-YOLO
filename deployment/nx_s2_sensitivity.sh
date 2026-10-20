#!/bin/bash
# S2: INT8 sensitivity analysis of the FINAL model on the VALIDATION split only (stopping rule pre-specified).
# Step 0: calibration algorithm MinMax vs Entropy2 for INT8-full,
# chosen on VAL AP; the chosen algorithm/cache is then used for every S2 build. Entropy2 INT8-full is evaluated on test only
# after the choice has been written to summary.txt (for Table 8 completeness).
# Start: INT8-full engine (MinMax cache from stage 2). Step 1: for each of 8 module groups build INT8 with that group forced to
# FP16, val AP -> gain over INT8-full. Step 2 (greedy): add groups in order of gain until val AP >= FP16 val AP - 0.5;
# that configuration is the "INT8-mixed" engine, then evaluated ONCE on test + timed (3x trtexec + tegrastats).
set -u; source $HOME/edgets_env.sh
BR=$1; DR=$2; SD=$BR/scripts; R=20261010_yolo11n-p2p4-nwd0.5_s0; W=$BR/precision/$R; OUT=$BR/sensitivity; mkdir -p $OUT
trap 'sudo -n tegrastats --stop >/dev/null 2>&1' EXIT
sudo -n nvpmodel -m 8 >/dev/null 2>&1; sudo -n jetson_clocks
declare -A G=( [G1_stem_0-1]='^/model\.(0|1)/' [G2_backbone_2-4]='^/model\.(2|3|4)/' [G3_backbone_5-8]='^/model\.(5|6|7|8)/' [G4_sppf_c2psa_9-10]='^/model\.(9|10)/'
               [G5_neck_topdown_11-16]='^/model\.(11|12|13|14|15|16)/' [G6_neck_p2_17-19]='^/model\.(17|18|19)/' [G7_neck_bottomup_20-25]='^/model\.(20|21|22|23|24|25)/' [G8_detect_26]='^/model\.26/' )
evalv() { # $1 = engine prefix, $2 = tag
  python3 $SD/val_engine.py $1_ul.engine $DR/TT100K45_nx.yaml val $OUT/eval $2_val > $OUT/val_$2.log 2>&1
  python3 $SD/coco_eval.py --gt $DR/gt_val.json --pred $OUT/eval/$2_val/predictions.json --out $OUT/coco_$2_val.json > /dev/null 2>&1
  python3 -c "import json;print(json.load(open('$OUT/coco_$2_val.json'))['AP'])" ; }
FP16=$(python3 -c "import json;print(json.load(open('$W/coco_fp16_val.json'))['AP'])"); FULL=$(python3 -c "import json;print(json.load(open('$W/coco_int8full_val.json'))['AP'])")
echo "BASE fp16_val_AP=$FP16 int8full_minmax_val_AP=$FULL" | tee $OUT/summary.txt
python3 $SD/build_trt.py --onnx $W/model.onnx --out $W/int8full_ent --precision int8 --calib-algo entropy2 --calib-list $DR/calib_abs.txt --cache $W/calib_entropy2.cache > $W/build_int8full_ent.log 2>&1
ENT=$(evalv $W/int8full_ent int8full_ent); cp $OUT/coco_int8full_ent_val.json $W/coco_int8full_ent_val.json
if python3 -c "import sys; sys.exit(0 if $ENT > $FULL else 1)"; then ALGO=entropy2; CACHE=$W/calib_entropy2.cache; FULL=$ENT; else ALGO=minmax; CACHE=$W/calib_minmax.cache; fi
echo "CALIB int8full_entropy2_val_AP=$ENT -> chosen=$ALGO (base for S2: $FULL)" | tee -a $OUT/summary.txt
python3 $SD/val_engine.py $W/int8full_ent_ul.engine $DR/TT100K45_nx.yaml test $W/eval int8full_ent_test > $W/val_int8full_ent_test.log 2>&1
python3 $SD/coco_eval.py --gt $DR/gt_test.json --pred $W/eval/int8full_ent_test/predictions.json --out $W/coco_int8full_ent_test.json > /dev/null 2>&1; echo "TEST int8full_ent $(tr -d '\n ' < $W/coco_int8full_ent_test.json)"
for k in 1 2 3; do sleep 60; sudo -n tegrastats --interval 100 --logfile $W/tegrastats_int8full_ent_run$k.log --start
  trtexec --loadEngine=$W/int8full_ent.engine --warmUp=3000 --iterations=1000 --avgRuns=1000 --useSpinWait > $W/run_int8full_ent_$k.log 2>&1; sudo -n tegrastats --stop; sleep 1; done
for g in "${!G[@]}"; do
  python3 $SD/build_trt.py --onnx $W/model.onnx --out $OUT/$g --precision int8 --fp16-groups "${G[$g]}" --calib-algo $ALGO --calib-list $DR/calib_abs.txt --cache $CACHE > $OUT/build_$g.log 2>&1
  ap=$(evalv $OUT/$g $g); echo "SINGLE $g ap=$ap gain=$(python3 -c "print(round($ap-$FULL,2))")" | tee -a $OUT/summary.txt
done
ORDER=$(grep '^SINGLE' $OUT/summary.txt | sed -E 's/SINGLE ([^ ]+) ap=[^ ]+ gain=(.*)/\2 \1/' | sort -gr | awk '{print $2}')
SEL=""; PATS=""
for g in $ORDER; do
  SEL="$SEL $g"; PATS="${PATS:+$PATS,}${G[$g]}"; tag=greedy_$(echo $SEL | wc -w)
  python3 $SD/build_trt.py --onnx $W/model.onnx --out $OUT/$tag --precision int8 --fp16-groups "$PATS" --calib-algo $ALGO --calib-list $DR/calib_abs.txt --cache $CACHE > $OUT/build_$tag.log 2>&1
  ap=$(evalv $OUT/$tag $tag); echo "GREEDY$SEL ap=$ap" | tee -a $OUT/summary.txt
  if python3 -c "import sys; sys.exit(0 if $ap >= $FP16 - 0.5 else 1)"; then echo "SELECTED$SEL ap=$ap (fp16 $FP16)" | tee -a $OUT/summary.txt; break; fi
done
cp $OUT/$tag.engine $W/int8mixed.engine; cp $OUT/${tag}_ul.engine $W/int8mixed_ul.engine; cp $OUT/${tag}_build.json $W/int8mixed_build.json; echo "$SEL" > $W/int8mixed_groups.txt
python3 $SD/val_engine.py $W/int8mixed_ul.engine $DR/TT100K45_nx.yaml test $W/eval int8mixed_test > $W/val_int8mixed_test.log 2>&1
python3 $SD/coco_eval.py --gt $DR/gt_test.json --pred $W/eval/int8mixed_test/predictions.json --out $W/coco_int8mixed_test.json > /dev/null 2>&1; echo "TEST int8mixed $(tr -d '\n ' < $W/coco_int8mixed_test.json)"
cp $OUT/coco_${tag}_val.json $W/coco_int8mixed_val.json
for k in 1 2 3; do sleep 60; sudo -n tegrastats --interval 100 --logfile $W/tegrastats_int8mixed_run$k.log --start
  trtexec --loadEngine=$W/int8mixed.engine --warmUp=3000 --iterations=1000 --avgRuns=1000 --useSpinWait > $W/run_int8mixed_$k.log 2>&1; echo "RUN int8mixed #$k rc=$?"; sudo -n tegrastats --stop; sleep 1; done
