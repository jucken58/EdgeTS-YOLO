# Jetson measurement protocol

**Device state.** nvpmodel mode 8 (20 W, 6 cores) with `jetson_clocks`; desktop (gdm3) stopped. Each session starts with an idle check (CPU at least 90% idle, idle VDD_IN about 4.8 W).

**Engines.** ONNX opset 17 (RT-DETR-L: opset 16, because TensorRT 8.5 has no LayerNormalization importer), static input 1 × 3 × N × N with N the training input size. Engines are built on the device with a 2 GiB workspace (`deployment/build_trt.py`, `deployment/bench_fp16_formal.sh`).
- INT8 is calibrated on the 500 training images in `data/splits/calib_train500_seed0.txt` with MinMax or entropy (`IInt8EntropyCalibrator2`) calibration.
- INT8-UL: the Ultralytics 8.4.166 INT8 scheme (head sigmoid in FP32, output convolutions and DFL in FP16).
- INT8-mixed: module groups forced to FP16 with `OBEY_PRECISION_CONSTRAINTS` (`--fp16-groups`).

**Engine latency.** `trtexec --warmUp=3000 --iterations=1000 --avgRuns=1000 --useSpinWait`; the reported value is trtexec "Latency" (H2D copy + execution + D2H copy of a single request). Three runs per configuration, each after 60 s idle; the mean of the three runs is reported. Throughput is reported separately, because trtexec overlaps the copies of one request with the execution of the next.

**PyTorch latency.** `deployment/torch_latency.py`, same scope (H2D + forward + D2H), 100 warm-up and 1,000 timed iterations.

**Power and energy.** `tegrastats --interval 100` runs during every measurement. Energy = mean VDD_IN power / throughput, i.e. the energy per completed inference or video frame in the same measurement window.

**End-to-end video.** `deployment/benchmark_video.py --imgsz N` on two 1920 × 1080, 29.97 fps dashcam videos; the first 100 frames are discarded and the remaining frames are timed. Stages: decode (OpenCV, CPU), pre-process, TensorRT inference, post-process (conf 0.25, NMS IoU 0.7). FPS = timed frames / elapsed wall-clock time.

**INT8 sensitivity analysis.** `deployment/nx_s2_sensitivity.sh`, on the validation set only: (0) choose MinMax or entropy calibration for INT8-full; (1) for each of 8 module groups, build INT8 with only that group in FP16; (2) add groups greedily by decreasing gain until validation mAP@0.5:0.95 is within 0.5 points of FP16. Only the selected engine is evaluated on the test set.

**Power modes.** nvpmodel modes 4 (10 W), 2 (15 W) and 8 (20 W), with `jetson_clocks` after each switch and 120 s idle.
