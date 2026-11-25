#!/usr/bin/env bash
# Profiling with NVIDIA Nsight Systems (timeline) and Nsight Compute (per kernel).
#
#   scripts/profile.sh          N = 1e7 paths, 252 steps
#   PATHS=1e8 scripts/profile.sh
#
# Outputs in results/profile/:
#   nsys_mc.nsys-rep      open in the Nsight Systems GUI (NVTX range per engine)
#   nsys_summary.txt      kernel / memcpy / NVTX time tables
#   ncu_<engine>.ncu-rep  open in the Nsight Compute GUI (full set incl. roofline chart)
#   ncu_<engine>.txt      text: SOL, roofline, occupancy, launch stats, warp stalls
#   roofline.csv          FLOP and DRAM byte counters for scripts/roofline.py
#
# If ncu reports ERR_NVGPUCTRPERM you need permission to read GPU performance
# counters: run as root / with sudo, or ask the admin to set
# NVreg_RestrictProfilingToAdminUsers=0.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=${BIN:-./bin/mc_pricer}
PATHS=${PATHS:-1e7}
STEPS=${STEPS:-252}
OUT=${OUT:-results/profile}
mkdir -p "$OUT"
[[ -x "$BIN" ]] || { echo "build first: make" >&2; exit 1; }

ARGS=(--paths "$PATHS" --steps "$STEPS" --repeat 1)

if command -v nsys >/dev/null; then
  echo "== Nsight Systems timeline"
  nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true -o "$OUT/nsys_mc" \
    "$BIN" "${ARGS[@]}" --engines gpu-pregen,gpu-atomic,gpu-shared,gpu-shuffle,gpu-xorwow,gpu-double
  nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true -o "$OUT/nsys_hostrng" \
    "$BIN" --paths 1e6 --steps "$STEPS" --repeat 1 --engines gpu-hostrng,gpu-pregen,gpu-shuffle
  { nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
      "$OUT/nsys_mc.nsys-rep" > "$OUT/nsys_summary.txt" &&
    nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
      "$OUT/nsys_hostrng.nsys-rep" >> "$OUT/nsys_summary.txt" &&
    echo "   -> $OUT/nsys_summary.txt"; } ||
    echo "   nsys stats failed; open the .nsys-rep files in the Nsight Systems GUI instead"
else
  echo "nsys not found, skipping Nsight Systems"
fi

if command -v ncu >/dev/null; then
  echo "== Nsight Compute, one kernel launch per engine"
  METRICS=gpu__time_duration.sum,dram__bytes.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__inst_executed_pipe_xu.sum,sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed
  : > "$OUT/roofline.csv"
  for eng in gpu-shuffle gpu-shared gpu-atomic gpu-xorwow gpu-double gpu-pregen; do
    kernel="mc_${eng#gpu-}_kernel"
    echo "-- $eng ($kernel)"
    # ncu prints its errors (==ERROR==) on stdout, so keep a log and show its ncu lines.
    log="$OUT/ncu_$eng.log"
    if ! ncu --set full -k "$kernel" --launch-count 1 -f -o "$OUT/ncu_$eng" \
         "$BIN" "${ARGS[@]}" --no-warmup --engines "$eng" > "$log" 2>&1 \
       || grep -q "==ERROR==" "$log" || [[ ! -s "$OUT/ncu_$eng.ncu-rep" ]]; then
      echo "   ncu failed for $eng:"
      grep -E "==(ERROR|WARNING)==" "$log" | head -5 | sed 's/^/     /'
      if grep -q "ERR_NVGPUCTRPERM" "$log"; then
        echo "   -> no permission to read GPU performance counters on this cluster."
        echo "      Ask the admin (or run with sudo). Skipping Nsight Compute."
        break
      fi
      continue
    fi
    ncu --import "$OUT/ncu_$eng.ncu-rep" --page details \
      --section SpeedOfLight --section SpeedOfLight_RooflineChart --section Occupancy \
      --section LaunchStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
      --section WarpStateStats > "$OUT/ncu_$eng.txt" 2>&1 || true
    ncu --csv --metrics "$METRICS" -k "$kernel" --launch-count 1 \
      "$BIN" "${ARGS[@]}" --no-warmup --engines "$eng" 2>&1 | grep -E '^"' >> "$OUT/roofline.csv" || true
  done
  echo "   -> $OUT/ncu_*.txt, $OUT/roofline.csv  (plot: python3 scripts/roofline.py)"
else
  echo "ncu not found, skipping Nsight Compute"
fi
