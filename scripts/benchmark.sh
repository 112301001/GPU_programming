#!/usr/bin/env bash
# Runs the experiments of the project plan and writes CSV files to results/.
#
#   scripts/benchmark.sh            full run (tens of minutes, dominated by the CPU baselines)
#   QUICK=1 scripts/benchmark.sh    smaller sizes, a few minutes
#
# Env: BIN (default ./bin/mc_pricer), OUT (default results), REPEAT (default 3)
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=${BIN:-./bin/mc_pricer}
OUT=${OUT:-results}
REPEAT=${REPEAT:-3}
QUICK=${QUICK:-0}
mkdir -p "$OUT"
[[ -x "$BIN" ]] || { echo "build first: make" >&2; exit 1; }

if [[ "$QUICK" == 1 ]]; then
  CONV_PATHS="1e3 1e4 1e5 1e6 1e7"
  SCALE_PATHS="1e4 1e5 1e6 1e7"
  N_ABL=4e6
  CPU1_MAX=1000000
  CPU_MAX=10000000
else
  CONV_PATHS="1e3 1e4 1e5 1e6 1e7 1e8"
  SCALE_PATHS="1e4 1e5 1e6 1e7 1e8"
  N_ABL=1e7
  CPU1_MAX=10000000
  CPU_MAX=100000000
fi
HOSTRNG_MAX=1000000

run() { "$BIN" --csv --repeat "$REPEAT" "$@"; }
header() { "$BIN" --csv-header > "$1"; }
as_int() { python3 -c "print(int(float('$1')))"; }

"$BIN" --paths 1e6 --engines gpu-shuffle --kernel-info --repeat 1 | tee "$OUT/device.txt"

echo "== 1. convergence: price and SE vs N, antithetic on/off (European, exact log step)"
f="$OUT/convergence.csv"; header "$f"
for n in $CONV_PATHS; do
  run --engines gpu-shuffle --paths "$n" --steps 1 --scheme log --antithetic    >> "$f"
  run --engines gpu-shuffle --paths "$n" --steps 1 --scheme log --no-antithetic >> "$f"
done

echo "== 2. scaling: time vs N, GPU vs single-core vs OpenMP CPU (European, 252 Euler steps)"
f="$OUT/scaling.csv"; header "$f"
for n in $SCALE_PATHS; do
  ni=$(as_int "$n")
  engines="gpu-pregen,gpu-shuffle"
  (( ni <= CPU_MAX )) && engines="cpu,$engines"
  (( ni <= CPU1_MAX )) && engines="cpu1,$engines"
  (( ni <= HOSTRNG_MAX )) && engines="$engines,gpu-hostrng"
  run --engines "$engines" --paths "$n" --steps 252 >> "$f"
done

echo "== 3. ablation: every GPU engine at N=$N_ABL (European, 252 Euler steps)"
f="$OUT/ablation.csv"; header "$f"
run --engines cpu,gpu-pregen,gpu-atomic,gpu-shared,gpu-shuffle,gpu-xorwow,gpu-double --paths "$N_ABL" --steps 252 >> "$f"
run --engines gpu-hostrng --paths 1e6 --steps 252 >> "$f"

echo "== 4. antithetic ablation: SE and time with and without antithetic variates"
f="$OUT/antithetic.csv"; header "$f"
for style in european asian; do
  run --engines cpu,gpu-shuffle --style "$style" --paths "$N_ABL" --steps 252 --antithetic    >> "$f"
  run --engines cpu,gpu-shuffle --style "$style" --paths "$N_ABL" --steps 252 --no-antithetic >> "$f"
done

echo "== 5. launch configuration: block size x blocks per SM (gpu-shuffle)"
f="$OUT/launch.csv"; header "$f"
for bs in 32 64 128 256; do
  for bps in 1 2 4 8 16 32; do
    run --engines gpu-shuffle --paths "$N_ABL" --steps 252 --block "$bs" --blocks-per-sm "$bps" >> "$f"
  done
done

echo "== 6. option styles and schemes"
f="$OUT/styles.csv"; header "$f"
for style in european asian geo-asian; do
  for scheme in euler log; do
    run --engines cpu,gpu-shuffle --style "$style" --scheme "$scheme" --paths "$N_ABL" --steps 252 >> "$f"
  done
done

echo "== 7. time steps per path"
f="$OUT/steps.csv"; header "$f"
for m in 1 12 52 252 1000; do
  run --engines gpu-shuffle,gpu-pregen --style asian --paths "$N_ABL" --steps "$m" >> "$f"
done

echo "done -> $OUT/*.csv   (plot with: python3 scripts/plot_results.py)"
