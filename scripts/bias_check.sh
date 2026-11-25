#!/usr/bin/env bash
# Bias check: price the same option with many seeds on every engine and test
# whether each engine's average deviation from the closed form is zero.
#
# With a correct engine, z = (price - reference) / std_error is ~N(0, 1), so
# over K seeds the mean z is ~N(0, 1/K) and the spread of z is ~1.
#
#   scripts/bias_check.sh                 16 seeds, 1e7 paths, 252 steps, log and Euler
#   SEEDS=32 PATHS=1e7 scripts/bias_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=${BIN:-./bin/mc_pricer}
OUT=${OUT:-results}
SEEDS=${SEEDS:-16}
PATHS=${PATHS:-1e7}
STEPS=${STEPS:-252}
ENGINES=${ENGINES:-cpu,gpu-pregen,gpu-shuffle,gpu-xorwow,gpu-double}
mkdir -p "$OUT"

f="$OUT/bias.csv"
echo "seed,$("$BIN" --csv-header)" > "$f"
for scheme in log euler; do
  for s in $(seq 1 "$SEEDS"); do
    "$BIN" --engines "$ENGINES" --paths "$PATHS" --steps "$STEPS" --scheme "$scheme" \
      --seed "$s" --csv --repeat 1 | sed "s/^/$s,/" >> "$f"
  done
done
python3 scripts/bias_summary.py "$f" | tee "$OUT/bias_summary.txt"
