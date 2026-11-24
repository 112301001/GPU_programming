#!/usr/bin/env bash
# One-shot: build, test, benchmark, plot, profile. Run on the GPU machine.
#   scripts/run_all.sh            (ARCH=sm_75 scripts/run_all.sh to pick the GPU arch)
#   QUICK=1 scripts/run_all.sh    smaller benchmark sizes
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results
make ARCH="${ARCH:-native}"
./bin/test_pricer | tee results/tests.txt
./bin/mc_pricer --engines all --paths 1e6 --steps 252 | tee results/quick_compare.txt
./bin/mc_pricer --paths 1e7 --steps 252 | tee results/main_compare.txt
scripts/benchmark.sh
python3 scripts/plot_results.py || echo "plotting skipped (pip install matplotlib)"
scripts/profile.sh || echo "profiling failed (see message above)"
[[ -s results/profile/roofline.csv ]] && { python3 scripts/roofline.py || true; }
echo "all results in results/"
