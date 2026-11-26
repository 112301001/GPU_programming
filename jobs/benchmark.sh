#!/bin/bash
#SBATCH --job-name=mc_benchmark
#SBATCH --partition=gpu01
#SBATCH --output=output/%j.out
#SBATCH --error=error/%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#
# Full benchmark: tests, main comparison, all experiments, plots.
# Results go to results/<GPU name>/ (e.g. results/A30/), so runs on different GPUs don't mix.
#
# Partitions on this cluster: gpu01 = node001 (A30), gpu04 = node004 (L40).
# To run on the L40 instead:  sbatch --partition=gpu04 jobs/benchmark.sh
#
# Build on master first:
#   module load cuda/11.3
#   make ARCH=sm_80 MARCH= CC=/usr/bin/gcc NVCC="nvcc -ccbin /usr/bin/gcc"
# Submit from ~/GPU_programming:
#   mkdir -p output error
#   sbatch jobs/benchmark.sh
# For a shorter run: sbatch --export=ALL,QUICK=1 jobs/benchmark.sh

module purge
module load gnu12 cuda/11.3                     # gnu12 provides the OpenMP runtime (libgomp)

cd ~/GPU_programming
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | awk '{print $NF}')
RES=results/$GPU
mkdir -p "$RES"
hostname
echo "GPU: $GPU -> $RES"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm --format=csv

# log SM clock / power / temperature every 500 ms during the whole job (both GPUs of the node)
nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu \
  --format=csv -lms 500 > "$RES/gpu_clocks.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

./bin/test_pricer > "$RES/tests.txt"
tail -1 "$RES/tests.txt"
./bin/mc_pricer --paths 1e7 --engines all --kernel-info > "$RES/main_compare.txt"
./bin/mc_pricer --paths 1e8 --engines cpu,gpu-pregen,gpu-shuffle > "$RES/main_compare_1e8.txt"

OUT="$RES" scripts/benchmark.sh

module load anaconda3                           # matplotlib for the plots
python3 scripts/plot_results.py "$RES"
echo "benchmark job done"
