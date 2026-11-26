#!/bin/bash
#SBATCH --job-name=mc_bias
#SBATCH --partition=gpu04
#SBATCH --output=output/%j.out
#SBATCH --error=error/%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#
# Bias check of every engine over 32 seeds (scripts/bias_check.sh): validation for the report.
# Results: results/<GPU name>/bias.csv and bias_summary.txt
#   sbatch jobs/bias_check.sh                       (L40, partition gpu04)
#   sbatch --partition=gpu01 jobs/bias_check.sh     (A30)

module purge
module load gnu12 cuda/11.3

cd ~/GPU_programming
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | awk '{print $NF}')
hostname
echo "GPU: $GPU"
SEEDS=32 OUT="results/$GPU" scripts/bias_check.sh
echo "bias job done"
