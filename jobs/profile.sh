#!/bin/bash
#SBATCH --job-name=mc_profile
#SBATCH --partition=gpu01
#SBATCH --output=output/%j.out
#SBATCH --error=error/%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#
# Profiling job: Nsight Systems timeline (+ Nsight Compute if counters are allowed).
#
# Build on master first:
#   module load cuda/11.3
#   make ARCH=sm_80 MARCH= CC=/usr/bin/gcc NVCC="nvcc -ccbin /usr/bin/gcc"
# Submit from ~/GPU_programming:
#   mkdir -p output error
#   sbatch jobs/profile.sh

module purge
module load gnu12 cuda/11.3                     # gnu12 provides the OpenMP runtime (libgomp)
export PATH=/opt/nvidia/nsight-systems/2023.1.2/target-linux-x64:$PATH   # nsys on the GPU nodes
export PATH=$PATH:/usr/local/cuda-12/bin        # ncu (fails without counter permission)

cd ~/GPU_programming
hostname
nvidia-smi --query-gpu=name,compute_cap --format=csv
which nsys ncu

./bin/mc_pricer --paths 1e6 --engines gpu-shuffle --kernel-info --repeat 1 > results_kernel_info.txt
scripts/profile.sh
echo "profile job done"
