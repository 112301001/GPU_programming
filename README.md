# GPU-Accelerated Monte Carlo Engine for Options Pricing

CS5658 GPU Programming, Group G15: Varun Jha (112301036), Sujit Kumar Choudhary (112301001).
Proposal: [docs/proposal.pdf](docs/proposal.pdf). Implementation note: [docs/PROJECT_NOTE.md](docs/PROJECT_NOTE.md).

A Monte Carlo pricer for European and Asian options written in **CUDA C**. Paths are simulated with
Euler–Maruyama time stepping of geometric Brownian motion entirely in registers. Random numbers come
from cuRAND Philox generated on the device. Payoffs are summed with warp shuffles (`__shfl_down_sync`)
and one `atomicAdd` per block. Antithetic variates reduce the variance. The engine is checked against
Black–Scholes and the closed-form geometric Asian price, and benchmarked against a single-core and an
OpenMP multi-core CPU implementation of the same algorithm.

## Results at a glance

European call, 252 time steps, antithetic variates, end-to-end time at sustained GPU clock
(details and ablations: [docs/PROJECT_NOTE.md](docs/PROJECT_NOTE.md) section 10, raw data and plots
in `results/`):

| | NVIDIA A30 | NVIDIA L40 |
| --- | --- | --- |
| 10⁸ paths, `gpu-shuffle` | 97.9 ms (1.0 G paths/s) | 28.6 ms (3.5 G paths/s) |
| speedup vs OpenMP CPU (64 threads) | 50× | 106× |
| speedup vs one CPU core | 1713× | 4430× |
| correctness | 50/50 tests | 50/50 tests; every engine unbiased over 32 seeds |

## Layout

```text
include/common.h        option/simulation structs, price + standard error from the sums
include/path_model.h    GBM step and payoffs, shared by the CPU and the GPU (__host__ __device__)
src/gpu_mc.cu           CUDA kernels and their host drivers
src/cpu_mc.c            CPU baseline (C + OpenMP, xoshiro256** + polar Box–Muller)
src/black_scholes.c     Black–Scholes and discrete geometric Asian closed forms
src/main.c              command-line driver (table or CSV output)
tests/test_pricer.c     correctness tests
scripts/benchmark.sh    the experiments (CSV files in results/)
scripts/plot_results.py plots from the CSV files (results/plots/)
scripts/profile.sh      Nsight Systems + Nsight Compute runs (results/profile/)
scripts/roofline.py     roofline plot from the Nsight Compute counters
scripts/run_all.sh      build, test, benchmark, plot and profile in one go
scripts/bias_check.sh   price with many seeds and test every engine for bias
jobs/*.sh               SLURM batch jobs for the institute GPU cluster
results/                measured data: A30/, L40/ (sustained clock), *_no_spinup/ (idle-clock runs), profile/
```

## Build and run (on the GPU machine)

Needs the CUDA toolkit (nvcc, cuRAND), gcc with OpenMP, and a GPU with compute capability ≥ 6.0
(double-precision `atomicAdd`).

```bash
make                    # or: make ARCH=sm_75 (2080 Ti), sm_80 (A30/A100), sm_86, sm_89 ...
make test               # correctness tests: closed forms, CPU, every GPU engine
./bin/mc_pricer         # 16.8M paths, 252 steps, European call, default engines
./bin/mc_pricer --help
```

`ARCH=native` needs CUDA ≥ 11.6. With an older toolkit, pass the architecture explicitly.

Examples:

```bash
./bin/mc_pricer --paths 1e8 --engines cpu,gpu-shuffle                 # headline speedup
./bin/mc_pricer --paths 1e7 --engines all                             # every engine side by side
./bin/mc_pricer --style asian --steps 252 --paths 1e7                 # arithmetic Asian call
./bin/mc_pricer --scheme log --steps 1 --paths 1e8 --engines gpu-shuffle   # exact European, 1 step
./bin/mc_pricer --kernel-info --engines gpu-shuffle                   # registers / occupancy per kernel
make regs                                                             # ptxas register/spill report
```

Output columns:

```text
engine   price   std_err   err   err/se   total_ms   kernel_ms   Mpaths/s   vs cpu   vs cpu1
```

`err/se` is the distance from the closed-form price in standard errors (|err/se| < 3 is expected).
`total_ms` covers everything: RNG, simulation, reduction and copies. `kernel_ms` is the cudaEvent time
of the kernel(s) only.

## Engines (ablation study)

`gpu-shuffle` is the design from the proposal. Every other GPU engine changes exactly one decision, so
comparing it with `gpu-shuffle` measures what that decision is worth.

| engine | random numbers | reduction | what it shows |
| --- | --- | --- | --- |
| `cpu1` | xoshiro256**, 1 core | — | single-core baseline |
| `cpu` | xoshiro256**, OpenMP all cores | OpenMP `reduction(+)` | multi-core baseline |
| `gpu-hostrng` | cuRAND host API **on the CPU**, copied over **PCIe** into global memory | shuffle | cost of host-generated random vectors |
| `gpu-pregen` | cuRAND generated on the GPU into a **global-memory array** | shuffle | memory-bound version vs registers |
| `gpu-atomic` | Philox in registers | **one global `atomicAdd` per sample** | atomic serialisation |
| `gpu-shared` | Philox in registers | **shared-memory tree** + `__syncthreads` per level | classic reduction vs shuffles |
| **`gpu-shuffle`** | **Philox in registers, `curand_normal4`** | **warp shuffle → shared[32] → 1 atomic/block** | **proposed engine** |
| `gpu-xorwow` | XORWOW in registers | shuffle | Philox vs XORWOW |
| `gpu-double` | Philox, **double** precision path | shuffle (double) | float vs double |

## Experiments

```bash
scripts/run_all.sh          # everything (QUICK=1 for smaller sizes)
# or step by step:
scripts/benchmark.sh        # -> results/*.csv
python3 scripts/plot_results.py   # -> results/plots/*.png  (pip install matplotlib)
scripts/profile.sh          # -> results/profile/  (nsys + ncu)
python3 scripts/roofline.py --peak-gflops <FP32 GFLOP/s> --peak-gbs <GB/s>
```

| file | experiment |
| --- | --- |
| `convergence.csv` | price and standard error vs N = 10³…10⁸, antithetic on/off; SE should fall as N^-1/2 |
| `scaling.csv` | time and throughput vs N: GPU vs single-core vs OpenMP CPU |
| `ablation.csv` | every engine at one N (the table above) |
| `antithetic.csv` | variance reduction factor of antithetic variates (European and Asian) |
| `launch.csv` | block size × blocks per SM (occupancy and latency hiding) |
| `styles.csv` | European / arithmetic Asian / geometric Asian × Euler / log scheme |
| `steps.csv` | cost vs number of time steps, in-register RNG vs pre-generated RNG |

`ncu` needs permission to read the GPU performance counters. If it fails with `ERR_NVGPUCTRPERM`, run it
with sudo or ask the admin. For debugging, `make debug` builds `bin/mc_pricer_debug` with `-G -g` for
`cuda-gdb` and `compute-sanitizer`.

## Numerical notes

- **Euler on S vs log-Euler.** `--scheme euler` (default, as in the proposal) steps S directly and has an
  O(Δt) bias. `--scheme log` steps ln S, which is exact for GBM. Exact comparisons against the closed
  forms use `--scheme log`. For European options a single log step is enough.
- **Antithetic variates.** One sample is ½[f(Z) + f(−Z)]. `--paths` counts both paths of each pair, so
  runs with and without antithetic variates simulate the same number of paths.
- **Precision.** Paths and per-thread sums are in float. Block totals are added in double with
  `atomicAdd`, which avoids the precision loss of summing 10⁸ payoffs in one float. `gpu-double`
  measures the cost of doing everything in double.
- **Reproducibility.** Results are deterministic for a fixed seed and launch configuration
  (`curand_init(seed, thread id, 0)`). Only the order of block atomics varies, which changes the last
  few bits.

## Running on the institute GPU cluster (SLURM)

The compute nodes have no C headers, so compile on the login node (compiling only), and run
everything through SLURM. Never run the programs on the login node.

```bash
# on master: build with CUDA 11.3 and the system gcc 8 (CUDA 11.3 rejects gcc 12)
module load cuda/11.3
make ARCH=sm_80 MARCH= CC=/usr/bin/gcc NVCC="nvcc -ccbin /usr/bin/gcc"

mkdir -p output error
sbatch jobs/benchmark.sh                     # A30 (partition gpu01) -> results/A30/
sbatch --partition=gpu04 jobs/benchmark.sh   # L40 (partition gpu04) -> results/L40/
sbatch jobs/profile.sh                       # Nsight Systems -> results/profile/
sbatch jobs/bias_check.sh                    # bias check over 32 seeds -> results/L40/bias_summary.txt
squeue -u $USER                              # logs in output/<jobid>.out, error/<jobid>.err
```

`ARCH=sm_80` code also runs on the L40 (sm_89). CUDA 11.3 cannot target sm_89 directly. Nsight Compute
is blocked for users on the cluster (`ERR_NVGPUCTRPERM`), so `profile.sh` skips it.

## CPU-only build

`make cpu` builds `bin/mc_pricer_cpu` and `bin/test_pricer_cpu` without CUDA, for working on the host
code on a machine without a GPU (`--engines cpu,cpu1`).
