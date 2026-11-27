# Implementation note: GPU Monte Carlo engine for options pricing

Group G15 · CS5658 GPU Programming. The note follows the 12-heading format used in the course for the
group-project question. Each heading points to where the code implements it. Measured results are in
section 10.

## 1. Problem, input and output

Price an option as the discounted risk-neutral expectation of its payoff:

V₀ = e^(−rT) E[Payoff] ≈ e^(−rT) (1/N) Σᵢ Payoff(path i)

- **Input:** S₀, K, r, σ, T, call/put, style (European, arithmetic Asian, geometric Asian), M time steps
  and N paths.
- **Output:** price, standard error SE = s/√N, 95 % CI = price ± 1.96 SE, and run time.
- **Why a GPU:** the error falls as 1/√N, so each 10× gain in accuracy costs 100× more paths
  (10⁷–10⁸ paths in practice).

## 2. Baseline and complexity

- Work is O(N·M) normal draws and path updates. Memory is O(1) per path.
- **Correctness baselines:** the Black–Scholes closed form (European) and the closed form for a
  discretely monitored geometric Asian option ([src/black_scholes.c](../src/black_scholes.c)).
- **Speed baselines:** the same algorithm in C on one core (`cpu1`) and on all cores with OpenMP
  (`cpu`, [src/cpu_mc.c](../src/cpu_mc.c)). Both use the same path code as the GPU
  ([include/path_model.h](../include/path_model.h)).

## 3. Sources of parallelism

Paths are independent (embarrassingly parallel). The only communication is the final sum of payoffs
and squared payoffs, which is a **reduction**. The time steps inside one path are sequential, so the
work is parallelised across paths, not across time.

## 4. Thread mapping

- One CUDA thread simulates one sample (an antithetic pair of paths) at a time, in a **grid-stride
  loop** `for (i = tid; i < n; i += gridDim.x * blockDim.x)`.
- The grid is sized to the hardware: blocks_per_sm × #SMs blocks (default 8 × #SMs) of 256 threads.
  It is not sized to N, so one launch handles any N.
- There is no divergence. Every path takes the same number of steps, the payoff is a branch-free
  `fmaxf`, and the style/scheme switches are the same for all threads.

## 5. Kernel pseudocode (`mc_shuffle_kernel`, [src/gpu_mc.cu](../src/gpu_mc.cu))

```text
kernel mc_shuffle_kernel(n, seed, d_acc):
    tid = blockIdx.x*blockDim.x + threadIdx.x ; stride = gridDim.x*blockDim.x
    curand_init(seed, tid, 0, &rng)                 // Philox, own subsequence per thread
    sum = sumsq = 0                                 // registers
    for i = tid; i < n; i += stride:
        x1 = x2 = ln S0 ; acc1 = acc2 = 0           // path and its mirror, registers
        for s = 0; s < M; s += 4:
            z = curand_normal4(&rng)                // 4 normals = 4 steps
            for each of the 4 normals zk:
                x1 += drift + vol*zk ; x2 += drift - vol*zk   (Asian: acc += S)
        v = 0.5 * (payoff(x1, acc1) + payoff(x2, acc2))
        sum += v ; sumsq += v*v
    block_shuffle_add(sum, sumsq, d_acc)            // section 7
host: price = e^(-rT) sum/n ;  SE = e^(-rT) sqrt((sumsq - n mean^2)/(n-1)/n)
```

With `--scheme euler` (as in the proposal), the step is `S *= 1 + r dt + σ√dt z` instead.

## 6. Memory hierarchy

| data | where | why |
| --- | --- | --- |
| option and model constants (`ModelConsts`) | `__constant__` memory, set with `cudaMemcpyToSymbol` | every thread reads the same address, which the constant cache broadcasts |
| RNG state, log-price, Asian running sum, per-thread sums | registers | no memory traffic inside the time loop |
| per-warp partial sums | `__shared__ float w_sum[32], w_sq[32]` | 256 B per block |
| result | 2 doubles in global memory | one `atomicAdd` per block for each |

There is no N×M random array in global memory and nothing is copied over PCIe. `gpu-pregen` and
`gpu-hostrng` show the cost of the alternative.

## 7. Synchronisation and atomics

```text
lane values --__shfl_down_sync 16,8,4,2,1--> lane 0 of each warp
           --> w_sum[warp] (shared)  --__syncthreads()-->  warp 0
           --__shfl_down_sync again--> lane 0 --atomicAdd(double)--> d_acc[0], d_acc[1]
```

- Shuffles move registers between lanes of a converged warp (full mask 0xffffffff) with no shared
  memory and no barrier.
- One `__syncthreads()` separates writing the per-warp partials from reading them.
- **One atomic per block** instead of N. `gpu-atomic` issues one per sample to measure the
  serialisation. `gpu-shared` uses the shared-memory tree from the reduction lecture
  (log₂(blockDim) levels, each followed by `__syncthreads`).
- Double-precision `atomicAdd` needs compute capability ≥ 6.0.

## 8. Random numbers

- The cuRAND **device API** uses `curandStatePhilox4_32_10_t` in registers:
  `curand_init(seed, tid, 0, &st)`.
- Philox is counter-based. Its state is small, `curand_init` with a subsequence is O(1), and
  `curand_normal4` returns 4 normals per call.
- Every thread uses the same seed with a distinct subsequence, which gives independent and reproducible
  streams. If every thread used the same subsequence, all paths would be identical (fake zero variance).
- `gpu-xorwow` compares XORWOW, the cuRAND default, whose `curand_init` with a subsequence is an
  expensive skip-ahead.

## 9. Variance reduction: antithetic variates

The estimator is Y = ½[f(Z) + f(−Z)], which is unbiased. Its variance is Var(Y) = ½ Var(f) (1 + ρ), so
at equal random-number cost the variance is multiplied by (1 + ρ)/2. That is at most ½ when ρ < 0,
which holds for monotone payoffs such as calls, puts and the Asian call. The cost is one extra path
update and payoff per draw, which is cheaper than generating another normal. The measured factor is
in `results/antithetic.csv`.

## 10. Experiments (`scripts/benchmark.sh`, `scripts/profile.sh`)

| experiment | result file | expected shape |
| --- | --- | --- |
| price and SE vs N = 10³…10⁸, compared with Black–Scholes | `convergence.csv` | SE slope −½ on a log-log plot; \|err\| within ~2 SE |
| time and paths/s: GPU vs `cpu1` vs `cpu` | `scaling.csv` | GPU wins once N fills the device |
| ablation: hostrng, pregen, atomic, shared, shuffle, xorwow, double | `ablation.csv` | shuffle fastest; pregen/hostrng memory or PCIe bound |
| antithetic on/off | `antithetic.csv` | variance factor > 2 for calls |
| block size × blocks/SM | `launch.csv` | plateau once enough warps are resident |
| Nsight Systems | `profile/nsys_summary.txt` | kernel time dominates; copies tiny |
| Nsight Compute | `profile/ncu_*.txt` | registers/thread, occupancy, DRAM ≈ 0 %, FMA/XU pipes busy (blocked on the cluster, see below) |
| roofline | `plots/roofline.png` | shuffle kernel far right of the ridge point (compute bound) |

### Results

Measured on the institute cluster (CUDA 11.3, code built for `sm_80`), with a European call
(S₀ = K = 100, r = 5 %, σ = 20 %, T = 1), 252 Euler steps, antithetic variates, 10⁷ paths unless
stated. Times are end to end: RNG, simulation, reduction and copies, median of 3 runs at the GPU's
sustained clock. The CPU baseline is the same algorithm in C with OpenMP on the 64-thread CPU of the
same node. Raw data: `results/A30/`, `results/L40/`; plots: `results/*/plots/`.

| | NVIDIA A30 (56 SMs) | NVIDIA L40 (142 SMs) |
| --- | --- | --- |
| tests (`make test`) | 50/50 | 50/50 |
| `gpu-shuffle`, 10⁸ paths | 97.9 ms, **1022 M paths/s** | 28.6 ms, **3493 M paths/s** |
| speedup vs OpenMP CPU, 10⁸ paths | **50×** | **106×** |
| speedup vs one CPU core, 10⁷ paths | **1713×** | **4430×** |
| registers/thread, theoretical occupancy | 53, 50 % | 53, 67 % |

**Ablation at 10⁷ paths** (time in ms; each engine changes one decision relative to `gpu-shuffle`):

| engine | A30 | L40 | what it shows |
| --- | --- | --- | --- |
| `gpu-shuffle` (proposed) | 9.8 | 2.7 | |
| `gpu-shared` (shared-memory tree) | 9.8 | 2.4 | same: the reduction runs once per thread after ~44 paths, so its cost is negligible |
| `gpu-atomic` (1 global atomic per path) | 22.6 (2.3×) | 16.4 (6.1×) | atomic serialisation; worse on the faster GPU |
| `gpu-xorwow` | 9.6 | 2.9 | Philox and XORWOW perform alike at this grid size (XORWOW's 32 registers give 100 % occupancy) |
| `gpu-double` | 29.5 (3.0×) | 87.4 (33×) | A30 FP64 runs at ½ the FP32 rate, L40 at 1/64 |
| `gpu-pregen` (normals stored in global memory) | 21.0 (2.1×) | 19.2 (7.2×) | memory bound: half the time is spent writing the 5 GB of normals, half reading them (~470 GB/s on both GPUs) |
| `gpu-hostrng` (normals made on the CPU, 10⁶ paths) | 2129 vs 1.0 | 1522 vs 0.28 | ~2000–5000× slower; Nsight Systems shows only 21 ms of PCIe copy (504 MB, ~24 GB/s); the rest is single-threaded host generation |

**Other findings**

- **Accuracy.** Standard error falls as N^-½ from 10³ to 10⁸ paths (`plots/convergence.png`). Over 32
  seeds (L40, `results/L40_no_spinup/bias_summary.txt`), every engine is unbiased with the exact log scheme (all |t| ≤ 1.1). Euler on S is about
  0.001 low (0.01 %), its O(Δt) discretisation bias. This is invisible at 10⁷ paths (SE 0.0033) but is
  1–3 SE at 10⁸ (SE 0.001). Use `--scheme log` for GBM at that accuracy.
- **Antithetic variates** (A30): variance factor 2.00 (European) and 2.08 (Asian). They also run
  1.74× faster at equal path count, because they need half the normals, so the efficiency gain is
  about 3.5×.
- **Scaling with N:** the GPU needs about 10⁶ paths to fill the device. At 10⁴ paths it is only
  about 7× faster than OpenMP (launch overhead). Throughput is flat from 10⁶ paths up.
- **Launch configuration:** throughput rises with resident blocks per SM (A30, 256 threads: 483 →
  1077 M paths/s from 1 to 32 blocks/SM). The default 8 blocks/SM is within 6 % of the best.
- **Occupancy is not the limiter.** `gpu-shuffle` at 50 % theoretical occupancy (53 registers) matches
  the 100 %-occupancy XORWOW kernel. The kernel is compute bound, with no DRAM traffic in the loop.
- **Special functions cost more than arithmetic.** Steps that call `expf` (arithmetic Asian, log
  scheme) or `logf` (geometric Asian, Euler) run at 50–70 % of the speed of steps without them.
- **GPU clocks.** The A30 idles at 930 MHz and needs about 100 ms of load to reach 1440 MHz, so short
  runs from idle measured 1.55× slower (`results/A30_no_spinup/`). The L40 starts at 2490 MHz but is
  power-limited to a median of about 2175 MHz under sustained load. The driver therefore keeps the GPU
  busy for 300 ms before timing each GPU engine, and the job logs the clocks
  (`results/*/gpu_clocks.csv`).
- **Nsight Compute** was not available: the cluster blocks GPU performance counters for users
  (`ERR_NVGPUCTRPERM`). Register and occupancy figures come from the CUDA occupancy API
  (`--kernel-info`), and the timeline comes from Nsight Systems.

## 11. Bottlenecks and risks

- **Compute bound.** The time loop touches no DRAM, so throughput is limited by the FP32 FMA pipe and
  the special-function units (exp/log/sqrt/sincos in Box–Muller and in the Asian `expf`). This is
  confirmed by the step-cost and style results above. `make FAST_MATH=1` switches to the intrinsics.
- **Register pressure** limits occupancy to 50 % on the A30. Here it does not cost throughput (see
  the XORWOW comparison). `__launch_bounds__(256)` and `make regs` keep it visible.
- **Float accumulation over 10⁸ payoffs** is handled by per-thread float sums plus double block totals.
  `gpu-double` agrees within its standard error.
- **Euler bias** is O(Δt) on S, about 0.001 at 252 steps. The log scheme is exact for GBM and is used
  for the exact checks.
- **Fair comparison:** the CPU baseline runs the same algorithm at `-O3` with OpenMP on all 64 threads,
  with the same N and the same variance reduction. It was compiled without `-march=native`, because
  the binaries are built on the login node, so its SIMD use is conservative. End-to-end and kernel-only
  times are reported separately.
- **Measurement noise:** GPU clock state (above) and the second GPU on each node, which other users
  may load, are both possible sources. Medians of 3 runs after a 300 ms spin-up are used.

## 12. Milestones (from the proposal)

1. Black–Scholes analytical pricer and CPU Monte Carlo (single-core and OpenMP): `black_scholes.c`,
   `cpu_mc.c`.
2. CUDA kernel for GBM paths with register time stepping: `sample_philox`, `path_model.h`.
3. cuRAND integration and warp-shuffle reduction: `mc_shuffle_kernel`, `block_shuffle_add`.
4. Profiling with Nsight Systems and Nsight Compute (registers, occupancy, bandwidth), plus a
   roofline: `scripts/profile.sh`, `scripts/roofline.py`. Done with Nsight Systems and the CUDA
   occupancy API; Nsight Compute and the measured roofline are pending counter access on the cluster.
