/* GPU Monte Carlo engine for options pricing (CUDA C).
 *
 * Proposed design (mc_shuffle_kernel):
 *   - thread mapping: one thread simulates one sample (a path, or an
 *     antithetic pair) at a time in a grid-stride loop; the grid is sized to
 *     the hardware (blocks_per_sm x #SMs), not to N, so one launch handles any N;
 *   - random numbers: cuRAND device API, Philox4x32-10 state in registers,
 *     curand_init(seed, tid, 0) gives every thread its own subsequence,
 *     curand_normal4 gives four normals (= four time steps) per call;
 *   - time stepping: the whole path lives in registers (path_model.h); option
 *     and model parameters are in __constant__ memory;
 *   - reduction: per-thread float sums -> __shfl_down_sync tree inside each
 *     warp -> shared[32] -> __syncthreads -> warp 0 shuffles again -> ONE
 *     double atomicAdd per block for sum and for sum of squares.
 *
 * The other kernels change one of those decisions each (see gpu_mc.h). */

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpu_mc.h"
#include "path_model.h"

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define MC_NVTX 1
#endif
#endif
#ifdef MC_NVTX
#define NVTX_PUSH(name) nvtxRangePushA(name) /* named ranges in the Nsight Systems timeline */
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP() ((void)0)
#endif

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t err_ = (call);                                                                       \
        if (err_ != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                          \
        }                                                                                                \
    } while (0)

#define CURAND_CHECK(call)                                                               \
    do {                                                                                 \
        curandStatus_t st_ = (call);                                                     \
        if (st_ != CURAND_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuRAND error %d at %s:%d\n", (int)st_, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

typedef unsigned long long u64;

#define FULL_MASK 0xffffffffu
#define RAND_BUFFER_BYTES (256ull << 20) /* normals per batch for the pre-generated baselines */

/* Option/model parameters, written once per pricing with cudaMemcpyToSymbol. */
__constant__ ModelConsts c_model;

/* ======================================================================== */
/* Block reductions: each ends with ONE double atomicAdd per block           */
/* ======================================================================== */

/* Warp shuffles -> shared[32] -> warp 0 shuffles -> atomicAdd (proposed). */
static __device__ __forceinline__ void block_shuffle_add(float sum, float sumsq, double* d_acc) {
    /* 1) reduce inside each warp: 5 steps, no shared memory, no barrier */
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, off);
        sumsq += __shfl_down_sync(FULL_MASK, sumsq, off);
    }
    /* 2) lane 0 of each warp publishes its warp's total */
    __shared__ float w_sum[32], w_sq[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        w_sum[warp] = sum;
        w_sq[warp] = sumsq;
    }
    __syncthreads();
    /* 3) warp 0 reduces the per-warp totals and issues one atomic per block */
    if (warp == 0) {
        const int n_warps = blockDim.x >> 5;
        sum = (lane < n_warps) ? w_sum[lane] : 0.0f;
        sumsq = (lane < n_warps) ? w_sq[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(FULL_MASK, sum, off);
            sumsq += __shfl_down_sync(FULL_MASK, sumsq, off);
        }
        if (lane == 0) {
            atomicAdd(&d_acc[0], (double)sum); /* double atomicAdd: compute capability >= 6.0 */
            atomicAdd(&d_acc[1], (double)sumsq);
        }
    }
}

/* Same in double precision (used by the double ablation kernel). */
static __device__ __forceinline__ void block_shuffle_add_double(double sum, double sumsq, double* d_acc) {
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, off);
        sumsq += __shfl_down_sync(FULL_MASK, sumsq, off);
    }
    __shared__ double w_sum[32], w_sq[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        w_sum[warp] = sum;
        w_sq[warp] = sumsq;
    }
    __syncthreads();
    if (warp == 0) {
        const int n_warps = blockDim.x >> 5;
        sum = (lane < n_warps) ? w_sum[lane] : 0.0;
        sumsq = (lane < n_warps) ? w_sq[lane] : 0.0;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(FULL_MASK, sum, off);
            sumsq += __shfl_down_sync(FULL_MASK, sumsq, off);
        }
        if (lane == 0) {
            atomicAdd(&d_acc[0], sum);
            atomicAdd(&d_acc[1], sumsq);
        }
    }
}

/* Classic shared-memory tree with sequential addressing (no bank conflicts):
 * log2(blockDim) levels, a __syncthreads after each. blockDim = power of two.
 * Needs 2 * blockDim.x floats of dynamic shared memory. */
static __device__ __forceinline__ void block_shared_tree_add(float sum, float sumsq, double* d_acc) {
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sq = smem + blockDim.x;
    const unsigned t = threadIdx.x;
    s_sum[t] = sum;
    s_sq[t] = sumsq;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            s_sum[t] += s_sum[t + stride];
            s_sq[t] += s_sq[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        atomicAdd(&d_acc[0], (double)s_sum[0]);
        atomicAdd(&d_acc[1], (double)s_sq[0]);
    }
}

/* ======================================================================== */
/* One Monte Carlo sample, all in registers                                  */
/* ======================================================================== */

/* Philox: curand_normal4 -> four time steps per RNG call. */
static __device__ __forceinline__ float sample_philox(curandStatePhilox4_32_10_t* rng) {
    const ModelConsts* c = &c_model;
    const int steps = c->steps;
    PathState a, b;
    path_init(&a, c);
    path_init(&b, c);
    int j = 0;
    for (; j + 4 <= steps; j += 4) {
        const float4 z = curand_normal4(rng);
        path_advance(&a, &b, z.x, c);
        path_advance(&a, &b, z.y, c);
        path_advance(&a, &b, z.z, c);
        path_advance(&a, &b, z.w, c);
    }
    if (j < steps) { /* last 1-3 steps when M is not a multiple of 4 */
        const float4 z = curand_normal4(rng);
        path_advance(&a, &b, z.x, c);
        if (j + 1 < steps) path_advance(&a, &b, z.y, c);
        if (j + 2 < steps) path_advance(&a, &b, z.z, c);
    }
    return sample_value(&a, &b, c);
}

/* XORWOW: one normal per curand_normal call. */
static __device__ __forceinline__ float sample_xorwow(curandState* rng) {
    const ModelConsts* c = &c_model;
    PathState a, b;
    path_init(&a, c);
    path_init(&b, c);
    for (int j = 0; j < c->steps; ++j) path_advance(&a, &b, curand_normal(rng), c);
    return sample_value(&a, &b, c);
}

/* Double-precision path (float-vs-double ablation). */
static __device__ __forceinline__ void dpath_step(double* s, double* x, double* acc, double z) {
    const ModelConsts* c = &c_model;
    if (c->scheme == SCHEME_EULER) {
        *s = fmax(*s * (c->d_one_plus_mu_dt + c->d_vol_sqdt * z), 0.0);
        if (c->style == STYLE_ASIAN_ARITH) *acc += *s;
        else if (c->style == STYLE_ASIAN_GEO) *acc += log(*s);
    } else {
        *x += c->d_drift_dt + c->d_vol_sqdt * z;
        if (c->style == STYLE_ASIAN_ARITH) *acc += exp(*x);
        else if (c->style == STYLE_ASIAN_GEO) *acc += *x;
    }
}

static __device__ __forceinline__ double dpath_payoff(double s, double x, double acc) {
    const ModelConsts* c = &c_model;
    double u;
    if (c->style == STYLE_EUROPEAN) u = (c->scheme == SCHEME_EULER) ? s : exp(x);
    else if (c->style == STYLE_ASIAN_ARITH) u = acc * c->d_inv_steps;
    else u = exp(acc * c->d_inv_steps);
    return fmax(c->is_call ? (u - c->d_strike) : (c->d_strike - u), 0.0);
}

static __device__ __forceinline__ double sample_double(curandStatePhilox4_32_10_t* rng) {
    const ModelConsts* c = &c_model;
    const int steps = c->steps;
    const int anti = c->antithetic;
    double sa = c->d_s0, xa = c->d_log_s0, aa = 0.0;
    double sb = c->d_s0, xb = c->d_log_s0, ab = 0.0;
    for (int j = 0; j < steps; j += 2) {
        const double2 z = curand_normal2_double(rng); /* two normals per call */
        dpath_step(&sa, &xa, &aa, z.x);
        if (anti) dpath_step(&sb, &xb, &ab, -z.x);
        if (j + 1 < steps) {
            dpath_step(&sa, &xa, &aa, z.y);
            if (anti) dpath_step(&sb, &xb, &ab, -z.y);
        }
    }
    double v = dpath_payoff(sa, xa, aa);
    if (anti) v = 0.5 * (v + dpath_payoff(sb, xb, ab));
    return v;
}

/* ======================================================================== */
/* Kernels                                                                   */
/* ======================================================================== */

/* PROPOSED: Philox in registers + warp-shuffle reduction. */
__global__ void __launch_bounds__(MC_MAX_BLOCK) mc_shuffle_kernel(u64 n, u64 seed, double* d_acc) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    curandStatePhilox4_32_10_t rng;   /* counter-based PRNG, state in registers */
    curand_init(seed, tid, 0, &rng);  /* subsequence = thread id -> independent streams */
    float sum = 0.0f, sumsq = 0.0f;   /* thread-local accumulators (registers) */
    for (u64 i = tid; i < n; i += stride) {
        const float v = sample_philox(&rng);
        sum += v;
        sumsq += v * v;
    }
    block_shuffle_add(sum, sumsq, d_acc);
}

/* Ablation: shared-memory tree reduction instead of warp shuffles. */
__global__ void __launch_bounds__(MC_MAX_BLOCK) mc_shared_kernel(u64 n, u64 seed, double* d_acc) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, tid, 0, &rng);
    float sum = 0.0f, sumsq = 0.0f;
    for (u64 i = tid; i < n; i += stride) {
        const float v = sample_philox(&rng);
        sum += v;
        sumsq += v * v;
    }
    block_shared_tree_add(sum, sumsq, d_acc);
}

/* Ablation: no block reduction at all; one global atomicAdd per sample. */
__global__ void __launch_bounds__(MC_MAX_BLOCK) mc_atomic_kernel(u64 n, u64 seed, double* d_acc) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, tid, 0, &rng);
    for (u64 i = tid; i < n; i += stride) {
        const float v = sample_philox(&rng);
        atomicAdd(&d_acc[0], (double)v);
        atomicAdd(&d_acc[1], (double)v * v);
    }
}

/* Ablation: XORWOW generator (cuRAND default) instead of Philox. */
__global__ void __launch_bounds__(MC_MAX_BLOCK) mc_xorwow_kernel(u64 n, u64 seed, double* d_acc) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    curandState rng;                  /* 48-byte state; init with subsequence = skip-ahead */
    curand_init(seed, tid, 0, &rng);
    float sum = 0.0f, sumsq = 0.0f;
    for (u64 i = tid; i < n; i += stride) {
        const float v = sample_xorwow(&rng);
        sum += v;
        sumsq += v * v;
    }
    block_shuffle_add(sum, sumsq, d_acc);
}

/* Ablation: double precision everywhere. */
__global__ void __launch_bounds__(MC_MAX_BLOCK) mc_double_kernel(u64 n, u64 seed, double* d_acc) {
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, tid, 0, &rng);
    double sum = 0.0, sumsq = 0.0;
    for (u64 i = tid; i < n; i += stride) {
        const double v = sample_double(&rng);
        sum += v;
        sumsq += v * v;
    }
    block_shuffle_add_double(sum, sumsq, d_acc);
}

/* Baseline: normals pre-generated in global memory, layout z[step * batch + sample]
 * so that consecutive threads read consecutive addresses (coalesced). Same
 * shuffle reduction, so the only difference is where the random numbers come from. */
__global__ void __launch_bounds__(MC_MAX_BLOCK) mc_pregen_kernel(const float* __restrict__ z, u64 batch,
                                                                 double* d_acc) {
    const ModelConsts* c = &c_model;
    const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 stride = (u64)gridDim.x * blockDim.x;
    float sum = 0.0f, sumsq = 0.0f;
    for (u64 i = tid; i < batch; i += stride) {
        PathState a, b;
        path_init(&a, c);
        path_init(&b, c);
        for (int j = 0; j < c->steps; ++j) path_advance(&a, &b, z[(u64)j * batch + i], c);
        const float v = sample_value(&a, &b, c);
        sum += v;
        sumsq += v * v;
    }
    block_shuffle_add(sum, sumsq, d_acc);
}

/* ======================================================================== */
/* Host side                                                                 */
/* ======================================================================== */

static void check_config(const SimConfig* cfg) {
    const int bs = cfg->block_size;
    if (bs < 32 || bs > MC_MAX_BLOCK || (bs & (bs - 1)) != 0) {
        fprintf(stderr, "error: block size must be a power of two in [32, %d]\n", MC_MAX_BLOCK);
        exit(EXIT_FAILURE);
    }
    if (cfg->blocks_per_sm < 1 || cfg->steps < 1) {
        fprintf(stderr, "error: blocks-per-sm and steps must be >= 1\n");
        exit(EXIT_FAILURE);
    }
}

static int sm_count(void) {
    int dev = 0, n_sm = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev));
    return n_sm;
}

/* blocks_per_sm x #SMs blocks (enough resident warps to hide latency), but no
 * more blocks than there is work for. */
static int grid_for(u64 n, const SimConfig* cfg) {
    u64 blocks = (u64)sm_count() * (u64)cfg->blocks_per_sm;
    const u64 needed = (n + cfg->block_size - 1) / cfg->block_size;
    if (needed < blocks) blocks = needed;
    return blocks < 1 ? 1 : (int)blocks;
}

extern "C" void gpu_launch_dims(const SimConfig* cfg, int* blocks, int* threads) {
    *blocks = grid_for(samples_for(cfg), cfg);
    *threads = cfg->block_size;
}

/* Engines that generate random numbers inside the kernel. */
static double run_fused(GpuEngine engine, const ModelConsts* c, const SimConfig* cfg, u64 n, double* sum,
                        double* sumsq, double* kernel_ms) {
    const int threads = cfg->block_size;
    const int blocks = grid_for(n, cfg);

    double* d_acc = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_acc, 2 * sizeof(double)));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    const double t0 = mc_now_ms();
    CUDA_CHECK(cudaMemcpyToSymbol(c_model, c, sizeof(ModelConsts)));
    CUDA_CHECK(cudaMemset(d_acc, 0, 2 * sizeof(double)));
    CUDA_CHECK(cudaEventRecord(start, 0));
    switch (engine) {
        case GPU_SHUFFLE: mc_shuffle_kernel<<<blocks, threads>>>(n, cfg->seed, d_acc); break;
        case GPU_SHARED:
            mc_shared_kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>(n, cfg->seed, d_acc);
            break;
        case GPU_ATOMIC: mc_atomic_kernel<<<blocks, threads>>>(n, cfg->seed, d_acc); break;
        case GPU_XORWOW: mc_xorwow_kernel<<<blocks, threads>>>(n, cfg->seed, d_acc); break;
        case GPU_DOUBLE: mc_double_kernel<<<blocks, threads>>>(n, cfg->seed, d_acc); break;
        default: fprintf(stderr, "internal error: not a fused engine\n"); exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    double h[2];
    CUDA_CHECK(cudaMemcpy(h, d_acc, sizeof(h), cudaMemcpyDeviceToHost)); /* waits for the kernel */
    const double t1 = mc_now_ms();

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    *kernel_ms = ms;
    *sum = h[0];
    *sumsq = h[1];
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_acc));
    return t1 - t0;
}

/* Engines that pre-generate all normals into a global-memory array first.
 * Large problems are processed in batches of at most RAND_BUFFER_BYTES. */
static double run_pregen(int host_rng, const ModelConsts* c, const SimConfig* cfg, u64 n, double* sum,
                         double* sumsq, double* kernel_ms) {
    const int steps = cfg->steps;
    const int threads = cfg->block_size;
    u64 batch = (RAND_BUFFER_BYTES / sizeof(float)) / (u64)steps;
    if (batch < 1) batch = 1;
    if (batch > n) batch = n;
    /* Always ask cuRAND for a multiple of 4 normals: the API needs an even count,
     * and the CUDA 11.x host Philox generator crashes when count % 4 != 0. */
    const size_t buf_floats = ((size_t)batch * steps + 3) & ~(size_t)3;

    curandGenerator_t gen;
    if (host_rng) CURAND_CHECK(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    else CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, cfg->seed));

    float *d_z = NULL, *h_z = NULL;
    double* d_acc = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_z, buf_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_acc, 2 * sizeof(double)));
    if (host_rng) CUDA_CHECK(cudaMallocHost((void**)&h_z, buf_floats * sizeof(float))); /* pinned */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    double kms_total = 0.0;
    const double t0 = mc_now_ms();
    CUDA_CHECK(cudaMemcpyToSymbol(c_model, c, sizeof(ModelConsts)));
    CUDA_CHECK(cudaMemset(d_acc, 0, 2 * sizeof(double)));
    for (u64 done = 0; done < n;) {
        const u64 cur = (n - done < batch) ? n - done : batch;
        const size_t count = (size_t)cur * steps;
        const size_t count4 = (count + 3) & ~(size_t)3; /* <= buf_floats; extras unused */
        if (host_rng) {
            CURAND_CHECK(curandGenerateNormal(gen, h_z, count4, 0.0f, 1.0f));               /* CPU */
            CUDA_CHECK(cudaMemcpy(d_z, h_z, count * sizeof(float), cudaMemcpyHostToDevice));  /* PCIe */
        } else {
            CURAND_CHECK(curandGenerateNormal(gen, d_z, count4, 0.0f, 1.0f));               /* GPU */
        }
        const int blocks = grid_for(cur, cfg);
        CUDA_CHECK(cudaEventRecord(start, 0));
        mc_pregen_kernel<<<blocks, threads>>>(d_z, cur, d_acc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        kms_total += ms;
        done += cur;
    }
    double h[2];
    CUDA_CHECK(cudaMemcpy(h, d_acc, sizeof(h), cudaMemcpyDeviceToHost));
    const double t1 = mc_now_ms();

    *kernel_ms = kms_total;
    *sum = h[0];
    *sumsq = h[1];
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_acc));
    if (h_z) CUDA_CHECK(cudaFreeHost(h_z));
    return t1 - t0;
}

extern "C" const char* gpu_engine_name(GpuEngine e) {
    switch (e) {
        case GPU_HOSTRNG: return "gpu-hostrng";
        case GPU_PREGEN: return "gpu-pregen";
        case GPU_ATOMIC: return "gpu-atomic";
        case GPU_SHARED: return "gpu-shared";
        case GPU_SHUFFLE: return "gpu-shuffle";
        case GPU_XORWOW: return "gpu-xorwow";
        case GPU_DOUBLE: return "gpu-double";
        default: return "?";
    }
}

extern "C" int gpu_available(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

extern "C" void gpu_device_info(char* buf, size_t len) {
    int dev = 0, driver = 0, runtime = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    struct cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    snprintf(buf, len, "%s (cc %d.%d, %d SMs, %zu MiB, CUDA runtime %d.%d, driver %d.%d)", p.name, p.major,
             p.minor, p.multiProcessorCount, (size_t)(p.totalGlobalMem >> 20), runtime / 1000,
             (runtime % 1000) / 10, driver / 1000, (driver % 1000) / 10);
}

extern "C" MCResult price_gpu(GpuEngine engine, const OptionParams* opt, const SimConfig* cfg) {
    check_config(cfg);
    NVTX_PUSH(gpu_engine_name(engine));
    const ModelConsts c = make_consts(opt, cfg);
    const u64 n = samples_for(cfg);
    double sum = 0.0, sumsq = 0.0, kms = 0.0, ms = 0.0;
    if (engine == GPU_HOSTRNG || engine == GPU_PREGEN)
        ms = run_pregen(engine == GPU_HOSTRNG, &c, cfg, n, &sum, &sumsq, &kms);
    else
        ms = run_fused(engine, &c, cfg, n, &sum, &sumsq, &kms);
    NVTX_POP();
    return make_result(sum, sumsq, n, opt, cfg, ms, kms);
}

extern "C" void gpu_warmup(const OptionParams* opt, const SimConfig* cfg) {
    CUDA_CHECK(cudaFree(0)); /* create the context */
    SimConfig small = *cfg;
    small.paths = 1 << 16;
    for (int e = GPU_PREGEN; e < GPU_ENGINE_COUNT; ++e) price_gpu((GpuEngine)e, opt, &small);
}

extern "C" void gpu_spin_up(double ms) {
    /* GPUs lower their clock when idle (DVFS) and need ~100 ms of load to reach the
     * boost clock. Short timed runs started from idle would otherwise measure the
     * base clock (on the A30, 930 vs 1440 MHz). Keep the GPU busy for `ms` first. */
    NVTX_PUSH("spin-up");
    OptionParams o = default_option();
    SimConfig cfg = default_config();
    cfg.paths = 1ull << 24;
    cfg.steps = 64;
    const ModelConsts c = make_consts(&o, &cfg);
    double sum, sumsq, kms;
    const double t0 = mc_now_ms();
    do {
        run_fused(GPU_SHUFFLE, &c, &cfg, samples_for(&cfg), &sum, &sumsq, &kms);
    } while (mc_now_ms() - t0 < ms);
    NVTX_POP();
}

static void describe_kernel(const char* name, const void* kernel, int block_size, size_t dyn_smem) {
    struct cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_size, dyn_smem));
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const double occ = (double)blocks_per_sm * block_size / prop.maxThreadsPerMultiProcessor;
    printf("  %-18s %3d regs/thread  %5zu B smem  %4zu B local(spill)  %2d blocks/SM  occupancy %3.0f%%\n", name,
           attr.numRegs, attr.sharedSizeBytes + dyn_smem, attr.localSizeBytes, blocks_per_sm, occ * 100.0);
}

extern "C" void gpu_kernel_report(const SimConfig* cfg) {
    check_config(cfg);
    const int bs = cfg->block_size;
    printf("  (block size %d; theoretical occupancy from cudaOccupancyMaxActiveBlocksPerMultiprocessor)\n", bs);
    describe_kernel("mc_shuffle_kernel", (const void*)mc_shuffle_kernel, bs, 0);
    describe_kernel("mc_shared_kernel", (const void*)mc_shared_kernel, bs, 2 * bs * sizeof(float));
    describe_kernel("mc_atomic_kernel", (const void*)mc_atomic_kernel, bs, 0);
    describe_kernel("mc_xorwow_kernel", (const void*)mc_xorwow_kernel, bs, 0);
    describe_kernel("mc_double_kernel", (const void*)mc_double_kernel, bs, 0);
    describe_kernel("mc_pregen_kernel", (const void*)mc_pregen_kernel, bs, 0);
    int blocks, threads;
    gpu_launch_dims(cfg, &blocks, &threads);
    printf("  grid: %d blocks x %d threads (%d SMs x %d blocks/SM) for %llu samples\n", blocks, threads, sm_count(),
           cfg->blocks_per_sm, (unsigned long long)samples_for(cfg));
}
