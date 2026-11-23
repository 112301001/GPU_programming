/* CUDA Monte Carlo engines. Implemented in src/gpu_mc.cu, callable from C.
 *
 * GPU_SHUFFLE is the engine described in the proposal. The others each change
 * exactly one design decision, so comparing them is an ablation study. */
#ifndef MC_GPU_MC_H
#define MC_GPU_MC_H

#include <stddef.h>

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Largest block size the kernels are compiled for (__launch_bounds__). */
#ifndef MC_MAX_BLOCK
#define MC_MAX_BLOCK 256
#endif

typedef enum {
    /* Normals generated on the HOST (cuRAND host API) and copied over PCIe
     * into a global-memory array, then read by the path kernel. */
    GPU_HOSTRNG = 0,
    /* Normals pre-generated on the device (cuRAND host API, device
     * generator) into a global-memory array: no PCIe, still memory bound. */
    GPU_PREGEN,
    /* On-device Philox in registers, but every sample does a global
     * atomicAdd (N atomics on one address, serialised in L2). */
    GPU_ATOMIC,
    /* On-device Philox; classic shared-memory tree reduction per block
     * (__syncthreads each level), then one atomicAdd per block. */
    GPU_SHARED,
    /* PROPOSED ENGINE: on-device Philox, curand_normal4, warp-shuffle
     * reduction (__shfl_down_sync) -> shared[32] -> one atomicAdd per block. */
    GPU_SHUFFLE,
    /* As GPU_SHUFFLE with the XORWOW generator instead of Philox. */
    GPU_XORWOW,
    /* As GPU_SHUFFLE with the whole path in double precision. */
    GPU_DOUBLE,
    GPU_ENGINE_COUNT
} GpuEngine;

const char* gpu_engine_name(GpuEngine e);

int gpu_available(void);
void gpu_device_info(char* buf, size_t len);

/* Grid used for this configuration: blocks_per_sm x SMs, capped by the work. */
void gpu_launch_dims(const SimConfig* cfg, int* blocks, int* threads);

/* Run small pricings so context creation and module loading are not timed. */
void gpu_warmup(const OptionParams* opt, const SimConfig* cfg);

/* Keep the GPU busy for `ms` milliseconds so it is at its boost clock before a
 * timed run (GPUs drop to a low clock when idle). */
void gpu_spin_up(double ms);

MCResult price_gpu(GpuEngine engine, const OptionParams* opt, const SimConfig* cfg);

/* Print registers/thread, shared and local (spill) memory and theoretical
 * occupancy of every kernel for this configuration. */
void gpu_kernel_report(const SimConfig* cfg);

#ifdef __cplusplus
}
#endif

#endif
