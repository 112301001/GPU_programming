/* CPU Monte Carlo baseline: OpenMP, one xoshiro256** generator per thread. */
#ifndef MC_CPU_MC_H
#define MC_CPU_MC_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Uses cfg->cpu_threads OpenMP threads (0 = all cores, 1 = single-core baseline). */
MCResult price_cpu(const OptionParams* opt, const SimConfig* cfg);

/* Number of threads price_cpu uses for this configuration. */
int cpu_thread_count(const SimConfig* cfg);

#ifdef __cplusplus
}
#endif

#endif
