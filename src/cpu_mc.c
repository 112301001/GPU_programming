/* CPU Monte Carlo baseline (C + OpenMP).
 *
 * Same algorithm and the same path arithmetic as the GPU (path_model.h).
 * Each OpenMP thread owns an independent xoshiro256** generator (seeded with
 * splitmix64 from the seed and the thread number) and draws normals with the
 * Marsaglia polar method. The sums are combined with an OpenMP reduction. */
#include "cpu_mc.h"

#include <omp.h>

#include "path_model.h"

/* ---- RNG: xoshiro256** (Blackman & Vigna) -------------------------------- */

typedef struct {
    uint64_t s[4];
    float spare;
    int has_spare;
} Rng;

static uint64_t splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void rng_seed(Rng* g, uint64_t seed, uint64_t stream) {
    uint64_t x = seed ^ (0xD1B54A32D192ED03ull * (stream + 1));
    for (int i = 0; i < 4; ++i) g->s[i] = splitmix64(&x);
    g->has_spare = 0;
    g->spare = 0.0f;
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(Rng* g) {
    uint64_t* s = g->s;
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

/* Uniform in (-1, 1) with 24 random bits. */
static inline float rng_sym_uniform(Rng* g) {
    return (float)(rng_next(g) >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

/* Standard normal, Marsaglia polar method (two values per accepted pair). */
static inline float rng_normal(Rng* g) {
    if (g->has_spare) {
        g->has_spare = 0;
        return g->spare;
    }
    float u, v, s;
    do {
        u = rng_sym_uniform(g);
        v = rng_sym_uniform(g);
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    const float m = sqrtf(-2.0f * logf(s) / s);
    g->spare = v * m;
    g->has_spare = 1;
    return u * m;
}

/* ---- Pricing --------------------------------------------------------------- */

int cpu_thread_count(const SimConfig* cfg) {
    int t = cfg->cpu_threads > 0 ? cfg->cpu_threads : omp_get_max_threads();
    const uint64_t n = samples_for(cfg);
    if (n > 0 && (uint64_t)t > n) t = (int)n;
    return t < 1 ? 1 : t;
}

MCResult price_cpu(const OptionParams* opt, const SimConfig* cfg) {
    const ModelConsts c = make_consts(opt, cfg);
    const int64_t n = (int64_t)samples_for(cfg);
    const int nthreads = cpu_thread_count(cfg);
    double sum = 0.0, sumsq = 0.0;

    const double t0 = mc_now_ms();
#pragma omp parallel num_threads(nthreads) reduction(+ : sum, sumsq)
    {
        Rng g;
        rng_seed(&g, cfg->seed, (uint64_t)omp_get_thread_num());
#pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            PathState a, b;
            path_init(&a, &c);
            path_init(&b, &c);
            for (int j = 0; j < c.steps; ++j) path_advance(&a, &b, rng_normal(&g), &c);
            const double v = sample_value(&a, &b, &c);
            sum += v;
            sumsq += v * v;
        }
    }
    const double t1 = mc_now_ms();
    return make_result(sum, sumsq, (uint64_t)n, opt, cfg, t1 - t0, t1 - t0);
}
