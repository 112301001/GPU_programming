/* Shared types for the Monte Carlo options pricing engine (C, also included by CUDA). */
#ifndef MC_COMMON_H
#define MC_COMMON_H

#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { OPT_CALL = 0, OPT_PUT = 1 } OptionType;

/* Payoff style.
 *   STYLE_EUROPEAN    : payoff on the terminal price S_T
 *   STYLE_ASIAN_ARITH : payoff on the arithmetic average of S(t_1..t_M)
 *   STYLE_ASIAN_GEO   : payoff on the geometric average of S(t_1..t_M)
 *                       (has a closed form, used to validate path-dependent pricing) */
typedef enum { STYLE_EUROPEAN = 0, STYLE_ASIAN_ARITH = 1, STYLE_ASIAN_GEO = 2 } Style;

/* Time discretisation of GBM  dS = r S dt + sigma S dW.
 *   SCHEME_EULER : Euler-Maruyama on S:      S += r S dt + sigma S sqrt(dt) Z
 *                  (what the proposal names; O(dt) bias)
 *   SCHEME_LOG   : Euler-Maruyama on log S:  x += (r - sigma^2/2) dt + sigma sqrt(dt) Z
 *                  (exact for GBM: no discretisation bias, S stays positive) */
typedef enum { SCHEME_EULER = 0, SCHEME_LOG = 1 } Scheme;

typedef struct {
    double S0;     /* spot */
    double K;      /* strike */
    double r;      /* risk-free rate */
    double sigma;  /* volatility */
    double T;      /* maturity in years */
    OptionType type;
    Style style;
} OptionParams;

typedef struct {
    uint64_t paths;     /* simulated paths (both legs of an antithetic pair count) */
    int steps;          /* time steps per path (M) */
    Scheme scheme;
    int antithetic;     /* 0/1 */
    uint64_t seed;
    int cpu_threads;    /* OpenMP threads, 0 = all cores */
    int block_size;     /* CUDA threads per block (power of two, 32..MC_MAX_BLOCK) */
    int blocks_per_sm;  /* grid = blocks_per_sm x number of SMs */
} SimConfig;

typedef struct {
    double price;
    double std_error;
    double time_ms;    /* end to end: RNG, simulation, reduction, copies */
    double kernel_ms;  /* GPU kernel time only (cudaEvent); equals time_ms for CPU */
    uint64_t samples;  /* independent estimates (antithetic pairs when antithetic is on) */
    uint64_t paths;    /* simulated paths */
} MCResult;

static inline OptionParams default_option(void) {
    OptionParams o;
    o.S0 = 100.0;
    o.K = 100.0;
    o.r = 0.05;
    o.sigma = 0.2;
    o.T = 1.0;
    o.type = OPT_CALL;
    o.style = STYLE_EUROPEAN;
    return o;
}

static inline SimConfig default_config(void) {
    SimConfig c;
    c.paths = 1ull << 24;
    c.steps = 252;
    c.scheme = SCHEME_EULER;
    c.antithetic = 1;
    c.seed = 1234;
    c.cpu_threads = 0;
    c.block_size = 256;
    c.blocks_per_sm = 8;
    return c;
}

/* Number of independent samples. With antithetic variates one sample is the
 * average payoff of the path driven by Z and its mirror driven by -Z. */
static inline uint64_t samples_for(const SimConfig* cfg) {
    return cfg->antithetic ? (cfg->paths + 1) / 2 : cfg->paths;
}

/* Raw sums of undiscounted payoffs -> discounted price and standard error:
 *   mean = sum/n,  var = (sumsq - n mean^2)/(n-1),  SE = sqrt(var/n). */
static inline MCResult make_result(double sum, double sumsq, uint64_t n, const OptionParams* opt,
                                   const SimConfig* cfg, double time_ms, double kernel_ms) {
    MCResult res;
    const double disc = exp(-opt->r * opt->T);
    const double mean = n ? sum / (double)n : 0.0;
    double var = 0.0;
    if (n > 1) {
        var = (sumsq - (double)n * mean * mean) / (double)(n - 1);
        if (var < 0.0) var = 0.0;
    }
    res.price = disc * mean;
    res.std_error = n ? disc * sqrt(var / (double)n) : 0.0;
    res.time_ms = time_ms;
    res.kernel_ms = kernel_ms;
    res.samples = n;
    res.paths = cfg->antithetic ? 2 * n : n;
    return res;
}

/* Wall-clock milliseconds from a monotonic clock. */
static inline double mc_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static inline const char* style_name(Style s) {
    return s == STYLE_EUROPEAN ? "european" : s == STYLE_ASIAN_ARITH ? "asian" : "geo-asian";
}
static inline const char* type_name(OptionType t) { return t == OPT_CALL ? "call" : "put"; }
static inline const char* scheme_name(Scheme s) { return s == SCHEME_EULER ? "euler" : "log"; }

#ifdef __cplusplus
}
#endif

#endif /* MC_COMMON_H */
