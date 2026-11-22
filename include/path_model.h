/* GBM time stepping and payoffs, shared by the CPU baseline (C) and the CUDA
 * kernels, so both run exactly the same arithmetic.
 *
 * A path is three floats (PathState) that live in registers: no time step is
 * ever written to memory. The option/model switches (style, scheme,
 * antithetic) are the same for every thread, so the branches on them never
 * cause warp divergence. */
#ifndef MC_PATH_MODEL_H
#define MC_PATH_MODEL_H

#include <math.h>

#include "common.h"

#if defined(__CUDACC__)
#define MC_HD static __host__ __device__ __forceinline__
#elif defined(__GNUC__)
#define MC_HD static inline __attribute__((always_inline))
#else
#define MC_HD static inline
#endif

/* Everything a path needs, precomputed once on the host. On the GPU this
 * struct sits in __constant__ memory (every thread reads the same address,
 * so the constant cache broadcasts it). */
typedef struct {
    /* single precision, used by the float kernels and the CPU */
    float s0;
    float log_s0;
    float strike;
    float one_plus_mu_dt; /* 1 + r dt                 (Euler on S) */
    float drift_dt;       /* (r - sigma^2 / 2) dt     (Euler on log S) */
    float vol_sqdt;       /* sigma sqrt(dt) */
    float inv_steps;      /* 1 / M                    (Asian averaging) */
    /* double precision copies, used by the float-vs-double ablation kernel */
    double d_s0, d_log_s0, d_strike, d_one_plus_mu_dt, d_drift_dt, d_vol_sqdt, d_inv_steps;
    int is_call;
    int style;      /* Style */
    int scheme;     /* Scheme */
    int antithetic; /* 0/1 */
    int steps;      /* M */
} ModelConsts;

static inline ModelConsts make_consts(const OptionParams* o, const SimConfig* cfg) {
    const double dt = o->T / cfg->steps;
    ModelConsts c;
    c.d_s0 = o->S0;
    c.d_log_s0 = log(o->S0);
    c.d_strike = o->K;
    c.d_one_plus_mu_dt = 1.0 + o->r * dt;
    c.d_drift_dt = (o->r - 0.5 * o->sigma * o->sigma) * dt;
    c.d_vol_sqdt = o->sigma * sqrt(dt);
    c.d_inv_steps = 1.0 / cfg->steps;
    c.s0 = (float)c.d_s0;
    c.log_s0 = (float)c.d_log_s0;
    c.strike = (float)c.d_strike;
    c.one_plus_mu_dt = (float)c.d_one_plus_mu_dt;
    c.drift_dt = (float)c.d_drift_dt;
    c.vol_sqdt = (float)c.d_vol_sqdt;
    c.inv_steps = (float)c.d_inv_steps;
    c.is_call = o->type == OPT_CALL;
    c.style = (int)o->style;
    c.scheme = (int)cfg->scheme;
    c.antithetic = cfg->antithetic ? 1 : 0;
    c.steps = cfg->steps;
    return c;
}

typedef struct {
    float s;    /* current price               (Euler scheme) */
    float x;    /* current log price           (log scheme) */
    float acc;  /* running sum for the Asian average (of S, or of log S) */
} PathState;

MC_HD void path_init(PathState* p, const ModelConsts* c) {
    p->s = c->s0;
    p->x = c->log_s0;
    p->acc = 0.0f;
}

/* One Euler-Maruyama step driven by the standard normal draw z. */
MC_HD void path_step(PathState* p, float z, const ModelConsts* c) {
    if (c->scheme == SCHEME_EULER) {
        /* S += r S dt + sigma S sqrt(dt) Z  ==  S *= (1 + r dt + sigma sqrt(dt) Z); clamp at 0 */
        p->s = fmaxf(p->s * (c->one_plus_mu_dt + c->vol_sqdt * z), 0.0f);
        if (c->style == STYLE_ASIAN_ARITH) p->acc += p->s;
        else if (c->style == STYLE_ASIAN_GEO) p->acc += logf(p->s);
    } else {
        /* x = ln S:  x += (r - sigma^2/2) dt + sigma sqrt(dt) Z */
        p->x += c->drift_dt + c->vol_sqdt * z;
        if (c->style == STYLE_ASIAN_ARITH) p->acc += expf(p->x);
        else if (c->style == STYLE_ASIAN_GEO) p->acc += p->x;
    }
}

/* Step the path with z and, with antithetic variates, its mirror with -z. */
MC_HD void path_advance(PathState* a, PathState* b, float z, const ModelConsts* c) {
    path_step(a, z, c);
    if (c->antithetic) path_step(b, -z, c);
}

/* Undiscounted payoff of a finished path. */
MC_HD float path_payoff(const PathState* p, const ModelConsts* c) {
    float u;
    if (c->style == STYLE_EUROPEAN) u = (c->scheme == SCHEME_EULER) ? p->s : expf(p->x);
    else if (c->style == STYLE_ASIAN_ARITH) u = p->acc * c->inv_steps;
    else u = expf(p->acc * c->inv_steps);
    return fmaxf(c->is_call ? (u - c->strike) : (c->strike - u), 0.0f);  /* branch-free max */
}

/* One Monte Carlo sample: the payoff, or 0.5 [f(Z) + f(-Z)] for an antithetic pair. */
MC_HD float sample_value(const PathState* a, const PathState* b, const ModelConsts* c) {
    float v = path_payoff(a, c);
    if (c->antithetic) v = 0.5f * (v + path_payoff(b, c));
    return v;
}

#endif /* MC_PATH_MODEL_H */
