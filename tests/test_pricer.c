/* Correctness tests: closed forms, CPU Monte Carlo and (when built with CUDA)
 * every GPU engine. Monte Carlo checks use a 4-sigma tolerance (false-failure
 * probability ~6e-5 per check; seeds are fixed so runs are reproducible). */
#include <math.h>
#include <stdio.h>

#include "black_scholes.h"
#include "common.h"
#include "cpu_mc.h"
#ifndef MC_NO_CUDA
#include "gpu_mc.h"
#endif

static int g_failures = 0;
static int g_checks = 0;

static void check(int ok, const char* what) {
    ++g_checks;
    if (!ok) ++g_failures;
    printf("[%s] %s\n", ok ? " OK " : "FAIL", what);
}

static void check_close(double got, double want, double tol, const char* what) {
    char buf[256];
    snprintf(buf, sizeof buf, "%s: got %.6f want %.6f (tol %.2e)", what, got, want, tol);
    check(fabs(got - want) <= tol, buf);
}

/* |mc - ref| <= k * se + bias_tol */
static void check_mc(MCResult r, double ref, const char* what, double bias_tol) {
    char buf[256];
    snprintf(buf, sizeof buf, "%s: price %.6f ref %.6f se %.2e (%+.2f se)", what, r.price, ref, r.std_error,
             r.std_error > 0 ? (r.price - ref) / r.std_error : 0.0);
    check(r.std_error > 0 && fabs(r.price - ref) <= 4.0 * r.std_error + bias_tol, buf);
}

static OptionParams option(OptionType type, Style style) {
    OptionParams o = default_option();
    o.type = type;
    o.style = style;
    return o;
}

static SimConfig config(uint64_t paths, int steps, Scheme scheme, int anti) {
    SimConfig c = default_config();
    c.paths = paths;
    c.steps = steps;
    c.scheme = scheme;
    c.antithetic = anti;
    c.seed = 42;
    return c;
}

static void test_closed_forms(void) {
    printf("\n== closed forms ==\n");
    const OptionParams call = option(OPT_CALL, STYLE_EUROPEAN);
    const OptionParams put = option(OPT_PUT, STYLE_EUROPEAN);
    check_close(black_scholes(&call), 10.450583572185565, 1e-9, "Black-Scholes call (textbook value)");
    check_close(black_scholes(&put), 5.573526022256971, 1e-9, "Black-Scholes put (textbook value)");
    check_close(black_scholes(&call) - black_scholes(&put), call.S0 - call.K * exp(-call.r * call.T), 1e-9,
                "put-call parity");
    check_close(geometric_asian(&call, 1), black_scholes(&call), 1e-9, "geometric Asian with 1 fixing == BS");
    check(geometric_asian(&call, 252) < black_scholes(&call), "geometric Asian call < European call");
    /* Continuous-monitoring limit (Kemna-Vorst): 5.5468 for these parameters. */
    check_close(geometric_asian(&call, 100000), 5.5468, 2e-3, "geometric Asian -> continuous-monitoring limit");
}

static void test_cpu(void) {
    printf("\n== CPU Monte Carlo (OpenMP) ==\n");
    const uint64_t N = 1 << 20;
    OptionParams o;
    SimConfig c;

    o = option(OPT_CALL, STYLE_EUROPEAN);
    c = config(N, 16, SCHEME_LOG, 1);
    check_mc(price_cpu(&o, &c), black_scholes(&o), "cpu european call, log", 0.0);

    o = option(OPT_PUT, STYLE_EUROPEAN);
    c = config(N, 16, SCHEME_LOG, 0);
    check_mc(price_cpu(&o, &c), black_scholes(&o), "cpu european put, log, no antithetic", 0.0);

    /* Euler on S: allow a small discretisation bias on top of the statistical error. */
    o = option(OPT_CALL, STYLE_EUROPEAN);
    c = config(N, 64, SCHEME_EULER, 1);
    check_mc(price_cpu(&o, &c), black_scholes(&o), "cpu european call, euler", 0.02);

    o = option(OPT_CALL, STYLE_ASIAN_GEO);
    c = config(N, 32, SCHEME_LOG, 1);
    check_mc(price_cpu(&o, &c), geometric_asian(&o, 32), "cpu geometric asian call, log", 0.0);

    o = option(OPT_CALL, STYLE_EUROPEAN);
    {
        SimConfig cp = config(N, 16, SCHEME_LOG, 0);
        SimConfig ca = config(N, 16, SCHEME_LOG, 1);
        const MCResult plain = price_cpu(&o, &cp);
        const MCResult anti = price_cpu(&o, &ca);
        char buf[160];
        snprintf(buf, sizeof buf, "antithetic lowers std error at equal path count (%.3e < %.3e)", anti.std_error,
                 plain.std_error);
        check(anti.std_error < plain.std_error, buf);
        check(anti.paths == plain.paths, "antithetic run simulates the same number of paths");
    }
    {
        SimConfig c1 = config(1 << 18, 16, SCHEME_LOG, 1);
        c1.cpu_threads = 1;
        check_mc(price_cpu(&o, &c1), black_scholes(&o), "cpu single thread european call", 0.0);
    }
}

#ifndef MC_NO_CUDA
static void test_gpu(void) {
    printf("\n== GPU Monte Carlo ==\n");
    if (!gpu_available()) {
        check(0, "CUDA device available");
        return;
    }
    char info[256];
    gpu_device_info(info, sizeof info);
    printf("device: %s\n", info);

    const uint64_t N = 1 << 22;
    char what[160];
    OptionParams o;
    SimConfig c;

    for (int e = 0; e < GPU_ENGINE_COUNT; ++e) {
        const GpuEngine eng = (GpuEngine)e;
        const char* name = gpu_engine_name(eng);

        o = option(OPT_CALL, STYLE_EUROPEAN);
        c = config(N, 16, SCHEME_LOG, 1);
        snprintf(what, sizeof what, "%s european call, log", name);
        check_mc(price_gpu(eng, &o, &c), black_scholes(&o), what, 0.0);

        o = option(OPT_PUT, STYLE_EUROPEAN);
        c = config(N, 17, SCHEME_LOG, 0);
        snprintf(what, sizeof what, "%s european put, log, 17 steps (tail), no antithetic", name);
        check_mc(price_gpu(eng, &o, &c), black_scholes(&o), what, 0.0);

        o = option(OPT_CALL, STYLE_ASIAN_GEO);
        c = config(N, 50, SCHEME_LOG, 1);
        snprintf(what, sizeof what, "%s geometric asian call, 50 steps", name);
        check_mc(price_gpu(eng, &o, &c), geometric_asian(&o, 50), what, 0.0);

        o = option(OPT_CALL, STYLE_EUROPEAN);
        c = config(N, 252, SCHEME_EULER, 1);
        snprintf(what, sizeof what, "%s european call, euler, 252 steps", name);
        check_mc(price_gpu(eng, &o, &c), black_scholes(&o), what, 0.01);
    }

    /* atomic / shared / shuffle use the same Philox streams with the same grid,
     * so they compute the same samples; only the summation order differs. */
    o = option(OPT_CALL, STYLE_ASIAN_ARITH);
    c = config(N, 64, SCHEME_EULER, 1);
    {
        const MCResult a = price_gpu(GPU_ATOMIC, &o, &c);
        const MCResult s = price_gpu(GPU_SHARED, &o, &c);
        const MCResult w = price_gpu(GPU_SHUFFLE, &o, &c);
        check_close(a.price, w.price, 1e-5 * w.price, "atomic == shuffle (same samples, float rounding only)");
        check_close(s.price, w.price, 1e-5 * w.price, "shared == shuffle (same samples, float rounding only)");
        const MCResult d = price_gpu(GPU_DOUBLE, &o, &c);
        const double tol = 4.0 * sqrt(d.std_error * d.std_error + w.std_error * w.std_error);
        check_close(d.price, w.price, tol, "double vs float agree statistically");
    }

    /* Arithmetic Asian has no closed form: GPU and CPU must agree statistically. */
    o = option(OPT_CALL, STYLE_ASIAN_ARITH);
    {
        SimConfig cg = config(N, 52, SCHEME_LOG, 1);
        SimConfig cc = config(1 << 20, 52, SCHEME_LOG, 1);
        const MCResult g = price_gpu(GPU_SHUFFLE, &o, &cg);
        const MCResult cr = price_cpu(&o, &cc);
        const double tol = 4.0 * sqrt(g.std_error * g.std_error + cr.std_error * cr.std_error);
        check_close(g.price, cr.price, tol, "arithmetic asian: gpu-shuffle vs cpu");
        const OptionParams geo = option(OPT_CALL, STYLE_ASIAN_GEO);
        check(g.price > geometric_asian(&geo, 52), "arithmetic asian call > geometric asian call (AM-GM)");
    }

    /* Odd path count with antithetic pairs; problem smaller than one block;
     * a different launch configuration. */
    o = option(OPT_CALL, STYLE_EUROPEAN);
    c = config(1001, 10, SCHEME_LOG, 1);
    {
        const MCResult r = price_gpu(GPU_SHUFFLE, &o, &c);
        check(r.samples == 501 && r.paths == 1002, "odd path count rounds up to whole antithetic pairs");
        check(isfinite(r.price) && r.price > 0, "tiny problem gives a finite price");
    }
    c = config(N, 16, SCHEME_LOG, 1);
    c.block_size = 64;
    c.blocks_per_sm = 3;
    check_mc(price_gpu(GPU_SHUFFLE, &o, &c), black_scholes(&o), "gpu-shuffle with 64-thread blocks, 3 blocks/SM",
             0.0);
    c.block_size = 32;
    check_mc(price_gpu(GPU_SHARED, &o, &c), black_scholes(&o), "gpu-shared with 32-thread blocks", 0.0);
}
#endif

int main(void) {
    test_closed_forms();
    test_cpu();
#ifndef MC_NO_CUDA
    test_gpu();
#endif
    printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
