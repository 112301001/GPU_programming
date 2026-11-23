/* Command-line driver: prices one option with each requested engine and prints
 * a comparison table, or CSV rows for the benchmark scripts. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "black_scholes.h"
#include "common.h"
#include "cpu_mc.h"
#ifndef MC_NO_CUDA
#include "gpu_mc.h"
#endif

/* Engine ids: the CPU baselines, then one per GpuEngine (same order). */
enum { ENG_CPU1 = 0, ENG_CPU, ENG_GPU0, ENG_COUNT = ENG_GPU0 + 7 };
static const char* kEngineNames[ENG_COUNT] = {"cpu1",       "cpu",        "gpu-hostrng", "gpu-pregen", "gpu-atomic",
                                              "gpu-shared", "gpu-shuffle", "gpu-xorwow", "gpu-double"};
#define DEFAULT_ENGINES "cpu,gpu-pregen,gpu-atomic,gpu-shared,gpu-shuffle"

#define MAX_ENGINES 32

typedef struct {
    OptionParams opt;
    SimConfig cfg;
    int engines[MAX_ENGINES];
    int n_engines;
    int repeat;
    int csv;
    int csv_header;
    int warmup;
    int kernel_info;
} Options;

static void usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "\n"
           "Option contract:\n"
           "  --S0 X  --K X  --r X  --sigma X  --T X     (defaults 100 100 0.05 0.2 1)\n"
           "  --type call|put                            (default call)\n"
           "  --style european|asian|geo-asian           (default european)\n"
           "\n"
           "Simulation:\n"
           "  --paths N        simulated paths, e.g. 1e7   (default 16777216)\n"
           "  --steps M        time steps per path         (default 252)\n"
           "  --scheme euler|log  Euler-Maruyama on S, or on log S (exact for GBM) (default euler)\n"
           "  --antithetic | --no-antithetic             (default on)\n"
           "  --seed N                                   (default 1234)\n"
           "\n"
           "Engines (--engines LIST, comma separated, or 'all'):\n"
           "  cpu1         single-core CPU\n"
           "  cpu          OpenMP CPU, all cores (or --cpu-threads N)\n"
           "  gpu-hostrng  normals generated on the host, copied over PCIe, read from global memory\n"
           "  gpu-pregen   normals pre-generated on the device into global memory\n"
           "  gpu-atomic   in-kernel Philox, one global atomicAdd per sample\n"
           "  gpu-shared   in-kernel Philox, shared-memory tree reduction per block\n"
           "  gpu-shuffle  in-kernel Philox, warp-shuffle reduction (the proposed engine)\n"
           "  gpu-xorwow   as gpu-shuffle with the XORWOW generator\n"
           "  gpu-double   as gpu-shuffle in double precision\n"
           "  default: " DEFAULT_ENGINES "\n"
           "\n"
           "Launch:\n"
           "  --block N          threads per block, power of two <= 256  (default 256)\n"
           "  --blocks-per-sm N  grid = N x number of SMs                (default 8)\n"
           "  --cpu-threads N    OpenMP threads for 'cpu' (0 = all cores) (default 0)\n"
           "\n"
           "Output:\n"
           "  --repeat N       timed runs per engine, median reported (default 3)\n"
           "  --csv            CSV rows instead of a table\n"
           "  --csv-header     print the CSV header and exit\n"
           "  --no-warmup      skip the GPU warm-up and the 300 ms clock spin-up per engine (use under ncu)\n"
           "  --kernel-info    registers / shared memory / occupancy of each kernel\n"
           "  --help\n",
           prog);
}

static void die(const char* msg, const char* arg) {
    fprintf(stderr, "error: %s%s%s (see --help)\n", msg, arg ? " " : "", arg ? arg : "");
    exit(EXIT_FAILURE);
}

static double parse_double(const char* s) {
    char* end = NULL;
    const double v = strtod(s, &end);
    if (end == s || *end != '\0') die("not a number:", s);
    return v;
}

/* Integers or scientific notation such as 1e7. */
static uint64_t parse_count(const char* s) {
    const double v = parse_double(s);
    if (v < 1.0) die("count must be >= 1:", s);
    return (uint64_t)llround(v);
}

static int engine_from_name(const char* name) {
    for (int i = 0; i < ENG_COUNT; ++i)
        if (strcmp(name, kEngineNames[i]) == 0) return i;
    die("unknown engine", name);
    return -1;
}

static void parse_engines(Options* o, const char* list) {
    o->n_engines = 0;
    if (strcmp(list, "all") == 0) {
        for (int i = 0; i < ENG_COUNT; ++i) o->engines[o->n_engines++] = i;
        return;
    }
    char buf[512];
    snprintf(buf, sizeof buf, "%s", list);
    for (char* tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        if (o->n_engines >= MAX_ENGINES) die("too many engines", NULL);
        o->engines[o->n_engines++] = engine_from_name(tok);
    }
    if (o->n_engines == 0) die("empty engine list", NULL);
}

static Options parse(int argc, char** argv) {
    Options o;
    memset(&o, 0, sizeof o);
    o.opt = default_option();
    o.cfg = default_config();
    o.repeat = 3;
    o.warmup = 1;
    parse_engines(&o, DEFAULT_ENGINES);

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
#define NEXT() (i + 1 < argc ? argv[++i] : (die("missing value for", a), (char*)NULL))
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            usage(argv[0]);
            exit(0);
        } else if (!strcmp(a, "--S0")) o.opt.S0 = parse_double(NEXT());
        else if (!strcmp(a, "--K")) o.opt.K = parse_double(NEXT());
        else if (!strcmp(a, "--r")) o.opt.r = parse_double(NEXT());
        else if (!strcmp(a, "--sigma")) o.opt.sigma = parse_double(NEXT());
        else if (!strcmp(a, "--T")) o.opt.T = parse_double(NEXT());
        else if (!strcmp(a, "--type")) {
            const char* v = NEXT();
            if (!strcmp(v, "call")) o.opt.type = OPT_CALL;
            else if (!strcmp(v, "put")) o.opt.type = OPT_PUT;
            else die("unknown --type", v);
        } else if (!strcmp(a, "--style")) {
            const char* v = NEXT();
            if (!strcmp(v, "european")) o.opt.style = STYLE_EUROPEAN;
            else if (!strcmp(v, "asian")) o.opt.style = STYLE_ASIAN_ARITH;
            else if (!strcmp(v, "geo-asian")) o.opt.style = STYLE_ASIAN_GEO;
            else die("unknown --style", v);
        } else if (!strcmp(a, "--scheme")) {
            const char* v = NEXT();
            if (!strcmp(v, "euler")) o.cfg.scheme = SCHEME_EULER;
            else if (!strcmp(v, "log")) o.cfg.scheme = SCHEME_LOG;
            else die("unknown --scheme", v);
        } else if (!strcmp(a, "--paths")) o.cfg.paths = parse_count(NEXT());
        else if (!strcmp(a, "--steps")) o.cfg.steps = (int)parse_count(NEXT());
        else if (!strcmp(a, "--antithetic")) o.cfg.antithetic = 1;
        else if (!strcmp(a, "--no-antithetic")) o.cfg.antithetic = 0;
        else if (!strcmp(a, "--seed")) o.cfg.seed = strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--engines")) parse_engines(&o, NEXT());
        else if (!strcmp(a, "--cpu-threads")) o.cfg.cpu_threads = atoi(NEXT());
        else if (!strcmp(a, "--block")) o.cfg.block_size = atoi(NEXT());
        else if (!strcmp(a, "--blocks-per-sm")) o.cfg.blocks_per_sm = atoi(NEXT());
        else if (!strcmp(a, "--repeat")) {
            o.repeat = atoi(NEXT());
            if (o.repeat < 1) o.repeat = 1;
        } else if (!strcmp(a, "--csv")) o.csv = 1;
        else if (!strcmp(a, "--csv-header")) o.csv_header = 1;
        else if (!strcmp(a, "--no-warmup")) o.warmup = 0;
        else if (!strcmp(a, "--kernel-info")) o.kernel_info = 1;
        else die("unknown argument", a);
#undef NEXT
    }
    if (o.opt.S0 <= 0 || o.opt.K <= 0 || o.opt.sigma <= 0 || o.opt.T <= 0)
        die("S0, K, sigma and T must be positive", NULL);
    return o;
}

static MCResult run_engine(int engine, const Options* o) {
    if (engine == ENG_CPU1) {
        SimConfig one = o->cfg;
        one.cpu_threads = 1;
        return price_cpu(&o->opt, &one);
    }
    if (engine == ENG_CPU) return price_cpu(&o->opt, &o->cfg);
#ifndef MC_NO_CUDA
    return price_gpu((GpuEngine)(engine - ENG_GPU0), &o->opt, &o->cfg);
#else
    die("this binary was built without CUDA; unavailable engine:", kEngineNames[engine]);
    return price_cpu(&o->opt, &o->cfg); /* unreachable */
#endif
}

static int threads_used(int engine, const Options* o) {
    if (engine == ENG_CPU1) return 1;
    if (engine == ENG_CPU) return cpu_thread_count(&o->cfg);
    return 0;
}

static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static const char* kCsvHeader =
    "engine,style,type,scheme,antithetic,paths,steps,block,blocks_per_sm,cpu_threads,price,std_error,"
    "reference,abs_error,time_ms,kernel_ms,mpaths_per_s,speedup_vs_cpu,speedup_vs_cpu1";

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    if (o.csv_header) {
        printf("%s\n", kCsvHeader);
        return 0;
    }

    int any_gpu = 0;
    for (int i = 0; i < o.n_engines; ++i) any_gpu |= o.engines[i] >= ENG_GPU0;

    double ref = 0.0;
    const int has_ref = reference_price(&o.opt, o.cfg.steps, &ref);

    if (!o.csv) {
        printf("Option   : %s %s  S0=%g K=%g r=%g sigma=%g T=%g\n", style_name(o.opt.style), type_name(o.opt.type),
               o.opt.S0, o.opt.K, o.opt.r, o.opt.sigma, o.opt.T);
        printf("Paths    : %llu  steps=%d  scheme=%s  antithetic=%s  seed=%llu\n", (unsigned long long)o.cfg.paths,
               o.cfg.steps, scheme_name(o.cfg.scheme), o.cfg.antithetic ? "on" : "off",
               (unsigned long long)o.cfg.seed);
        printf("CPU      : OpenMP, %d threads\n", cpu_thread_count(&o.cfg));
    }

#ifndef MC_NO_CUDA
    if (any_gpu) {
        if (!gpu_available()) die("no CUDA device available", NULL);
        if (!o.csv) {
            char info[256];
            int blocks, threads;
            gpu_device_info(info, sizeof info);
            gpu_launch_dims(&o.cfg, &blocks, &threads);
            printf("GPU      : %s\n", info);
            printf("Launch   : <<<%d, %d>>>  (%d blocks/SM)\n", blocks, threads, o.cfg.blocks_per_sm);
        }
        if (o.kernel_info) {
            printf("Kernels  :\n");
            gpu_kernel_report(&o.cfg);
        }
        if (o.warmup) gpu_warmup(&o.opt, &o.cfg);
    }
#else
    if (any_gpu) die("this binary was built without CUDA (make cpu); use --engines cpu,cpu1", NULL);
#endif

    if (!o.csv) {
        if (has_ref)
            printf("Reference: %.6f (%s)\n", ref,
                   o.opt.style == STYLE_EUROPEAN ? "Black-Scholes" : "geometric Asian closed form");
        else printf("Reference: none (arithmetic Asian has no closed form)\n");
        if (has_ref && o.cfg.scheme == SCHEME_EULER)
            printf("Note     : Euler on S has O(dt) bias; --scheme log is exact for GBM\n");
        printf("\n");
    }

    MCResult results[MAX_ENGINES];
    double* times = (double*)malloc((size_t)o.repeat * sizeof(double));
    double* ktimes = (double*)malloc((size_t)o.repeat * sizeof(double));
    for (int e = 0; e < o.n_engines; ++e) {
        MCResult last;
#ifndef MC_NO_CUDA
        /* bring the GPU to its boost clock before timing it (it idles during CPU engines) */
        if (o.warmup && o.engines[e] >= ENG_GPU0) gpu_spin_up(300.0);
#endif
        for (int k = 0; k < o.repeat; ++k) {
            last = run_engine(o.engines[e], &o);
            times[k] = last.time_ms;
            ktimes[k] = last.kernel_ms;
        }
        qsort(times, (size_t)o.repeat, sizeof(double), cmp_double);
        qsort(ktimes, (size_t)o.repeat, sizeof(double), cmp_double);
        last.time_ms = times[o.repeat / 2];
        last.kernel_ms = ktimes[o.repeat / 2];
        results[e] = last;
    }
    free(times);
    free(ktimes);

    double cpu_ms = 0.0, cpu1_ms = 0.0;
    for (int e = 0; e < o.n_engines; ++e) {
        if (o.engines[e] == ENG_CPU) cpu_ms = results[e].time_ms;
        if (o.engines[e] == ENG_CPU1) cpu1_ms = results[e].time_ms;
    }

    if (!o.csv) {
        printf("%-12s %11s %10s %10s %7s %10s %10s %9s %9s %9s\n", "engine", "price", "std_err", "err", "err/se",
               "total_ms", "kernel_ms", "Mpaths/s", "vs cpu", "vs cpu1");
        printf("-----------------------------------------------------------------------------------------------"
               "-----\n");
    }
    for (int e = 0; e < o.n_engines; ++e) {
        const MCResult* m = &results[e];
        const char* name = kEngineNames[o.engines[e]];
        const double err = has_ref ? m->price - ref : 0.0;
        const double mps = m->time_ms > 0 ? (double)m->paths / (m->time_ms * 1e3) : 0.0;
        const double sp = (cpu_ms > 0 && m->time_ms > 0) ? cpu_ms / m->time_ms : 0.0;
        const double sp1 = (cpu1_ms > 0 && m->time_ms > 0) ? cpu1_ms / m->time_ms : 0.0;
        if (o.csv) {
            char refs[32] = "", errs[32] = "", sps[32] = "", sp1s[32] = "";
            if (has_ref) {
                snprintf(refs, sizeof refs, "%.8f", ref);
                snprintf(errs, sizeof errs, "%.8f", fabs(err));
            }
            if (cpu_ms > 0) snprintf(sps, sizeof sps, "%.3f", sp);
            if (cpu1_ms > 0) snprintf(sp1s, sizeof sp1s, "%.3f", sp1);
            printf("%s,%s,%s,%s,%d,%llu,%d,%d,%d,%d,%.8f,%.8e,%s,%s,%.4f,%.4f,%.3f,%s,%s\n", name,
                   style_name(o.opt.style), type_name(o.opt.type), scheme_name(o.cfg.scheme), o.cfg.antithetic,
                   (unsigned long long)m->paths, o.cfg.steps, o.cfg.block_size, o.cfg.blocks_per_sm,
                   threads_used(o.engines[e], &o), m->price, m->std_error, refs, errs, m->time_ms, m->kernel_ms,
                   mps, sps, sp1s);
        } else {
            char errbuf[24] = "-", zbuf[24] = "-", spbuf[24] = "-", sp1buf[24] = "-";
            if (has_ref) {
                snprintf(errbuf, sizeof errbuf, "%+.2e", err);
                if (m->std_error > 0) snprintf(zbuf, sizeof zbuf, "%+.2f", err / m->std_error);
            }
            if (cpu_ms > 0) snprintf(spbuf, sizeof spbuf, "%.1fx", sp);
            if (cpu1_ms > 0) snprintf(sp1buf, sizeof sp1buf, "%.1fx", sp1);
            printf("%-12s %11.6f %10.3e %10s %7s %10.2f %10.2f %9.1f %9s %9s\n", name, m->price, m->std_error,
                   errbuf, zbuf, m->time_ms, m->kernel_ms, mps, spbuf, sp1buf);
        }
    }
    if (!o.csv) {
        printf("\n95%% CI of the last engine: %.6f +/- %.6f\n", results[o.n_engines - 1].price,
               1.96 * results[o.n_engines - 1].std_error);
        printf("(median of %d runs; total = RNG + simulation + reduction + copies, kernel = cudaEvent time)\n",
               o.repeat);
    }
    return 0;
}
