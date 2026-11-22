#include "black_scholes.h"

#include <math.h>

double norm_cdf(double x) { return 0.5 * erfc(-x / sqrt(2.0)); }

/* Price an option whose underlying at maturity is lognormal with log-mean mu
 * and log-variance v, discounted at r over T. */
static double lognormal_option(double mu, double v, const OptionParams* o) {
    const double disc = exp(-o->r * o->T);
    const double sd = sqrt(v);
    const double fwd = exp(mu + 0.5 * v); /* E[underlying] */
    if (sd <= 0.0) {
        const double payoff = o->type == OPT_CALL ? fmax(fwd - o->K, 0.0) : fmax(o->K - fwd, 0.0);
        return disc * payoff;
    }
    const double d1 = (mu - log(o->K) + v) / sd;
    const double d2 = d1 - sd;
    if (o->type == OPT_CALL) return disc * (fwd * norm_cdf(d1) - o->K * norm_cdf(d2));
    return disc * (o->K * norm_cdf(-d2) - fwd * norm_cdf(-d1));
}

double black_scholes(const OptionParams* o) {
    const double mu = log(o->S0) + (o->r - 0.5 * o->sigma * o->sigma) * o->T;
    const double v = o->sigma * o->sigma * o->T;
    return lognormal_option(mu, v, o);
}

double geometric_asian(const OptionParams* o, int n) {
    /* log G = log S0 + (r - sigma^2/2) T (n+1)/(2n) + (sigma/n) sum_i W(t_i)
     * Var[(1/n) sum_i W(t_i)] = T (n+1)(2n+1) / (6 n^2) */
    const double nn = (double)n;
    const double mu = log(o->S0) + (o->r - 0.5 * o->sigma * o->sigma) * o->T * (nn + 1.0) / (2.0 * nn);
    const double v = o->sigma * o->sigma * o->T * (nn + 1.0) * (2.0 * nn + 1.0) / (6.0 * nn * nn);
    return lognormal_option(mu, v, o);
}

int reference_price(const OptionParams* opt, int steps, double* price) {
    switch (opt->style) {
        case STYLE_EUROPEAN: *price = black_scholes(opt); return 1;
        case STYLE_ASIAN_GEO: *price = geometric_asian(opt, steps); return 1;
        default: return 0;
    }
}
