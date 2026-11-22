/* Closed-form reference prices. */
#ifndef MC_BLACK_SCHOLES_H
#define MC_BLACK_SCHOLES_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

double norm_cdf(double x);

/* Black-Scholes price of a European call/put. */
double black_scholes(const OptionParams* opt);

/* Exact price of a discretely monitored geometric-average Asian option with
 * `fixings` equally spaced observations at T/n, 2T/n, ..., T (S0 excluded).
 * With fixings == 1 this reduces to Black-Scholes. */
double geometric_asian(const OptionParams* opt, int fixings);

/* Reference price if one exists (Black-Scholes for European, closed form for
 * geometric Asian). Returns 0 for arithmetic Asian, which has none. */
int reference_price(const OptionParams* opt, int steps, double* price);

#ifdef __cplusplus
}
#endif

#endif
