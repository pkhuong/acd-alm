/** Accelerated parallel proximal coordinate descent for bound-constrained
 ** least squares of the form:
 **
 **  min_x \sum_i weight_i/2 |constraints_i x - b_i|^2 + linear'x
 **   subject to lower <= x <= upper
 **
 **/
#ifndef APPROX_H
#define APPROX_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "../spmv/spmv.h"
#include "../thread_pool/thread_pool.h"

typedef struct approx * approx_t;

/* Create an approx instance.
 *
 * All values are copied, except for the first argument, which must
 * remain live as long as the approx_t object is used.
 *
 * constraints, rhs and weight define the least square component of
 * the objective function: it's 1/2|constraints x - b|_2^2, with a
 * row-by-row reweighting.
 *
 * If weight is NULL, then it's a vector of 1.
 *
 * nrhs is only used for error checking (rhs and weight are arrays of
 * nrhs doubles).
 *
 * linear is the linear term in the objective function (a vector of
 * nvars doubles), or NULL (zero).
 *
 * lower and upper are vectors of nvars lower and upper bounds for x;
 * if NULL, they default to -HUGE_VAL and HUGE_VAL (i.e., no bound
 * constraint).
 *
 * Will one day return NULL on error.
 */
approx_t approx_make(sparse_matrix_t * constraints, /* Must remain alive */
                     size_t nrhs, const double * rhs, const double * weight,
                     size_t nvars,
                     const double * linear,
                     const double * lower, const double * upper);

/* Free an approx object. The constraints sparse matrix is *not*
 * freed.
 *
 * Safe to call on NULL.
 *
 * Returns 0 on success.
 */
int approx_free(approx_t);

/* Accessors for the approx object's internal vectors.
 */
sparse_matrix_t * approx_matrix(approx_t);
size_t approx_nrhs(approx_t);
double * approx_rhs(approx_t);
double * approx_weight(approx_t);
size_t approx_nvars(approx_t);
double * approx_linear(approx_t);
double * approx_lower(approx_t);
double * approx_upper(approx_t);

/* This must be called after any modification to the objective function
 * (i.e. matrix, weight, or linear; changes to rhs are OK).
 *
 * Returns 0 on success.
 */
int approx_update_step_sizes(approx_t);

/* Solve an approx instance, starting from an the initial solution x.
 *
 * x is an array of n doubles, and the approx_t instance is minimised,
 * starting from x, for up to niter iterations.  The best solution is
 * written to x on exit.
 *
 * The solver also stops whenever:
 *
 *  - the 2-norm of the projected gradient is < max_pg;
 *  - the objective value is < max_value;
 *  - the solution has moved (2-norm) by less than min_delta in the
 *    last 100 iterations (relative to the 2-norm of the solution +
 *    1e-10).
 *
 * Progress is logged to log if non-NULL, every period iterations.
 *
 * if OUT_diagnosis is non-NULL, it is overwritten on exit with:
 *   OUT_diagnosis[0]: objective value;
 *   OUT_diagnosis[1]: norm of the gradient;
 *   OUT_diagnosis[2]: norm of the projected gradient;
 *   OUT_diagnosis[3]: difference with the solution at the latest
 *                      iteration # that's a multiple of 100;
 *   OUT_diagnosis[4]: number of iterations.
 *
 * For convenience purposes, offset is a constant increment to the
 * objective value. i.e., k in min_x |Ax-b|_2^2 + cx + k.
 *
 * Finally, pool is either a thread pool handle, for internal
 * parallelisation, or NULL for serial execution.
 *
 * Logging takes the form:
 *
 *  [Iteration #] [objective value] [2-norm gradient]   \
 *    [2-norm projected gradient] [step size multiplier] \
 *    [relative distance from solution 100 iter earlier]
 *
 * This is printed to log on the first and last iteration, and every
 * period iteration if period is non-zero.
 */
int approx_solve(double * x, size_t n, approx_t approx, size_t niter,
                 double max_pg, double max_value, double min_delta,
                 FILE * log, size_t period,
                 double * OUT_diagnosis /* NULL or double[5] */,
                 double offset, thread_pool_t * pool);
#endif
