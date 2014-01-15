/** Augmented Lagrangian method for LPs (and linearly constrained LS
 ** at some point)
 **
 ** Solves
 **
 **   min_x linear'x
 ** subject to
 **   constraints x = rhs (1)
 **   lower <= x <= upper
 **
 ** with an augmented Lagrangian relaxation of (1).
 **/
#ifndef ALM_H
#define ALM_H
#include <stddef.h>
#include <stdio.h>
#include "spmv/spmv.h"
#include "approx/approx.h"
#include "thread_pool/thread_pool.h"

typedef struct alm * alm_t;
/* Allocate an alm instance. Everything is copied, except the
 * constraint matrix, which must remain live as long as the alm
 * instance is in use.
 *
 * constraints is an nrhs * nvars sparse matrix for the linear
 * equality constraint (1).  rhs is a an array of nrhs doubles.
 *
 * linear is the objective vector (an array of nvars doubles), or 0 if
 * NULL.
 *
 * lower and upper are arrays of nvars doubles (upper and lower bounds
 * on x), or -HUGE_VAL and HUGE_VAL if NULL.
 *
 * lambda_{lower,upper} are the same for the dual (Lagrange)
 * multipliers for constraint (1).  They can be left NULL (i.e.,
 * unconstrained) without changing the optimal solution, but explicit
 * ranges (e.g., for constraints that are actually inequalities with
 * explicit slack variables) can't hurt.
 *
 * Will eventually return NULL on error.
 */
alm_t alm_make(sparse_matrix_t * constraints,
               size_t nrhs, const double * rhs,
               size_t nvars, const double * linear,
               const double * lower, const double * upper,
               const double * lambda_lower, const double * lambda_upper);
/* Releases an alm instance and its internal storage; the sparse
 * matrix is left unaffected.
 *
 * Returns 0 on success.
 */
int alm_free(alm_t);

/* Accessors for an alm instance. The right-hand side and linear
 * objective vectors can be modified, as can the lower/upper bound
 * vectors (both primal and dual)
 */
sparse_matrix_t * alm_matrix(alm_t);
size_t alm_nrhs(alm_t);
double * alm_rhs(alm_t);
size_t alm_nvars(alm_t);
double * alm_linear(alm_t);
double * alm_lower(alm_t);
double * alm_upper(alm_t);
double * alm_lambda_lower(alm_t);
double * alm_lambda_upper(alm_t);

/* Solve an alm instance for up to niter iterations, starting from
 * primal solution x and dual multipliers lambda.
 *
 * x is an array of nvars doubles, and lambda an array of nconstraints
 * doubles.  They can be initialised to anything, e.g. zero, but good
 * initial solutions help.
 *
 * Logging output is written to log if non-NULL, from both the inner
 * APPROX solver and the outer augmented Lagrangian loop.
 *
 * On exit, OUT_diagnosis is filled with information if non-NULL:
 *   OUT_diagnosis[0]: the objective value of the primal solution
 *   OUT_diagnosis[1]: maximal contraint violation for the solution
 *   OUT_diagnosis[2]: norm of the finaly projected gradient for the
 *     primal solution, in the underlying APPROX solver.
 *
 * Finally, pool is either a thread pool for parallelisation, or NULL.
 *
 * The search stops if the maximal constraint violation is < 1e-5, and
 * the norm of the projected gradient is < 1e-6 (or APPROX stalls
 * without reducing that norm), or when it reaches the iteration limit.
 *
 * Returns 0 if the search stops with an almost feasible solution, and
 * 1 if it reached the iteration limit.
 *
 * Logging output is:
 *  [iteration #]: [max violation] [2-norm violation]  \
 *    [2-norm of projected gradient] [Lagrangian (lower) bound] \
 *    [primal objective value]
 */
int alm_solve(alm_t, size_t niter,
              double * x, size_t nvars,
              double * lambda, size_t nconstraints,
              FILE * log, double * OUT_diagnosis,
              thread_pool_t * pool);

/* Read a plain text LP instance
 *
 * First, a sparse matrix (see spmv.h), followed by:
 *  1. nrhs doubles for the rhs vector;
 *  2. nvars doubles for the linear objective vector;
 *  3. nvars doubles for the lower bound ("-inf" for none)
 *  4. nvars doubles for the upper bound ("inf" for none)
 *  5. nrhs doubles for the dual lower bound
 *  6. nrhs doubles for the dual upper bound
 *
 * Will eventually return NULL on error.
 */
alm_t alm_read(FILE * stream);
#endif
