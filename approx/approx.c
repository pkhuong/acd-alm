#include "approx.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include "../huge_alloc/huge_alloc.h"
#include "../thread_pool/thread_pool.h"

struct approx {
        size_t nrhs, nvars;
        sparse_matrix_t * matrix;
        double * rhs;
        double * weight; /* \sum_i weight_i [(Ax-b)_i]^2 */
        double * linear; /* linear x + [LS] */

        double * lower, * upper; /* box */

        uint32_t * beta;
        double * v;
        double * inv_v;

        approx_t * permuted;
};

static double * copy_double(const double * x, size_t n)
{
        double * out = huge_calloc(n, sizeof(double));
        memcpy(out, x, sizeof(double)*n);
        return out;
}

static double * copy_double_default(const double * x, size_t n, double missing)
{
        double * out = huge_calloc(n, sizeof(double));
        if (x != NULL) {
                memcpy(out, x, sizeof(double)*n);
        } else {
                for (size_t i = 0; i < n; i++)
                        out[i] = missing;
        }
        return out;
}

static approx_t * approx_make_1(sparse_matrix_t * constraints,
                                size_t nrhs, const double * rhs,
                                const double * weight,
                                size_t nvars,
                                const double * linear,
                                const double * lower, const double * upper)
{
        assert(nrhs == sparse_matrix_nrows(constraints));
        assert(nvars == sparse_matrix_ncolumns(constraints));

        approx_t * approx = calloc(1, sizeof(approx_t));
        approx->nrhs = nrhs;
        approx->nvars = nvars;
        approx->matrix = constraints;
        approx->rhs = copy_double(rhs, nrhs);
        approx->weight = copy_double_default(weight, nrhs, 1);
        approx->linear = copy_double_default(linear, nvars, 0);
        approx->lower = copy_double_default(lower, nvars, -HUGE_VAL);
        approx->upper = copy_double_default(upper, nvars, HUGE_VAL);

        approx->beta = huge_calloc(nrhs, sizeof(uint32_t));
        approx->v = huge_calloc(nvars, sizeof(double));
        approx->inv_v = huge_calloc(nvars, sizeof(double));

        return approx;
}

approx_t * approx_make(sparse_matrix_t * constraints,
                       size_t nrhs, const double * rhs,
                       const double * weight,
                       size_t nvars,
                       const double * linear,
                       const double * lower, const double * upper)
{
        approx_t * approx = approx_make_1(constraints,
                                          nrhs, rhs,
                                          weight,
                                          nvars, linear, lower, upper);

        approx->permuted = approx_make_1(sparse_matrix_copy(constraints,
                                                            1),
                                         nrhs, rhs,
                                         weight,
                                         nvars, linear, lower, upper);

        approx_update(approx);
        return approx;
}

#define DEF(TYPE, FIELD)                                        \
        TYPE approx_##FIELD(approx_t * approx)                  \
        {                                                       \
                return approx->FIELD;                           \
        }

DEF(sparse_matrix_t *, matrix)
DEF(size_t, nrhs)
DEF(double *, rhs)
DEF(double *, weight)
DEF(size_t, nvars)
DEF(double *, linear)
DEF(double *, lower)
DEF(double *, upper)

#undef DEF

int approx_free(approx_t * approx)
{
        if (approx == NULL) return 0;

        if (approx->permuted != NULL) {
                sparse_matrix_t * permuted = approx->permuted->matrix;
                approx_free(approx->permuted);
                sparse_matrix_free(permuted);
        }

        huge_free(approx->rhs);
        huge_free(approx->weight);
        huge_free(approx->linear);
        huge_free(approx->lower);
        huge_free(approx->upper);
        huge_free(approx->beta);
        huge_free(approx->v);
        huge_free(approx->inv_v);
        memset(approx, 0, sizeof(approx_t));
        free(approx);

        return 0;
}

static int approx_update_step_size(approx_t * approx)
{
        assert(approx->nrhs == sparse_matrix_nrows(approx->matrix));
        assert(approx->nvars == sparse_matrix_ncolumns(approx->matrix));

        uint32_t * beta = approx->beta;
        double * v = approx->v;
        const double * weight = approx->weight;
        memset(beta, 0, approx->nrhs*sizeof(uint32_t));
        memset(v, 0, approx->nvars*sizeof(double));

        sparse_matrix_t * matrix = approx->matrix;
        size_t nnz = sparse_matrix_nnz(matrix);
        const uint32_t * rows = sparse_matrix_rows(matrix),
                * columns = sparse_matrix_columns(matrix);
        const double * values = sparse_matrix_values(matrix);

        for (size_t i = 0; i < nnz; i++)
                beta[rows[i]]++;

        /* for 1/2 |Ax-b|^2 : just add beta_j (A_ji)^2 */
        /* for 1/2 |weight (*) (Ax-b)|^2: weight_j *(A_ij)^2 */
        for (size_t i = 0; i < nnz; i++) {
                uint32_t row = rows[i];
                double value = values[i];
                double w = weight[row];
                v[columns[i]] += w*beta[row]*value*value;
        }

        size_t nvars = approx->nvars;
        double * inv_v = approx->inv_v;
        const double * c = approx->linear;
        for (size_t i = 0; i < nvars; i++) {
                double vi = v[i];
                /* avoid 0*inf -> nan: if variable appears nowhere in
                 * the obj fun, directional gradient = 0. Always leave it
                 * in place by letting inv_v[i] = 1. */
                if ((v == 0) && (c[i] == 0))
                        inv_v[i] = 1;
                else    inv_v[i] = 1.0/vi;
        }

        return 0;
}

int approx_update(approx_t * approx)
{
        assert(!approx_update_step_size(approx));

        if (approx->permuted) {
#define PERMUTE(FIELD, LENGTH)                                          \
                assert(!sparse_matrix_row_permute(approx->permuted->matrix, \
                                                  approx->permuted->FIELD, \
                                                  approx->LENGTH,       \
                                                  approx->FIELD, 1))    \

                PERMUTE(rhs, nrhs);
                PERMUTE(weight, nrhs);
#undef PERMUTE
#define PERMUTE(FIELD, LENGTH)                                          \
                assert(!sparse_matrix_col_permute(approx->permuted->matrix, \
                                                  approx->permuted->FIELD, \
                                                  approx->LENGTH,       \
                                                  approx->FIELD, 1))    \

                PERMUTE(linear, nvars);
                PERMUTE(lower, nvars);
                PERMUTE(upper, nvars);
#undef PERMUTE
                assert(!approx_update_step_size(approx->permuted));
        }

        return 0;
}

#include "vector.inc"
#include "gradient_value.inc"
#include "step_project.inc"

static double dot_diff(const struct vector * gv,
                       const struct vector * zv, const struct vector * zpv)
{
        size_t n = gv->n;
        assert(zv->n == n);
        assert(zpv->n == n);

        const v2d * g = (v2d*)gv->x,
                * z = (v2d*)zv->x, * zp = (v2d*)zpv->x;

        v2d acc = {0,0};
        size_t vector_n = (n+1)/2;
        for (size_t i = 0; i < vector_n; i++)
                acc += g[i]*(zp[i]-z[i]);

        return acc[0]+acc[1];
}

static double project_gradient_norm(const struct vector * gv,
                                    const struct vector * xv,
                                    const double * lower, const double * upper)
{
        size_t n = gv->n;
        assert(xv->n == n);
        const v2d * g = (v2d*)gv->x, * x = (v2d*)xv->x,
                * l = (v2d*)lower, * u = (v2d*)upper;

        v2d acc = {0, 0};
        size_t vector_n = (n+1)/2;
        for (size_t i = 0; i < vector_n; i++) {
                v2d xi = x[i];
                v2d xp = xi-g[i];
                xp = __builtin_ia32_maxpd(l[i], xp);
                xp = __builtin_ia32_minpd(u[i], xp);
                v2d delta = xi-xp;
                acc += delta*delta;
        }
        return sqrt(acc[0]+acc[1]);
}

struct approx_state
{
        struct vector y;
        struct vector z;
        struct vector zp;
        struct vector x;

        size_t iteration;
        double theta;

        struct vector g, g2;

        double step_length;
};

static void init_state(struct approx_state * state,
                       size_t nvars, size_t nrows)
{
        init_vector(&state->y, nvars, nrows);
        init_vector(&state->z, nvars, nrows);
        init_vector(&state->zp, nvars, nrows);
        init_vector(&state->x, nvars, nrows);

        state->iteration = 0;
        state->theta = 1;

        init_vector(&state->g, nvars, 0);
        init_vector(&state->g2, nvars, 0);

        state->step_length = 1;
}

static void destroy_state(struct approx_state * state)
{
        destroy_vector(&state->y);
        destroy_vector(&state->z);
        destroy_vector(&state->zp);
        destroy_vector(&state->x);
        destroy_vector(&state->g);
        destroy_vector(&state->g2);

        memset(state, 0, sizeof(struct approx_state));
}

static double next_theta(struct approx_state * state)
{
#ifndef COMPLICATED_THETA
        size_t k = ++state->iteration;
        return state->theta = 2.0/(k+2);
#else
        double theta = state->theta;
        double theta2 = theta*theta,
                theta4 = theta2*theta2;
        return state->theta = .5*(sqrt(theta4 + 4*theta2)-theta2);
#endif
}

/* Assumption: y = linterp(y, theta, x, z); (only violation)
 */
static int try_long_step(approx_t * approx, struct approx_state * state,
                         double step_length, thread_pool_t * pool)
{
        int safe = (step_length <= 1+1e-6);
        if (safe) step_length = 1;

        double expected_improvement;
        {
                assert(state->z.violationp);
                assert(state->y.violationp);
                struct vector * g[2] = {&state->g, &state->g2};
                struct vector * x[2]= {&state->z, &state->y};
                expected_improvement
                        = gradient2_long_step(approx, pool,
                                              g, x,
                                              &state->zp,
                                              state->theta, step_length,
                                              approx->lower, approx->upper,
                                              approx->inv_v, approx->v);
        }
        double initial = compute_value(approx, &state->z, pool);
        double now = compute_value(approx, &state->zp, pool);
        assert(expected_improvement <= 0);

        if (now <= initial+expected_improvement) {
                state->step_length = step_length*1.01;
                return 1;
        }

        state->step_length = .9*step_length;
        if (!safe)
                step(&state->zp, state->theta,
                     &state->g2, &state->z,
                     approx->lower, approx->upper,
                     approx->inv_v);

        return 0;
}

static int short_step(approx_t * approx, struct approx_state * state,
                      double step_length, thread_pool_t * pool)
{
        {
                assert(state->z.violationp);
                assert(state->y.violationp);
                /* FIXME: state->g not always needed! */
                struct vector * g[2] = {&state->g, &state->g2};
                struct vector * x[2]= {&state->z, &state->y};
                gradient2(g, approx, x, pool);
        }

        step(&state->zp, state->theta,
             &state->g2, &state->z,
             approx->lower, approx->upper,
             approx->inv_v);

        state->step_length = step_length * 1.01;
        if (state->step_length > 1)
                state->step_length = 1+1e-6;

        return 0;
}

static const struct vector *
iter(approx_t * approx, struct approx_state * state, double * OUT_pg,
     thread_pool_t * pool)
{
        int descent_achieved;
        {
                double step_length = state->step_length;
#ifdef STATIC_STEP
                step_length = state->step_length = 1;
#endif
                if (step_length <= 1)
                        descent_achieved = short_step(approx, state,
                                                      step_length, pool);
                else    descent_achieved = try_long_step(approx, state,
                                                         step_length, pool);
        }

        if (OUT_pg != NULL)
                *OUT_pg = project_gradient_norm(&state->g, &state->z,
                                                approx->lower, approx->upper);

        if ((!descent_achieved) /* Value improvement OK */
            && (dot_diff(&state->g, &state->z, &state->zp) > 0)) {
                /* Oscillation */
                copy_vector(&state->x, &state->z);
                copy_vector(&state->y, &state->z);
                state->iteration = 0;
                state->theta = 1;
                return &state->x;
        }

        {
#ifndef NO_CACHING
                if (!state->zp.violationp)
#endif
                        compute_violation(&state->zp, approx, pool);
                double theta = state->theta;
                double next = next_theta(state);
                linterp_xy(&state->y, &state->x, &state->zp,
                           theta, next, pool);
        }
        {
                /* swap */
                struct vector temp = state->z;
                state->z = state->zp;
                state->zp = temp;
        }

        return &state->zp;
}

static double diff(const double * x, const double * y, size_t n)
{
        v2d acc = {0,0};
        const v2d * x2 = (v2d*)x, * y2 = (v2d*)y;
        size_t vector_n = (n+1)/2;
        for (size_t i = 0; i < vector_n; i++) {
                v2d delta = x2[i]-y2[i];
                acc += delta*delta;
        }
        return sqrt(acc[0]+acc[1]);
}

static double norm_2(const struct vector * xv)
{
        size_t n = (xv->n+1)/2;
        const v2d * x = (v2d*)xv->x;
        v2d acc = {0, 0};
        for (size_t i = 0; i < n; i++) {
                v2d xi = x[i];
                acc += xi*xi;
        }
        return sqrt(acc[0]+acc[1]);
}

static void print_log(FILE * log, size_t k,
                      double value, double ng, double pg,
                      double step, double diff)
{
        if (log == NULL) return;

        if (diff < HUGE_VAL)
                fprintf(log, "\t%10zu %12g %12g %12g %8g %12g\n",
                        k, value, ng, pg, step, diff);
        else   fprintf(log, "\t%10zu %12g %12g %12g %8g\n",
                       k, value, ng, pg, step);
}

int approx_solve(double * x, size_t n, approx_t * approx, size_t niter,
                 double max_pg, double max_value, double min_delta,
                 FILE * log, size_t period, double * OUT_diagnosis,
                 double offset, thread_pool_t * pool)
{
        assert(n == approx->nvars);

        if (approx->permuted)
                approx = approx->permuted;

        struct approx_state state;
        init_state(&state, approx->nvars, approx->nrhs);

        sparse_matrix_col_permute(approx->matrix, state.x.x, n,
                                  x, 1);
        project(&state.x, approx->lower, approx->upper);
        compute_violation(&state.x, approx, pool);
        compute_value(approx, &state.x, pool);
        copy_vector(&state.z, &state.x);
        copy_vector(&state.y, &state.x);
        double * prev_x = huge_calloc(n, sizeof(double));
        memcpy(prev_x, state.x.x, n*sizeof(double));

        const struct vector * center = &state.x;
        double value = compute_value(approx, (struct vector *)center, pool)+offset;
        double ng = HUGE_VAL, pg = HUGE_VAL;
        double delta = HUGE_VAL;
        size_t i;
        int restart = 0, reason = 0;
        for (i = 0; i < niter; i++) {
                delta = HUGE_VAL;
                pg = HUGE_VAL;
                {
                        double * pgp = NULL;
                        if ((i == 0) || ((i+1)%10 == 0))
                                pgp = &pg;
                        center = iter(approx, &state, pgp, pool);
                }
                if (center == &state.x) {
                        if (!restart) {
                                restart = 1;
                                if (log != NULL)
                                        fprintf(log, "\t\t ");
                        }
                        if (log != NULL) {
                                fprintf(log, "R");
                                fflush(log);
                        }
                }

                value = compute_value(approx, (struct vector *)center, pool)+offset;
                if (value < max_value) {
                        reason = 1;
                        break;
                }

                if (pg < max_pg) {
                        reason = 2;
                        break;
                }

                if ((i+1)%100 == 0) {
                        center = &state.x;
                        value = compute_value(approx, &state.x, pool);
                        gradient(&state.g, approx, &state.x, pool);
                        pg = project_gradient_norm(&state.g, &state.x,
                                                   approx->lower,
                                                   approx->upper);
                        delta = (diff(prev_x, state.x.x, n)
                                 /(norm_2(&state.x)+1e-10));
                        if (value < max_value) {
                                reason = 1;
                                break;
                        }
                        if (pg < max_pg) {
                                reason = 2;
                                break;
                        }
                        if (delta < min_delta) {
                                reason = 3;
                                break;
                        }
                        memcpy(prev_x, state.x.x, n*sizeof(double));
                        compute_violation(&state.x, approx, pool);
                }
                if ((i == 0) || (period && ((i+1)%period == 0))) {
                        if (restart) {
                                restart = 0;
                                printf("\n");
                        }
                        ng = norm_2(&state.g);
                        print_log(log, i+1, value, ng, pg,
                                  state.step_length, delta);
                }
        }
        if (restart) {
                restart = 0;
                printf("\n");
        }

        delta = diff(prev_x, center->x, n)/(norm_2(center)+1e-10);
        value = compute_value(approx, (struct vector*)center, pool)
                + offset;
        gradient(&state.g, approx, (struct vector *)center, pool);
        ng = norm_2(&state.g);
        /* We're about to free center, anyway */
        pg = project_gradient_norm(&state.g, center,
                                   approx->lower, approx->upper);

        print_log(log, i+1, value, ng, pg,
                  state.step_length, delta);

        sparse_matrix_col_permute(approx->matrix, x, n,
                                  center->x, -1);

        if (OUT_diagnosis != NULL) {
                OUT_diagnosis[0] = value;
                OUT_diagnosis[1] = ng;
                OUT_diagnosis[2] = pg;
                OUT_diagnosis[3] = delta;
                OUT_diagnosis[4] = i+1;
        }

        huge_free(prev_x);
        destroy_state(&state);

        return reason;
}

#ifdef TEST_APPROX
sparse_matrix_t * random_matrix(size_t nrows, size_t ncolumns)
{
        size_t total = nrows*ncolumns;
        size_t nnz = 0;
        uint32_t * columns = calloc(total, sizeof(uint32_t));
        uint32_t * rows = calloc(total, sizeof(uint32_t));
        double * values = calloc(total, sizeof(double));

        for (size_t row = 0; row < nrows; row++) {
                for (size_t column = 0; column < ncolumns; column++) {
                        if (((1.0*random()/RAND_MAX) < .5)
                            && (row != column))
                                continue;
                        columns[nnz] = column;
                        rows[nnz] = row;
                        values[nnz] = (2.0*random()/RAND_MAX) -1;
                        nnz++;
                }
        }

        sparse_matrix_t * m = sparse_matrix_make(ncolumns, nrows, nnz,
                                                 rows, columns, values,
                                                 0);
        free(values);
        free(rows);
        free(columns);

        return m;
}

void random_vector(double * vector, size_t n)
{
        for (size_t i = 0; i < n; i++)
                vector[i] = ((2.0*random())/RAND_MAX)-1;
}

void test_1(size_t nrows, size_t ncolumns)
{
        sparse_matrix_t * m = random_matrix(nrows, ncolumns);
        double * solution = calloc(ncolumns, sizeof(double));
        random_vector(solution, ncolumns);
        double * rhs = calloc(nrows, sizeof(double));
        double * x = calloc(ncolumns, sizeof(double));

        assert(0 == sparse_matrix_multiply(rhs, nrows,
                                           m, solution, ncolumns, 0,
                                           NULL));

        approx_t * a = approx_make(m, nrows, rhs, NULL, ncolumns,
                                 NULL, NULL, NULL);
        double diagnosis[5];
        int r = approx_solve(x, ncolumns, a, -1U,
                             0, 1e-13, 0,
                             stdout, 10000, diagnosis, 0,
                             NULL);

        assert(r > 0);

        double * residual = calloc(nrows, sizeof(double));
        assert(0 == sparse_matrix_multiply(residual, nrows,
                                           m, x, ncolumns, 0,
                                           NULL));
        double d = diff(rhs, residual, nrows);
        printf("r: %.18f %.18f %p\n", diagnosis[0], d, x);

        assert(d < 1e-6);

        free(residual);
        approx_free(a);
        free(x);
        free(rhs);
        free(solution);
        sparse_matrix_free(m);
}

int main ()
{
        sparse_matrix_init();
        for (size_t i = 10; i <= 20; i++) {
                for (size_t j = 10; j <= 20; j++) {
                        printf("%zu %zu\n", i, j);
                        test_1(i, j);
                }
        }

        return 0;
}
#endif
