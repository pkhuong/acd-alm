#include "approx.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <strings.h>

struct approx {
        size_t nrhs, nvars;
        sparse_matrix_t matrix;
        double * rhs;
        double * weight; /* \sum_i weight_i [(Ax-b)_i]^2 */
        double * linear; /* linear x + [LS] */

        double * lower, * upper; /* box */

        uint32_t * beta;
        double * inv_v;
};

static double * copy_double(const double * x, size_t n)
{
        double * out = calloc(n, sizeof(double));
        memcpy(out, x, sizeof(double)*n);
        return out;
}

static double * copy_double_default(const double * x, size_t n, double missing)
{
        double * out = calloc(n, sizeof(double));
        if (x != NULL) {
                memcpy(out, x, sizeof(double)*n);
        } else {
                for (size_t i = 0; i < n; i++)
                        out[i] = missing;
        }
        return out;
}

approx_t approx_make(sparse_matrix_t constraints,
                     size_t nrhs, const double * rhs,
                     const double * weight,
                     size_t nvars,
                     const double * linear,
                     const double * lower, const double * upper)
{
        assert(nrhs == sparse_matrix_nrows(constraints));
        assert(nvars == sparse_matrix_ncolumns(constraints));

        approx_t approx = calloc(1, sizeof(struct approx));
        approx->nrhs = nrhs;
        approx->nvars = nvars;
        approx->matrix = constraints;
        approx->rhs = copy_double(rhs, nrhs);
        approx->weight = copy_double_default(weight, nrhs, 1);
        approx->linear = copy_double_default(linear, nvars, 0);
        approx->lower = copy_double_default(lower, nvars, -HUGE_VAL);
        approx->upper = copy_double_default(upper, nvars, HUGE_VAL);

        approx->beta = calloc(nrhs, sizeof(uint32_t));
        approx->inv_v = calloc(nvars, sizeof(double));

        approx_update_step_sizes(approx);

        return approx;
}

#define DEF(TYPE, FIELD)                                        \
        TYPE approx_##FIELD(approx_t approx)                    \
        {                                                       \
                return approx->FIELD;                           \
        }

DEF(sparse_matrix_t, matrix)
DEF(size_t, nrhs)
DEF(double *, rhs)
DEF(double *, weight)
DEF(size_t, nvars)
DEF(double *, linear)
DEF(double *, lower)
DEF(double *, upper)

#undef DEF

int approx_free(approx_t approx)
{
        free(approx->rhs);
        free(approx->weight);
        free(approx->linear);
        free(approx->lower);
        free(approx->upper);
        free(approx->beta);
        free(approx->inv_v);
        memset(approx, 0, sizeof(struct approx));
        free(approx);

        return 0;
}

int approx_update_step_sizes(approx_t approx)
{
        assert(approx->nrhs == sparse_matrix_nrows(approx->matrix));
        assert(approx->nvars == sparse_matrix_ncolumns(approx->matrix));

        uint32_t * beta = approx->beta;
        double * inv_v = approx->inv_v;
        const double * weight = approx->weight;
        memset(beta, 0, approx->nrhs*sizeof(uint32_t));
        memset(inv_v, 0, approx->nvars*sizeof(double));

        sparse_matrix_t matrix = approx->matrix;
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
                double v = values[i];
                inv_v[columns[i]] += weight[row]*beta[row]*v*v;
        }

        for (size_t i = 0; i < approx->nvars; i++)
                inv_v[i] = 1.0/inv_v[i];

        return 0;
}

static void extrapolate_y(double * OUT_y, size_t nvars, double theta,
                        const double * x, const double * z)
{
        double scale = 1-theta;
        for (size_t i = 0; i < nvars; i++)
                OUT_y[i] = scale*x[i]+z[i];
}

static double dot(const double * x, const double * y, size_t n)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++)
                acc += x[i]*y[i];
        return acc;
}

static void gradient(double * OUT_grad, size_t nvars,
                     double * OUT_violation, size_t nrows, /* scaled */
                     approx_t approx,
                     const double * x, double * OUT_value)
{
        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);
        assert(0 == sparse_matrix_multiply(OUT_violation, nrows,
                                           approx->matrix, x, nvars, 0));

        {
                const double * rhs = approx->rhs,
                        * weight = approx->weight;
                if (OUT_value != NULL) {
                        double value = 0;
                        for (size_t i = 0; i < nrows; i++) {
                                double v = OUT_violation[i] - rhs[i];
                                double w = weight[i];
                                value += .5*w*v*v;
                                OUT_violation[i] = v*w;
                        }
                        *OUT_value = value;
                } else {
                        for (size_t i = 0; i < nrows; i++) {
                                double v = OUT_violation[i] - rhs[i];
                                OUT_violation[i] = v*weight[i];
                        }
                }
        }

        assert(0 == sparse_matrix_multiply(OUT_grad, nvars,
                                           approx->matrix,
                                           OUT_violation, nrows,
                                           1));

        {
                const double * linear = approx->linear;
                for (size_t i = 0; i < nvars; i++)
                        OUT_grad[i] += linear[i];
        }

        if (OUT_value != NULL)
                *OUT_value += dot(approx->linear, x, nvars);
}

static inline double min(double x, double y)
{
        return (x<y)?x:y;
}

static inline double max(double x, double y)
{
        return (x>y)?x:y;
}

static void project(double * x, size_t n,
                    const double * lower, const double * upper)
{
        for (size_t i = 0; i < n; i++)
                x[i] = min(max(lower[i], x[i]), upper[i]);
}

static void step(double * zp, size_t n, double theta,
                 const double * g, const double * z,
                 const double * lower, const double * upper,
                 const double * inv_v)
{
        double inv_theta = 1/theta;
        for (size_t i = 0; i < n; i++) {
                double gi = g[i], zi = z[i],
                        li = lower[i], ui = upper[i],
                        inv_vi = inv_v[i];
                double step = inv_theta*inv_vi;

                if (step == HUGE_VAL) {
                        if (gi == 0) {
                                zp[i] = zi;
                        } else if (gi > 0) {
                                assert(li > -HUGE_VAL);
                                zp[i] = li;
                        } else {
                                assert(ui < HUGE_VAL);
                                zp[i] = ui;
                        }
                } else {
                        double trial = zi - step*gi;
                        zp[i] = min(max(li, trial), ui);
                }
        }
}

static void extrapolate_x(double * OUT_x, size_t n, double theta,
                          const double * y,
                          const double * z, const double * zp)
{
        for (size_t i = 0; i < n; i++)
                OUT_x[i] = y[i] + theta*(zp[i]-z[i]);
}

static double next_theta(double theta)
{
        double theta2 = theta*theta,
                theta4 = theta2*theta2;
        return .5*(sqrt(theta4 + 4*theta2)-theta2);
}

static double dot_diff(const double * g, const double * z, const double * zp,
                       size_t n)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++)
                acc += g[i]*(zp[i]-z[i]);

        return acc;
}

static double project_gradient_norm(const double * g, const double * x,
                                    size_t n,
                                    const double * lower, const double * upper)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++) {
                double xi = x[i];
                double xp = xi-g[i];
                xp = min(max(lower[i], xp), upper[i]);
                double delta = xi-xp;
                acc += delta*delta;
        }
        return sqrt(acc);
}

struct approx_state
{
        double * y;
        double * z;
        double * zp;
        double * x;

        double theta;

        double * g;
        double * violation;
        double value;
};

static void init_state(struct approx_state * state,
                       size_t nvars, size_t nrows)
{
        state->y = calloc(nvars, sizeof(double));
        state->z = calloc(nvars, sizeof(double));
        state->zp = calloc(nvars, sizeof(double));
        state->x = calloc(nvars, sizeof(double));

        state->theta = 1;

        state->g = calloc(nvars, sizeof(double));
        state->violation = calloc(nrows, sizeof(double));
        state->value = HUGE_VAL;
}

static void destroy_state(struct approx_state * state)
{
        free(state->y);
        free(state->z);
        free(state->zp);
        free(state->x);
        free(state->g);
        free(state->violation);

        memset(state, 0, sizeof(struct approx_state));
}

static double * iter(approx_t approx, struct approx_state * state,
                     double * OUT_pg)
{
        extrapolate_y(state->y, approx->nvars, state->theta,
                      state->x, state->z);
        gradient(state->g, approx->nvars,
                 state->violation, approx->nrhs,
                 approx, state->y, NULL);
        step(state->zp, approx->nvars, state->theta,
             state->g, state->z,
             approx->lower, approx->upper,
             approx->inv_v);

        gradient(state->g, approx->nvars, state->violation, approx->nrhs,
                 approx, state->z, &state->value);
        *OUT_pg = project_gradient_norm(state->g, state->z,
                                        approx->nvars,
                                        approx->lower, approx->upper);

        if (dot_diff(state->g, state->z, state->zp, approx->nvars) > 0) {
                /* Oscillation */
                size_t total = approx->nvars*sizeof(double);
                memcpy(state->x, state->z, total);
                state->theta = 1;
                return state->x;
        }

        extrapolate_x(state->x, approx->nvars, state->theta,
                      state->y,
                      state->z, state->zp);
        state->theta = next_theta(state->theta);
        {
                /* swap */
                double * temp = state->z;
                state->z = state->zp;
                state->zp = temp;
        }

        return state->zp;
}

static double diff(const double * x, const double * y, size_t n)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++) {
                double delta = x[i]-y[i];
                acc += delta*delta;
        }
        return sqrt(acc);
}

static double norm_2(const double * x, size_t n)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++) {
                double xi = x[i];
                acc += xi*xi;
        }
        return sqrt(acc);
}

static void print_log(FILE * log, size_t k,
                      double value, double ng, double pg, double diff)
{
        if (log == NULL) return;

        if (diff < HUGE_VAL)
                fprintf(log, "\t%10zu %12f %12f %12f %12f\n",
                        k, value, ng, pg, diff);
        else   fprintf(log, "\t%10zu %12f %12f %12f\n",
                       k, value, ng, pg);
}

double approx_solve(double * x, size_t n, approx_t approx, size_t niter,
                    double max_pg, double max_value, double min_delta,
                    FILE * log, size_t period)
{
        assert(n == approx->nvars);

        struct approx_state state;
        init_state(&state, approx->nvars, approx->nrhs);
        memcpy(state.x, x, n*sizeof(double));
        project(state.x, n, approx->lower, approx->upper);
        memcpy(state.z, state.x, n*sizeof(double));

        double * prev_x = calloc(n, sizeof(double));
        memcpy(prev_x, state.x, n*sizeof(double));

        const double * center = state.x;
        double value = state.value;
        double ng = HUGE_VAL, pg = HUGE_VAL;
        double delta = HUGE_VAL;
        size_t i;
        int restart = 0;
        for (i = 0; i < niter; i++) {
                delta = HUGE_VAL;
                center = iter(approx, &state, &pg);
                /* if (center == state.x) { */
                /*         if (!restart) { */
                /*                 restart = 1; */
                /*                 if (log != NULL) */
                /*                         fprintf(log, "\t\t"); */
                /*         } */
                /*         if (log != NULL) */
                /*                 fprintf(log, "R"); */
                /* } */
                ng = norm_2(state.g, n);
                value = state.value;
                if (pg < max_pg) break;
                if (value < max_value) break;

                if ((i+1)%100 == 0) {
                        center = state.x;
                        gradient(state.g, approx->nvars, state.violation,
                                 approx->nrhs,
                                 approx, state.x, &value);
                        pg = project_gradient_norm(state.g, state.x,
                                                   approx->nvars,
                                                   approx->lower, 
                                                   approx->upper);
                        delta = (diff(prev_x, state.x, n)
                                 /norm_2(state.x, n));
                        if (delta < min_delta)
                                break;
                        if (pg < max_pg) break;
                        if (value < max_value) break;
                        memcpy(prev_x, state.x, n*sizeof(double));
                }
                if ((i == 0) || ((i+1)%period == 0)) {
                        if (restart) {
                                restart = 0;
                                printf("\n");
                        }
                        print_log(log, i+1, value, ng, pg, delta);
                }
        }
        if (restart) {
                restart = 0;
                printf("\n");
        }
        print_log(log, i+1, value, ng, pg, delta);
        memcpy(x, center, n*sizeof(double));

        free(prev_x);
        destroy_state(&state);

        return value;
}

#ifdef TEST_APPROX
sparse_matrix_t random_matrix(size_t nrows, size_t ncolumns)
{
        size_t total = nrows*ncolumns;
        size_t nnz = 0;
        uint32_t * columns = calloc(total, sizeof(uint32_t));
        uint32_t * rows = calloc(total, sizeof(uint32_t));
        double * values = calloc(total, sizeof(double));
        
        for (size_t row = 0; row < nrows; row++) {
                for (size_t column = 0; column < ncolumns; column++) {
                        /* if (((1.0*random()/RAND_MAX) < .5) */
                        /*     && (row != column)) */
                        /*         continue; */
                        columns[nnz] = column;
                        rows[nnz] = row;
                        values[nnz] = (2.0*random()/RAND_MAX) -1;
                        nnz++;
                }
        }

        sparse_matrix_t m = sparse_matrix_make(ncolumns, nrows, nnz,
                                               rows, columns, values);
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
        sparse_matrix_t m = random_matrix(nrows, ncolumns);
        double * solution = calloc(ncolumns, sizeof(double));
        random_vector(solution, ncolumns);
        double * rhs = calloc(nrows, sizeof(double));
        double * x = calloc(ncolumns, sizeof(double));

        assert(0 == sparse_matrix_multiply(rhs, nrows,
                                           m, solution, ncolumns, 0));

        approx_t a = approx_make(m, nrows, rhs, NULL, ncolumns,
                                 NULL, NULL, NULL);
        double v = approx_solve(x, ncolumns, a, -1U,
                                0, 1e-10, 0,
                                stdout, 10000);

        double * residual = calloc(nrows, sizeof(double));
        assert(0 == sparse_matrix_multiply(residual, nrows,
                                           m, x, ncolumns, 0));
        double d = diff(rhs, residual, nrows);
        printf("r: %.18f %.18f %p\n", v, d, x);

        assert(d < 1e-4);

        free(residual);
        approx_free(a);
        free(x);
        free(rhs);
        free(solution);
        sparse_matrix_free(m);
}

int main ()
{
        for (size_t i = 1; i < 20; i++) {
                for (size_t j = 1; j < 20; j++) {
                        printf("%zu %zu\n", i, j);
                        test_1(i, j);
                }
        }

        return 0;
}
#endif
