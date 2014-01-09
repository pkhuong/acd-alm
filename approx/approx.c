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
                double vi = values[i];
                double w = weight[row];
                inv_v[columns[i]] += w*beta[row]*vi*vi;
        }

        size_t nvars = approx->nvars;
        for (size_t i = 0; i < nvars; i++)
                inv_v[i] = 1.0/inv_v[i];

        return 0;
}

struct vector {
        double * x;
        size_t n;
};

/* y <- (1-theta)x + theta z */
static void linterp(struct vector * OUT_yv, double theta,
                    const struct vector * xv, const struct vector * zv)
{
        assert(theta >= 0);
        assert(theta <= 1);
        double scale = 1-theta;
        size_t nvars = OUT_yv->n;
        assert(xv->n == nvars);
        assert(zv->n == nvars);

        double * OUT_y = OUT_yv->x;
        const double * x = xv->x, * z = zv->x;

        for (size_t i = 0; i < nvars; i++)
                OUT_y[i] = scale*x[i]+theta*z[i];
}

static double dot(const double * x, const struct vector * yv)
{
        size_t n = yv->n;
        const double * y = yv->x;
        double acc = 0;
        for (size_t i = 0; i < n; i++)
                acc += x[i]*y[i];
        return acc;
}

static void gradient(struct vector * OUT_grad,
                     struct vector * OUT_violation,
                     approx_t approx,
                     const struct vector * xv, double * OUT_value)
{
        size_t nvars = OUT_grad->n,
                nrows = OUT_violation->n;
        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);
        assert(nvars == xv->n);
        assert(0 == sparse_matrix_multiply(OUT_violation->x, nrows,
                                           approx->matrix, xv->x, nvars, 0));

        {
                const double * rhs = approx->rhs,
                        * weight = approx->weight;
                double * viol = OUT_violation->x;
                if (OUT_value != NULL) {
                        double value = 0;
                        for (size_t i = 0; i < nrows; i++) {
                                double v = viol[i] - rhs[i];
                                double w = weight[i];
                                value += .5*w*v*v;
                                viol[i] = v*w;
                        }
                        *OUT_value = value;
                } else {
                        for (size_t i = 0; i < nrows; i++) {
                                double v = viol[i] - rhs[i];
                                viol[i] = v*weight[i];
                        }
                }
        }

        assert(0 == sparse_matrix_multiply(OUT_grad->x, nvars,
                                           approx->matrix,
                                           OUT_violation->x, nrows,
                                           1));

        {
                double * grad = OUT_grad->x;
                const double * linear = approx->linear;
                for (size_t i = 0; i < nvars; i++)
                        grad[i] += linear[i];
        }

        if (OUT_value != NULL)
                *OUT_value += dot(approx->linear, xv);
}

static inline double min(double x, double y)
{
        return (x<y)?x:y;
}

static inline double max(double x, double y)
{
        return (x>y)?x:y;
}

static void project(struct vector * xv,
                    const double * lower, const double * upper)
{
        size_t n = xv->n;
        double * x = xv->x;
        for (size_t i = 0; i < n; i++)
                x[i] = min(max(lower[i], x[i]), upper[i]);
}

static void step(struct vector * zpv, double theta,
                 const struct vector * gv, const struct vector * zv,
                 const double * lower, const double * upper,
                 const double * inv_v)
{
        size_t n = zpv->n;
        assert(gv->n == n);
        assert(zv->n == n);
        double * zp = zpv->x;
        const double * g = gv->x, * z = zv->x;
        double inv_theta = (1-1e-6)/theta; /* protect vs rounding */
        for (size_t i = 0; i < n; i++) {   /* errors. */
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
                        double trial = zi - gi*step;
                        zp[i] = min(max(li, trial), ui);
                }
        }
}

static double next_theta(double theta)
{
        double theta2 = theta*theta,
                theta4 = theta2*theta2;
        return .5*(sqrt(theta4 + 4*theta2)-theta2);
}

static double dot_diff(const struct vector * gv,
                       const struct vector * zv, const struct vector * zpv)
{
        size_t n = gv->n;
        assert(zv->n == n);
        assert(zpv->n == n);

        const double * g = gv->x, * z = zv->x, * zp = zpv->x;

        double acc = 0;
        for (size_t i = 0; i < n; i++)
                acc += g[i]*(zp[i]-z[i]);

        return acc;
}

static double project_gradient_norm(const struct vector * gv,
                                    const struct vector * xv,
                                    const double * lower, const double * upper)
{
        size_t n = gv->n;
        assert(xv->n == n);
        const double * g = gv->x, * x = xv->x;
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
        struct vector y;
        struct vector z;
        struct vector zp;
        struct vector x;

        double theta;

        struct vector g;
        struct vector violation;
        double value;
};

static void init_vector(struct vector * x, size_t n)
{
        x->n = n;
        x->x = calloc(n, sizeof(double));
}

static void copy_vector(struct vector * x, const struct vector * y)
{
        size_t n = x->n;
        assert(n == y->n);
        memcpy(x->x, y->x, n*sizeof(double));
}

static void destroy_vector(struct vector * x)
{
        free(x->x);
        memset(x, 0, sizeof(struct vector));
}

static void init_state(struct approx_state * state,
                       size_t nvars, size_t nrows)
{
        init_vector(&state->y, nvars);
        init_vector(&state->z, nvars);
        init_vector(&state->zp, nvars);
        init_vector(&state->x, nvars);

        state->theta = 1;

        init_vector(&state->g, nvars);
        init_vector(&state->violation, nrows);
        state->value = HUGE_VAL;
}

static void destroy_state(struct approx_state * state)
{
        destroy_vector(&state->y);
        destroy_vector(&state->z);
        destroy_vector(&state->zp);
        destroy_vector(&state->x);
        destroy_vector(&state->g);
        destroy_vector(&state->violation);

        memset(state, 0, sizeof(struct approx_state));
}

static const struct vector *
iter(approx_t approx, struct approx_state * state, double * OUT_pg)
{
        linterp(&state->y, state->theta,
                &state->x, &state->z);
        gradient(&state->g, &state->violation,
                 approx, &state->y, NULL);
        step(&state->zp, state->theta,
             &state->g, &state->z,
             approx->lower, approx->upper,
             approx->inv_v);

        gradient(&state->g, &state->violation,
                 approx, &state->z, &state->value);
        *OUT_pg = project_gradient_norm(&state->g, &state->z,
                                        approx->lower, approx->upper);

        if (dot_diff(&state->g, &state->z, &state->zp) > 0) {
                /* Oscillation */
                copy_vector(&state->x, &state->z);
                state->theta = 1;
                return &state->x;
        }

        linterp(&state->x, state->theta,
                &state->x, &state->zp);
        state->theta = next_theta(state->theta);
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
        double acc = 0;
        for (size_t i = 0; i < n; i++) {
                double delta = x[i]-y[i];
                acc += delta*delta;
        }
        return sqrt(acc);
}

static double norm_2(const struct vector * xv)
{
        size_t n = xv->n;
        const double * x = xv->x;
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
                fprintf(log, "\t%10zu %12g %12g %12g %12g\n",
                        k, value, ng, pg, diff);
        else   fprintf(log, "\t%10zu %12g %12g %12g\n",
                       k, value, ng, pg);
}

int approx_solve(double * x, size_t n, approx_t approx, size_t niter,
                 double max_pg, double max_value, double min_delta,
                 FILE * log, size_t period, double * OUT_diagnosis,
                 double offset)
{
        assert(n == approx->nvars);

        struct approx_state state;
        init_state(&state, approx->nvars, approx->nrhs);
        memcpy(state.x.x, x, n*sizeof(double));
        project(&state.x, approx->lower, approx->upper);
        copy_vector(&state.z, &state.x);

        double * prev_x = calloc(n, sizeof(double));
        memcpy(prev_x, state.x.x, n*sizeof(double));

        const struct vector * center = &state.x;
        double value = state.value;
        double ng = HUGE_VAL, pg = HUGE_VAL;
        double delta = HUGE_VAL;
        size_t i;
        int restart = 0, reason = 0;
        for (i = 0; i < niter; i++) {
                delta = HUGE_VAL;
                center = iter(approx, &state, &pg);
                if (center == &state.x) {
                        if (!restart) {
                                restart = 1;
                                if (log != NULL)
                                        fprintf(log, "\t\t");
                        }
                        if (log != NULL)
                                fprintf(log, "R");
                }
                ng = norm_2(&state.g);
                value = state.value;
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
                        gradient(&state.g, &state.violation,
                                 approx, &state.x, &value);
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
                }
                if ((i == 0) || (period && ((i+1)%period == 0))) {
                        if (restart) {
                                restart = 0;
                                printf("\n");
                        }
                        print_log(log, i+1, value+offset, ng, pg, delta);
                }
        }
        if (restart) {
                restart = 0;
                printf("\n");
        }
        print_log(log, i+1, value+offset, ng, pg, delta);

        memcpy(x, center->x, n*sizeof(double));
        if (OUT_diagnosis != NULL) {
                OUT_diagnosis[0] = value+offset;
                OUT_diagnosis[1] = ng;
                OUT_diagnosis[2] = pg;
                OUT_diagnosis[3] = delta;
                OUT_diagnosis[4] = i+1;
        }

        free(prev_x);
        destroy_state(&state);

        return reason;
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
                        if (((1.0*random()/RAND_MAX) < .5)
                            && (row != column))
                                continue;
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
        double diagnosis[5];
        int r = approx_solve(x, ncolumns, a, -1U,
                             0, 1e-13, 0,
                             stdout, 10000, diagnosis, 0);

        assert(r > 0);

        double * residual = calloc(nrows, sizeof(double));
        assert(0 == sparse_matrix_multiply(residual, nrows,
                                           m, x, ncolumns, 0));
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
        for (size_t i = 10; i <= 20; i++) {
                for (size_t j = 10; j <= 20; j++) {
                        printf("%zu %zu\n", i, j);
                        test_1(i, j);
                }
        }

        return 0;
}
#endif
