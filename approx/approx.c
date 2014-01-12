#include "approx.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include "../huge_alloc/huge_alloc.h"

struct approx {
        size_t nrhs, nvars;
        sparse_matrix_t matrix;
        double * rhs;
        double * weight; /* \sum_i weight_i [(Ax-b)_i]^2 */
        double * linear; /* linear x + [LS] */

        double * lower, * upper; /* box */

        uint32_t * beta;
        double * v;
        double * inv_v;
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

        approx->beta = huge_calloc(nrhs, sizeof(uint32_t));
        approx->v = huge_calloc(nvars, sizeof(double));
        approx->inv_v = huge_calloc(nvars, sizeof(double));

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
        huge_free(approx->rhs);
        huge_free(approx->weight);
        huge_free(approx->linear);
        huge_free(approx->lower);
        huge_free(approx->upper);
        huge_free(approx->beta);
        huge_free(approx->v);
        huge_free(approx->inv_v);
        memset(approx, 0, sizeof(struct approx));
        free(approx);

        return 0;
}

int approx_update_step_sizes(approx_t approx)
{
        assert(approx->nrhs == sparse_matrix_nrows(approx->matrix));
        assert(approx->nvars == sparse_matrix_ncolumns(approx->matrix));

        uint32_t * beta = approx->beta;
        double * v = approx->v;
        const double * weight = approx->weight;
        memset(beta, 0, approx->nrhs*sizeof(uint32_t));
        memset(v, 0, approx->nvars*sizeof(double));

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

struct vector {
        double * x;
        double * violation; /* Ax-b */
        size_t n;
        size_t nviolation;
        int violationp;
        double value;
};

static void init_vector(struct vector * x, size_t n, size_t nviolation)
{
        x->n = n;
        x->x = huge_calloc(n, sizeof(double));
        if (nviolation) {
                x->violation = huge_calloc(nviolation, sizeof(double));
                x->nviolation = nviolation;
        } else {
                x->violation = NULL;
                x->nviolation = 0;
        }
        x->violationp = 0;
        x->value = nan("");
}

static void copy_vector(struct vector * x, const struct vector * y)
{
        size_t n = x->n;
        assert(n == y->n);
        memcpy(x->x, y->x, n*sizeof(double));
        if (y->violationp
            && (y->violation != NULL)
            && (x->violation != NULL)) {
                size_t m = x->nviolation;
                assert(m == y->nviolation);
                memcpy(x->violation, y->violation, m*sizeof(double));
                x->violationp = 1;
        } else {
                x->violationp = 0;
        }
        x->value = y->value;
}

static void project(struct vector * xv,
                    const double * lower, const double * upper);

static void set_vector(struct vector * x, const double * src,
                       approx_t approx)
{
        memcpy(x->x, src, x->n*sizeof(double));
        project(x, approx->lower, approx->upper);
}

static void destroy_vector(struct vector * x)
{
        huge_free(x->x);
        huge_free(x->violation);
        memset(x, 0, sizeof(struct vector));
}

typedef double v2d __attribute__ ((vector_size (16)));
typedef uint64_t v2ul __attribute__ ((vector_size (16)));

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

        v2d scale2 = {scale, scale}, theta2 = {theta, theta};

        {
                v2d * OUT_y = (v2d*)OUT_yv->x;
                const v2d * x = (v2d*)xv->x, * z = (v2d*)zv->x;

                size_t n = (nvars+1)/2;
                for (size_t i = 0; i < n; i++)
                        OUT_y[i] = scale2*x[i]+theta2*z[i];
        }

        size_t nviolation = OUT_yv->nviolation;
        if (nviolation && xv->violationp && zv->violationp) {
                assert(xv->nviolation == nviolation);
                assert(zv->nviolation == nviolation);
                v2d * OUT_y = (v2d*)OUT_yv->violation;
                const v2d * x = (v2d*)xv->violation,
                        * z = (v2d*)zv->violation;
                size_t n = (nviolation+1)/2;
                for (size_t i = 0; i < n; i++)
                        OUT_y[i] = scale2*x[i]+theta2*z[i];
                OUT_yv->violationp = 1;
        } else {
                OUT_yv->violationp = 0;
        }
        OUT_yv->value = nan("");
}

static double dot(const double * xp, const struct vector * yv)
{
        size_t n = (yv->n+1)/2;
        const v2d * x = (v2d*)xp;
        const v2d * y = (v2d*)yv->x;
        v2d acc = {0,0};
        for (size_t i = 0; i < n; i++)
                acc += x[i]*y[i];
        return acc[0]+acc[1];
}

static void compute_violation(struct vector * xv, approx_t approx)
{
        size_t nvars = xv->n,
                nrows = xv->nviolation;

        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);

        assert(0 == sparse_matrix_multiply(xv->violation, nrows,
                                           approx->matrix, xv->x, nvars, 0,
                                           NULL));
        const v2d * rhs = (v2d*)approx->rhs;
        v2d * viol = (v2d*)xv->violation;
        size_t n = (nrows+1)/2;
        for (size_t i = 0; i < n; i++)
                viol[i] -= rhs[i];
        xv->violationp = 1;
}

static double value(approx_t approx, struct vector * xv)
{
        size_t nvars = xv->n;
        size_t nrows = xv->nviolation;

        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);

        {
                double value = xv->value;
                if (!isnan(value)) return value;
        }

#ifndef NO_CACHING
        if (!xv->violationp)
#endif
                compute_violation(xv, approx);

        double value;
        {
                const v2d * weight = (v2d*)approx->weight;
                v2d * viol = (v2d*)xv->violation;

                v2d acc = {0, 0};
                size_t n = (nrows+1)/2;
                for (size_t i = 0; i < n; i++) {
                        v2d v = viol[i];
                        v2d w = weight[i];
                        v2d scaled = v*w;
                        acc += v*scaled;
                }
                value = .5*(acc[0]+acc[1]);
        }

        return xv->value = value+dot(approx->linear, xv);
}

static void gradient(struct vector * OUT_grad,
                     approx_t approx, struct vector * OUT_scaled,
                     struct vector * xv, double * OUT_value)
{
        size_t nvars = OUT_grad->n,
                nrows = xv->nviolation;
        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);
        assert(nrows == OUT_scaled->n);
        assert(nvars == xv->n);

#ifndef NO_CACHING
        if (!xv->violationp)
#endif
                compute_violation(xv, approx);

        double * scaled = OUT_scaled->x;
        {
                size_t n = (nrows+1)/2;
                v2d * out = (v2d*)scaled;
                const v2d * weight = (v2d*)approx->weight;
                v2d * viol = (v2d*)xv->violation;
                if (OUT_value == NULL) {
                        for (size_t i = 0; i < n; i++)
                                out[i] = weight[i]*viol[i];
                } else  {
                        v2d value = {0,0};
                        for (size_t i = 0; i < n; i++) {
                                v2d v = viol[i];
                                v2d w = weight[i];
                                v2d scaled = v*w;
                                value += v*scaled;
                                out[i] = scaled;
                        }
                        *OUT_value = .5*(value[0]+value[1]);
                }
        }

        assert(0 == sparse_matrix_multiply(OUT_grad->x, nvars,
                                           approx->matrix,
                                           scaled, nrows,
                                           1, NULL));

        {
                v2d * grad = (v2d*)OUT_grad->x;
                const v2d * linear = (v2d*)approx->linear;
                size_t n = (nvars+1)/2;
                for (size_t i = 0; i < n; i++)
                        grad[i] += linear[i];
        }

        if (OUT_value != NULL)
                *OUT_value += dot(approx->linear, xv);
}

static void gradient2(struct vector ** OUT_grad,
                      approx_t approx, struct vector ** OUT_scaled,
                      struct vector ** xv, double ** OUT_value)
{
        size_t nvars = OUT_grad[0]->n,
                nrows = xv[0]->nviolation;
        assert(OUT_grad[1]->n == nvars);
        assert(xv[1]->nviolation == nrows);
        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);
        for (size_t i = 0; i < 2; i++) {
                assert(nrows == OUT_scaled[i]->n);
                assert(nvars == xv[i]->n);
        }

        for (size_t i = 0; i < 2; i++) {
#ifndef NO_CACHING
                if (!xv[i]->violationp)
#endif
                        compute_violation(xv[i], approx);
        }

        for (size_t i = 0; i < 2; i++) {
                v2d * scaled = (v2d*)OUT_scaled[i]->x;
                {
                        const v2d * weight = (v2d*)approx->weight;
                        v2d * viol = (v2d*)xv[i]->violation;
                        size_t n = (nrows+1)/2;
                        if (OUT_value[i] == NULL) {
                                for (size_t i = 0; i < n; i++)
                                        scaled[i] = weight[i]*viol[i];
                        } else {
                                v2d value = {0,0};
                                for (size_t i = 0; i < n; i++) {
                                        v2d v = viol[i];
                                        v2d w = weight[i];
                                        v2d s = v*w;
                                        value += s*v;
                                        scaled[i] = s;
                                }
                                *OUT_value[i] = .5*(value[0]+value[1]);
                        }
                }
        }

        {
                double * grad[2] = {OUT_grad[0]->x, OUT_grad[1]->x};
                const double * scaled[2] = {OUT_scaled[0]->x,
                                            OUT_scaled[1]->x};
                assert(0 == sparse_matrix_multiply_2(grad, nvars,
                                                     approx->matrix,
                                                     scaled, nrows,
                                                     1, NULL));
        }

        for (size_t i = 0; i < 2; i++) {
                v2d * grad = (v2d*)OUT_grad[i]->x;
                const v2d * linear = (v2d*)approx->linear;
                size_t n = (nvars+1)/2;
                for (size_t i = 0; i < n; i++)
                        grad[i] += linear[i];
                if (OUT_value[i] != NULL)
                        *OUT_value[i] += dot(approx->linear, xv[i]);
        }
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
        size_t n = (xv->n+1)/2;
        v2d * x = (v2d*)xv->x;
        const v2d * l = (const v2d*)lower, * u = (const v2d*)upper;
        for (size_t i = 0; i < n; i++) {
                v2d clamp_low = __builtin_ia32_maxpd(l[i], x[i]);
                v2d clamp_high = __builtin_ia32_minpd(clamp_low, u[i]);
                x[i] = clamp_high;
        }
}

static void step(struct vector * zpv, double theta,
                 const struct vector * gv, const struct vector * zv,
                 const double * restrict lower, const double * restrict upper,
                 const double * restrict inv_v)
{
        size_t n = zpv->n;
        assert(gv->n == n);
        assert(zv->n == n);

        size_t vector_n = (n+1)/2;
        v2d * zp = (v2d*)zpv->x;
        const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                * l = (const v2d*)lower, * u = (const v2d*)upper,
                * iv = (const v2d*)inv_v;
        v2d max_z = {0,0};
        double inv_theta = (1-1e-6)/theta;   /* protect vs rounding */
        v2d itheta = {inv_theta, inv_theta}; /* errors. */
        v2ul mask = {~(1ull<<63), ~(1ull<<63)};
        for (size_t i = 0; i < vector_n; i++) {
                v2d gi = g[i], zi = z[i],
                        li = l[i], ui = u[i],
                        inv_vi = iv[i];
                v2d step = itheta*inv_vi;
                v2d trial = zi - gi*step;
                trial = __builtin_ia32_maxpd(li, trial);
                trial = __builtin_ia32_minpd(ui, trial);
                zp[i] = trial;
                max_z = __builtin_ia32_maxpd(max_z,
                                             (v2d)((v2ul)trial&mask));
        }
        assert(max(max_z[0], max_z[1]) < HUGE_VAL);
        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
}

/*
 * f(x) ~= f(z) + g'x + \sum_i (theta v_i)/2 (x-z)^2_i
 *   min in z_i - 1/(theta v_i) g_i
 */
static double
long_step(struct vector * zpv, double theta, double length,
          const struct vector * gv, const struct vector * zv,
          const double * restrict lower, const double * restrict upper,
          const double * restrict inv_v, const double * restrict vs)
{
        size_t n = zpv->n;
        assert(gv->n == n);
        assert(zv->n == n);

        size_t vector_n = (n+1)/2;
        v2d * zp = (v2d*)zpv->x;
        const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                * l = (const v2d*)lower, * u = (const v2d*)upper,
                * iv = (const v2d*)inv_v, *v = (const v2d*)vs;
        v2d max_z = {0,0};
        assert(length >= 1);
        double inv_theta = (length-1e-6)/theta;
        v2d itheta = {inv_theta, inv_theta};
        v2d theta2 = {theta/length, theta/length};
        v2ul mask = {~(1ull<<63), ~(1ull<<63)};
        v2d linear_estimate = {0, 0};
        v2d quad_estimate = {0, 0};
        for (size_t i = 0; i < vector_n; i++) {
                v2d gi = g[i], zi = z[i],
                        li = l[i], ui = u[i],
                        inv_vi = iv[i],
                        vi = v[i];
                v2d step = itheta*inv_vi;
                v2d trial = zi - gi*step;
                trial = __builtin_ia32_maxpd(li, trial);
                trial = __builtin_ia32_minpd(ui, trial);
                zp[i] = trial;
                max_z = __builtin_ia32_maxpd(max_z,
                                             (v2d)((v2ul)trial&mask));
                v2d delta = trial - zi;
                linear_estimate += gi*delta;
                quad_estimate += delta*delta*theta2*vi;
        }
        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
        if (isfinite(max(max_z[0], max_z[1])))
                return ((linear_estimate[0]+linear_estimate[1])
                        +.5*(quad_estimate[0]+quad_estimate[1]));
        return -HUGE_VAL; /* Might be a finite precision problem. fail */
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

        double theta;

        struct vector g, g2;
        struct vector violation, violation2;
        double value;

        double step_length;
};

static void init_state(struct approx_state * state,
                       size_t nvars, size_t nrows)
{
        init_vector(&state->y, nvars, nrows);
        init_vector(&state->z, nvars, nrows);
        init_vector(&state->zp, nvars, nrows);
        init_vector(&state->x, nvars, nrows);

        state->theta = 1;

        init_vector(&state->g, nvars, 0);
        init_vector(&state->g2, nvars, 0);
        init_vector(&state->violation, nrows, 0);
        init_vector(&state->violation2, nrows, 0);
        state->value = HUGE_VAL;

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
        destroy_vector(&state->violation);
        destroy_vector(&state->violation2);

        memset(state, 0, sizeof(struct approx_state));
}

static const struct vector *
iter(approx_t approx, struct approx_state * state, double * OUT_pg)
{
        linterp(&state->y, state->theta,
                &state->x, &state->z);
        {
                struct vector * g[2] = {&state->g, &state->g2};
                struct vector * violation[2] = {&state->violation,
                                                &state->violation2};
                struct vector * x[2]= {&state->z, &state->y};
                double * values[2] = {NULL, NULL};
                gradient2(g, approx, violation, x, values);
                state->value = value(approx, &state->z);
        }

        int descent_achieved = 0;
        for (int i = 0; i < 2; i++) {
                double step_length = state->step_length;
#ifdef STATIC_STEP
                step_length = state->step_length = 1;
#endif
                if (i || (step_length <= 1)) {
                        step(&state->zp, state->theta,
                             &state->g2, &state->z,
                             approx->lower, approx->upper,
                             approx->inv_v);
                        if (i == 0) {
                                state->step_length *= 1.01;
                                if (state->step_length > 1)
                                        state->step_length = 1+1e-6;
                        }
                        break;
                } else {
                        /* Special code to tell when the step is
                         * definitely OK. If so, perform a normal
                         * step, but update step_length.
                         */
                        int safe = (step_length <= 1+1e-6);
                        if (safe) step_length = 1;
                        double expected_improvement
                                = long_step(&state->zp,
                                            state->theta, step_length,
                                            &state->g2, &state->z,
                                            approx->lower, approx->upper,
                                            approx->inv_v, approx->v);
                        double initial = value(approx, &state->z);
                        double now = value(approx, &state->zp);
                        assert(expected_improvement <= 0);
                        if (now > initial+expected_improvement) {
                                state->step_length = .9*step_length;
                                /* Bad guess, but safe step anyway */
                                if (safe) break;
                        } else {
                                state->step_length = step_length*1.01;
                                descent_achieved = 1;
                                break;
                        }
                }
        }

        *OUT_pg = project_gradient_norm(&state->g, &state->z,
                                        approx->lower, approx->upper);

        if ((!descent_achieved) /* Value improvement OK */
            && (dot_diff(&state->g, &state->z, &state->zp) > 0)) {
                /* Oscillation */
                copy_vector(&state->x, &state->z);
                state->theta = 1;
                return &state->x;
        }

        compute_violation(&state->zp, approx);
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
        const v2d * x = (v2d*)xv->x+1;
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

int approx_solve(double * x, size_t n, approx_t approx, size_t niter,
                 double max_pg, double max_value, double min_delta,
                 FILE * log, size_t period, double * OUT_diagnosis,
                 double offset)
{
        assert(n == approx->nvars);

        struct approx_state state;
        init_state(&state, approx->nvars, approx->nrhs);

        set_vector(&state.x, x, approx);
        compute_violation(&state.x, approx);
        copy_vector(&state.z, &state.x);
        double * prev_x = huge_calloc(n, sizeof(double));
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
                        gradient(&state.g, approx, &state.violation,
                                 &state.x, &value);
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
                        compute_violation(&state.x, approx);
                }
                if ((i == 0) || (period && ((i+1)%period == 0))) {
                        if (restart) {
                                restart = 0;
                                printf("\n");
                        }
                        print_log(log, i+1, value+offset, ng, pg,
                                  state.step_length, delta);
                }
        }
        if (restart) {
                restart = 0;
                printf("\n");
        }
        print_log(log, i+1, value+offset, ng, pg,
                  state.step_length, delta);

        memcpy(x, center->x, n*sizeof(double));
        if (OUT_diagnosis != NULL) {
                OUT_diagnosis[0] = value+offset;
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
                                           m, solution, ncolumns, 0,
                                           NULL));

        approx_t a = approx_make(m, nrows, rhs, NULL, ncolumns,
                                 NULL, NULL, NULL);
        double diagnosis[5];
        int r = approx_solve(x, ncolumns, a, -1U,
                             0, 1e-13, 0,
                             stdout, 10000, diagnosis, 0);

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
