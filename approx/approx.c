#include "approx.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <strings.h>

#include <sys/mman.h>

#if defined(MAP_HUGETLB) && defined(USE_MMAP)
void * large_calloc(size_t n, size_t size)
{
        size_t mask = (1ul<<21)-1;
        size_t bytes = (n*size+16+mask)&(~mask);
        void * ret = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                          MAP_ANONYMOUS|MAP_HUGETLB|MAP_PRIVATE,
                          -1, 0);
        if (ret == MAP_FAILED)
                ret = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                           MAP_ANONYMOUS|MAP_PRIVATE,
                           -1, 0);
        assert(ret != MAP_FAILED);
        return ret;
}

void large_free(void * ptr, size_t n, size_t size)
{
        size_t mask = (1ul<<21)-1;
        size_t bytes = (n*size+16+mask)&(~mask);
        munmap(ptr, bytes);
}
#else
void * large_calloc(size_t n, size_t size)
{
#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
        void * ptr = 0;
        assert(0 == posix_memalign(&ptr, 16, n*size+16));
        return ptr;
#else
        return calloc(n, size);
#endif
}

void large_free(void * ptr, size_t n, size_t size)
{
        (void)n;
        (void)size;
        free(ptr);
}
#endif

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
        double * out = large_calloc(n, sizeof(double));
        memcpy(out, x, sizeof(double)*n);
        return out;
}

static double * copy_double_default(const double * x, size_t n, double missing)
{
        double * out = large_calloc(n, sizeof(double));
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

        approx->beta = large_calloc(nrhs, sizeof(uint32_t));
        approx->inv_v = large_calloc(nvars, sizeof(double));

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
        size_t nrhs = approx->nrhs,
                nvars = approx->nvars;

        large_free(approx->rhs, nrhs, sizeof(double));
        large_free(approx->weight, nrhs, sizeof(double));
        large_free(approx->linear, nvars, sizeof(double));
        large_free(approx->lower, nvars, sizeof(double));
        large_free(approx->upper, nvars, sizeof(double));
        large_free(approx->beta, nrhs, sizeof(double));
        large_free(approx->inv_v, nvars, sizeof(double));
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
        double * violation; /* Ax-b */
        size_t n;
        size_t nviolation;
        int violationp;
};

static void init_vector(struct vector * x, size_t n, size_t nviolation)
{
        x->n = n;
        x->x = large_calloc(n, sizeof(double));
        if (nviolation) {
                x->violation = large_calloc(nviolation, sizeof(double));
                x->nviolation = nviolation;
        } else {
                x->violation = NULL;
                x->nviolation = 0;
        }
        x->violationp = 0;
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
        large_free(x->x, x->n, sizeof(double));
        large_free(x->violation, x->nviolation, sizeof(double));
        memset(x, 0, sizeof(struct vector));
}

typedef double v2d __attribute__ ((vector_size (16)));

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

static void compute_violation(struct vector * xv, approx_t approx)
{
        size_t nvars = xv->n,
                nrows = xv->nviolation;

        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);

        assert(0 == sparse_matrix_multiply(xv->violation, nrows,
                                           approx->matrix, xv->x, nvars, 0));
        const double * rhs = approx->rhs;
        double * viol = xv->violation;
        for (size_t i = 0; i < nrows; i++)
                viol[i] -= rhs[i];
        xv->violationp = 1;
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
                const double * weight = approx->weight;
                double * viol = xv->violation;
                if (OUT_value == NULL) {
                        for (size_t i = 0; i < nrows; i++)
                                scaled[i] = weight[i]*viol[i];
                } else  {
                        double value = 0;
                        for (size_t i = 0; i < nrows; i++) {
                                double v = viol[i];
                                double w = weight[i];
                                value += .5*w*v*v;
                                scaled[i] = v*w;
                        }
                        *OUT_value = value;           
                }
        }

        assert(0 == sparse_matrix_multiply(OUT_grad->x, nvars,
                                           approx->matrix,
                                           scaled, nrows,
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
                double * scaled = OUT_scaled[i]->x;
                {
                        const double * weight = approx->weight;
                        double * viol = xv[i]->violation;
                        if (OUT_value[i] == NULL) {
                                for (size_t i = 0; i < nrows; i++)
                                        scaled[i] = weight[i]*viol[i];
                        } else  {
                                double value = 0;
                                for (size_t i = 0; i < nrows; i++) {
                                        double v = viol[i];
                                        double w = weight[i];
                                        value += .5*w*v*v;
                                        scaled[i] = v*w;
                        }
                                *OUT_value[i] = value;           
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
                                                     1));
        }

        for (size_t i = 0; i < 2; i++) {
                double * grad = OUT_grad[i]->x;
                const double * linear = approx->linear;
                for (size_t i = 0; i < nvars; i++)
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
        size_t n = xv->n;
        double * x = xv->x;
        for (size_t i = 0; i < n; i++)
                x[i] = min(max(lower[i], x[i]), upper[i]);
}

static void step(struct vector * zpv, double theta,
                 const struct vector * gv, const struct vector * zv,
                 const double * restrict lower, const double * restrict upper,
                 const double * restrict inv_v)
{
        size_t n = zpv->n;
        assert(gv->n == n);
        assert(zv->n == n);
        double * restrict zp = zpv->x;
        const double * restrict g = gv->x, * restrict z = zv->x;
        double max_z = 0;
        double inv_theta = (1-1e-6)/theta; /* protect vs rounding */
        for (size_t i = 0; i < n; i++) {   /* errors. */
                double gi = g[i], zi = z[i],
                        li = lower[i], ui = upper[i],
                        inv_vi = inv_v[i];
                double step = inv_theta*inv_vi;
#ifndef UNSAFE_DESCENT_STEP
                step = ((step >= HUGE_VAL) && (gi == 0))?0:step;
#endif
                double trial = zi - gi*step;
                zp[i] = trial = min(max(li, trial), ui);
                max_z = max(max_z, fabs(trial));
        }
        assert(max_z < HUGE_VAL);
        zpv->violationp = 0; /* cache is now invalid */
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

        struct vector g, g2;
        struct vector violation, violation2;
        double value;
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
                double * values[2] = {&state->value, NULL};
                gradient2(g, approx, violation, x, values);
        }
        step(&state->zp, state->theta,
             &state->g2, &state->z,
             approx->lower, approx->upper,
             approx->inv_v);

        *OUT_pg = project_gradient_norm(&state->g, &state->z,
                                        approx->lower, approx->upper);

        if (dot_diff(&state->g, &state->z, &state->zp) > 0) {
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

        set_vector(&state.x, x, approx);
        compute_violation(&state.x, approx);
        copy_vector(&state.z, &state.x);
        double * prev_x = large_calloc(n, sizeof(double));
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

        large_free(prev_x, n, sizeof(double));
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
