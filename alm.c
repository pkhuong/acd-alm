#include "alm.h"
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <assert.h>

struct alm {
        size_t nrhs, nvars;
        sparse_matrix_t * matrix;
        double * linear;
        approx_t * approx;

        double * lambda_lower, * lambda_upper;
};

#define DEF(TYPE, FIELD)                                        \
        TYPE alm_##FIELD(alm_t * alm)                             \
        {                                                       \
                return alm->FIELD;                              \
        }

DEF(sparse_matrix_t *, matrix)
DEF(size_t, nrhs)
DEF(size_t, nvars)
DEF(double *, linear)
DEF(double *, lambda_lower)
DEF(double *, lambda_upper)

#undef DEF

double * alm_rhs(alm_t * alm)
{
        return approx_rhs(alm->approx);
}

double * alm_lower(alm_t * alm)
{
        return approx_lower(alm->approx);
}

double * alm_upper(alm_t * alm)
{
        return approx_upper(alm->approx);
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

alm_t * alm_make(sparse_matrix_t * constraints,
                 size_t nrhs, const double * rhs,
                 size_t nvars, const double * linear,
                 const double * lower, const double * upper,
                 const double * lambda_lower, const double * lambda_upper)
{
        alm_t * alm = calloc(1, sizeof(alm_t));
        alm->nrhs = nrhs;
        alm->nvars = nvars;

        alm->matrix = constraints;
        alm->linear = copy_double_default(linear, nvars, 0);
        alm->approx = approx_make(constraints, nrhs, rhs,
                                  NULL,
                                  nvars, NULL,
                                  lower, upper);
        assert(nrhs == approx_nrhs(alm->approx));
        assert(nvars == approx_nvars(alm->approx));
        alm->lambda_lower = copy_double_default(lambda_lower, nrhs,
                                                -HUGE_VAL);
        alm->lambda_upper = copy_double_default(lambda_upper, nrhs,
                                                HUGE_VAL);

        return alm;
}

int alm_free(alm_t * alm)
{
        if (alm == NULL) return 0;

        free(alm->linear);
        free(alm->lambda_lower);
        free(alm->lambda_upper);
        approx_free(alm->approx);
        memset(alm, 0, sizeof(alm_t));
        free(alm);
        return 0;
}

static double dot(const double * x, const double * y, size_t n)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++)
                acc += x[i]*y[i];
        return acc;
}

static double penalise_linear(alm_t * alm, const double * lambda)
{
        size_t nvars = alm->nvars;
        double * c = approx_linear(alm->approx);
        assert(0 == sparse_matrix_multiply(c, nvars,
                                           alm->matrix,
                                           lambda, alm->nrhs,
                                           1, NULL));
        double * linear = alm->linear;
        for (size_t i = 0; i < nvars; i++)
                c[i] = linear[i]-c[i];

        return dot(lambda, approx_rhs(alm->approx), alm->nrhs);
}

static void violation(double * OUT_viol, alm_t * alm, const double * x)
{
        size_t nrhs = alm->nrhs;
        assert(0 == sparse_matrix_multiply(OUT_viol, nrhs,
                                           alm->matrix,
                                           x, alm->nvars,
                                           0, NULL));

        const double * rhs = approx_rhs(alm->approx);
        for (size_t i = 0; i < nrhs; i++)
                OUT_viol[i] -= rhs[i];
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

static double norm_1(const double * x, size_t n)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++) {
                double xi = x[i];
                acc += fabs(xi);
        }
        return acc;
}

static double norm_inf(const double * x, size_t n, size_t * OUT_i)
{
        double max = -HUGE_VAL;
        size_t idx = -1ul;

        for (size_t i = 0; i < n; i++) {
                double xi = fabs(x[i]);
                if (xi > max) {
                        max = xi;
                        idx = i;
                }
        }

        if (OUT_i != NULL) *OUT_i = idx;
        return max;
}

static void project_multipliers(double * multipliers, alm_t * alm)
{
        size_t nrhs = alm->nrhs;
        const double * lower = alm->lambda_lower,
                * upper = alm->lambda_upper;

        for (size_t i = 0; i < nrhs; i++)
                multipliers[i] = fmax(lower[i], 
                                      fmin(multipliers[i], upper[i]));
}

static void update_multipliers(double * multipliers,
                               alm_t * alm, const double * violation)
{
        size_t nrhs = alm->nrhs;
        const double * weight = approx_weight(alm->approx);

        for (size_t i = 0; i < nrhs; i++)
                multipliers[i] -= weight[i]*violation[i];

        project_multipliers(multipliers, alm);
}

static double update_weights(alm_t * alm, const double * violation,
                             double prev_norm)
{
        size_t nrhs = alm->nrhs;
        double norm = norm_2(violation, nrhs);
        if (norm < .5*prev_norm) return norm;
        
        double * weights = approx_weight(alm->approx);
        double scale = 2*fmin(1, norm/prev_norm);
        for (size_t i = 0; i < nrhs; i++)
                weights[i] *= scale;

        size_t max_i;
        double max = norm_inf(violation, nrhs, &max_i);
        if (max > .5*norm)
                weights[max_i] *= 10;

        return norm;
}

struct alm_state {
        double * violation;
        double prev_viol_norm;
        double precision;
};


static void init_alm_state(struct alm_state * state, alm_t * alm)
{
        size_t nrhs = alm->nrhs;
        state->violation = calloc(nrhs, sizeof(double));
        state->prev_viol_norm = HUGE_VAL;
        state->precision = 1;
}

static void free_alm_state(struct alm_state * state)
{
        free(state->violation);
        memset(state, 0, sizeof(struct alm_state));
}

static void update_precision(struct alm_state * state, alm_t * alm,
                             const double * violation)
{
        size_t nrhs = alm->nrhs;
        double viol = norm_inf(violation, nrhs, NULL);
        double w = nrhs/norm_1(approx_weight(alm->approx), nrhs);
        state->precision = fmin(state->precision,
                                fmin(pow(viol, 1),
                                     pow(w, 1.5)));
        state->precision = fmax(state->precision, 1e-7);
}

static int iter(struct alm_state * state, alm_t * alm,
                double * x, double * lambda, FILE * log, size_t k,
                double * OUT_pg, double * OUT_max_viol,
                thread_pool_t * pool)
{
        double offset = penalise_linear(alm, lambda);
        double diag[5];
        approx_update(alm->approx);
        int reason = cd_solve(x, alm->nvars, alm->approx, -1u,
                              fmin(1, state->precision), -HUGE_VAL, 1e-11,
                              log, 10000, diag, offset, pool);
        violation(state->violation, alm, x);
        double max_viol = norm_inf(state->violation, alm->nrhs, NULL);
        if (log != NULL)
                fprintf(log, "%4zu: %12g %12g %12g %.18g %.18g",
                        k,
                        max_viol,
                        norm_2(state->violation, alm->nrhs),
                        diag[2],
                        diag[0],
                        dot(x, alm->linear, alm->nvars));

        if (OUT_pg != NULL) *OUT_pg = diag[2];
        if (OUT_max_viol != NULL) *OUT_max_viol = max_viol;

        update_multipliers(lambda, alm, state->violation);
        if (reason != 2) {
                if (log != NULL)
                        fprintf(log, " -stalled-\n");
                update_precision(state, alm, state->violation);
                return 1;
        }

        state->prev_viol_norm = update_weights(alm, state->violation,
                                               state->prev_viol_norm);
        update_precision(state, alm, state->violation);
        if (log != NULL)
                fprintf(log, " (%8g)\n", norm_1(approx_weight(alm->approx),
                                               alm->nrhs)/alm->nrhs);
        return 0;
}

int alm_solve(alm_t * alm, size_t niter, double * x, size_t nvars,
              double * lambda, size_t nconstraints,
              FILE * log, double * OUT_diagnosis, 
              thread_pool_t * pool)
{
        struct alm_state state;
        assert(nvars == alm->nvars);
        assert(nconstraints == alm->nrhs);

        init_alm_state(&state, alm);
        project_multipliers(lambda, alm);

        double pg = HUGE_VAL, max_viol = HUGE_VAL;
        int ret = 1;
        for (size_t i = 0; i < niter; i++) {
                int status = iter(&state, alm, x, lambda, log, i+1,
                                  &pg, &max_viol, pool);
                if ((max_viol < 1e-5)
                    && (status /* stalled */
                        || (pg < 1e-5))) {
                        ret = 0;
                        break;
                }
        }

        if (OUT_diagnosis != NULL) {
                OUT_diagnosis[0] = dot(x, alm->linear, alm->nvars);
                OUT_diagnosis[1] = max_viol;
                OUT_diagnosis[2] = pg;
        }

        free_alm_state(&state);
        return ret;
}

void read_doubles(FILE * stream, double * out, size_t n)
{
        for (size_t i = 0; i < n; i++)
                assert(1 == fscanf(stream, " %lf", out+i));
}

alm_t * alm_read(FILE * stream)
{
        sparse_matrix_t * m = sparse_matrix_read(stream, 0);
        size_t nvars = sparse_matrix_ncolumns(m),
                nrhs = sparse_matrix_nrows(m);
        double * rhs = calloc(nrhs, sizeof(double));
        double * linear = calloc(nvars, sizeof(double));

        double * lower = calloc(nvars, sizeof(double)),
                * upper = calloc(nvars, sizeof(double));
        double * lambda_lower = calloc(nrhs, sizeof(double)),
                * lambda_upper = calloc(nrhs, sizeof(double));

        read_doubles(stream, rhs, nrhs);
        read_doubles(stream, linear, nvars);

        read_doubles(stream, lower, nvars);
        read_doubles(stream, upper, nvars);

        read_doubles(stream, lambda_lower, nrhs);
        read_doubles(stream, lambda_upper, nrhs);

        alm_t * alm = alm_make(m, nrhs, rhs,
                             nvars, linear,
                             lower, upper,
                             lambda_lower, lambda_upper);
        free(rhs);
        free(linear);
        free(lower);
        free(upper);
        free(lambda_lower);
        free(lambda_upper);

        return alm;
}

#ifdef TEST_ALM
int main (int argc, char ** argv)
{
        sparse_matrix_init();
        int nthreads = 1;
        assert(argc > 1);
        FILE * instance = fopen(argv[1], "r");
        alm_t * alm = alm_read(instance);
        fclose(instance);
        if (argc > 2)
                nthreads = atoi(argv[2]);

        thread_pool_t * pool = NULL;
        if (nthreads >= 0)
                pool = thread_pool_init(nthreads);
        double * x = calloc(alm_nvars(alm), sizeof(double)),
                * y = calloc(alm_nrhs(alm), sizeof(double));
        alm_solve(alm, 1000, x, alm_nvars(alm),
                  y, alm_nrhs(alm),
                  stdout, NULL, pool);
        if (pool != NULL)
                thread_pool_free(pool);
        free(y);
        free(x);
        sparse_matrix_free(alm_matrix(alm));
        alm_free(alm);

        return 0;
}
#endif
