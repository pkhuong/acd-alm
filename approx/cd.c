#include "approx.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include "../huge_alloc/huge_alloc.h"
#include "../spmv/spmv_internal.h"

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


static void fisher_yates(void * data, size_t nmemb, size_t size)
{
        void * temp = calloc(size, 1);
        for (size_t i = 0, offset = 0; i < nmemb; i++, offset += size) {
                size_t j = (1.0*random()/RAND_MAX)*(nmemb-i);
                j += i;
                void * current = (char*)data+offset,
                        * pick = (char*)data+(j*size);
                memcpy(temp, current, size);
                memcpy(current, pick, size);
                memcpy(pick, temp, size);
        }
        free(temp);
}

static void clamp(double * x, const approx_t * instance)
{
        const double * lower = instance->lower,
                * upper = instance->upper;
        for (size_t i = 0; i < instance->nvars; i++) {
                double xi = x[i],
                        li = lower[i],
                        ui = upper[i];
                x[i] = fmin(fmax(li, xi), ui);
        }
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

static double norm_pg(const double * g, size_t n,
                      const double * x,
                      const double * lower, const double * upper)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++) {
                double xi = x[i];
                double xp = xi-g[i];
                if (xp < lower[i]) xp = lower[i];
                if (xp > upper[i]) xp = upper[i];
                double d = xp-xi;
                acc += d*d;
        }
        return sqrt(acc);
}

static double gradient(double * OUT_g, double * OUT_violation,
                       const double * x,
                       const approx_t * instance)
{
        double value = 0;
        double * scaled = calloc(instance->nrhs, sizeof(double));
        sparse_matrix_multiply(OUT_violation, instance->nrhs,
                               instance->matrix,
                               x, instance->nvars, 0, NULL);
        for (size_t i = 0; i < instance->nrhs; i++) {
                double v = OUT_violation[i] -= instance->rhs[i];
                double w = instance->weight[i];
                scaled[i] = w*v;
                value += w*v*v;
        }
        value *= .5;
        sparse_matrix_multiply(OUT_g, instance->nvars,
                               instance->matrix,
                               scaled, instance->nrhs, 1, NULL);

        for (size_t i = 0; i < instance->nvars; i++) {
                double li = instance->linear[i];
                OUT_g[i] += li;
                value += li*x[i];
        }

        free(scaled);

        return value;
}

struct sparse_vector
{
        size_t nnz;
        uint32_t * indices;
        double * values;
};

static void spaxpy(double * y, double a, const struct sparse_vector * x)
{
        size_t nnz = x->nnz;
        const uint32_t * indices = x->indices;
        const double * values = x->values;

        for (size_t i = 0; i < nnz; i++)
                y[indices[i]] += a*values[i];
}

static void sparsify(struct sparse_vector * out,
                     const double * x, size_t n)
{
        size_t nnz = 0;
        uint32_t * indices = calloc(n, sizeof(uint32_t));
        double * values = calloc(n, sizeof(double));
        for (size_t i = 0; i < n; i++) {
                double xi = x[i];
                if (fabs(xi) < 1e-10) continue;
                indices[nnz] = i;
                values[nnz] = xi;
                nnz++;
        }
        indices = realloc(indices, nnz*sizeof(uint32_t));
        values = realloc(values, nnz*sizeof(double));
        out->nnz = nnz;
        out->indices = indices;
        out->values = values;
};

static void clear_sparse(struct sparse_vector * vector)
{
        free(vector->indices);
        free(vector->values);
        memset(vector, 0, sizeof(struct sparse_vector));
}

struct column_info
{
        double probability, inv_step, lo, hi;
        // [A' D(weight) A]_i
        struct sparse_vector qi;
        // to update viol = [Ax - b]
        struct sparse_vector ai;
};

static void column_info(struct column_info * OUT_info,
                        const approx_t * instance, uint32_t column)
{
        assert(column < instance->nvars);
        size_t nvars = instance->nvars, nrhs = instance->nrhs;
        double * dense = calloc(nrhs, sizeof(double));
        {
                const struct csr * csr = &instance->matrix->transpose;
                uint32_t begin = csr->rows_indices[column],
                        end = csr->rows_indices[column+1];
                for (size_t i = begin; i < end; i++)
                        dense[csr->columns[i]] += csr->values[i];
        }
        sparsify(&OUT_info->ai, dense, instance->nrhs);

        double norm = 0;
        for (size_t i = 0; i < nrhs; i++) {
                double d = dense[i], w = instance->weight[i];
                dense[i] = w*d;
                norm += w*d*d;
        }

        double * q = calloc(nvars, sizeof(double));
        sparse_matrix_multiply(q, instance->nvars,
                               instance->matrix,
                               dense, instance->nrhs, 1, NULL);
        sparsify(&OUT_info->qi, q, instance->nvars);

        OUT_info->probability = (norm*norm)+1e-2;
#ifdef FLAT_PROBABILITY
        OUT_info->probability = 1;
#endif

        OUT_info->inv_step = 1.0/norm;
        assert(OUT_info->inv_step > 0);
        OUT_info->lo = instance->lower[column];
        OUT_info->hi = instance->upper[column];

        free(q);
        free(dense);
}

static void clear_column(struct column_info * column)
{
        clear_sparse(&column->qi);
        clear_sparse(&column->ai);
        memset(column, 0, sizeof(struct column_info));
}

struct cd_state
{
        double * x;
        double * g;
        double * violation;
        size_t ncolumn;
        struct column_info * info;
};

static void init_cd_state(struct cd_state * state,
                          const approx_t * instance)
{
        size_t nvars = instance->nvars, nrhs = instance->nrhs;        
        state->x = calloc(nvars, sizeof(double));
        state->g = calloc(nvars, sizeof(double));
        state->violation = calloc(nrhs, sizeof(double));
        state->info = calloc(nvars, sizeof(struct column_info));
        state->ncolumn = nvars;

        double min_inv_step = HUGE_VAL;
        for (size_t i = 0; i < nvars; i++) {
                column_info(state->info+i, instance, i);
                min_inv_step = fmin(min_inv_step, state->info[i].inv_step);
        }
        for (size_t i = 0; i < nvars; i++)
                state->info[i].inv_step = min_inv_step;
}

static void clear_cd_state(struct cd_state * state)
{
        for (size_t i = 0; i < state->ncolumn; i++)
                clear_column(state->info+i);
        free(state->info);
        free(state->violation);
        free(state->g);
        free(state->x);
        memset(state, 0, sizeof(struct cd_state));
}

static double one_step(struct cd_state * state, uint32_t column,
                       double scale)
{
        assert(column < state->ncolumn);
        double xi = state->x[column], gi = state->g[column];
        if (gi == 0) return 0;
        struct column_info * info = state->info+column;
        double xp = xi - scale*(gi*info->inv_step);
        int clamped = 0;
        if (xp < info->lo) {
                clamped = 1;
                xp = info->lo;
        }
        if (xp > info->hi) {
                clamped = 1;
                xp = info->hi;
        }
        double step = xp-xi;
        if (step == 0) return 0;

        state->x[column] = xp;
        spaxpy(state->g, step, &info->qi);
        //spaxpy(state->violation, step, &info->ai);

        /* if (!clamped) { */
        /*         assert(fabs(state->g[column]) < 1e-4); */
        /*         state->g[column] = 0; */
        /* } */

        return step;
}

static double major_step(struct cd_state * state, approx_t * instance,
                         double offset, FILE * log,
                         double scale)
{
        clamp(state->x, instance);
        double value = gradient(state->g, state->violation,
                                state->x, instance);
        if (log != NULL)
                fprintf(log, "\tEntering: %12g %12g %12g\n",
                        value + offset,
                        norm_pg(state->g, state->ncolumn,
                                state->x, instance->lower, instance->upper),
                        norm_2(state->violation, instance->nrhs));

        size_t n = 0, maxn = 11*state->ncolumn;
        uint32_t * columns = calloc(maxn*state->ncolumn, sizeof(uint32_t));
        double total_p = 0;
        for (size_t i = 0; i < state->ncolumn; i++)
                total_p += state->info[i].probability;

        for (size_t i = 0; i < state->ncolumn; i++) {
                double p = state->info[i].probability;
                size_t count = ceil(10.0*state->ncolumn*p/total_p);
                for (size_t j = 0; j < count; j++) {
                        assert(n < maxn);
                        columns[n++] = i;
                }
        }

        fisher_yates(columns, n, sizeof(uint32_t));
        for (size_t i = 0; i < n; i++) {
                one_step(state, columns[i], scale);
        }

        free(columns);

        clamp(state->x, instance);
        value = gradient(state->g, state->violation,
                         state->x, instance);
        if (log != NULL)
                printf("\t\tLeaving:  %12g %12g %12g\n",
                       value + offset,
                       norm_pg(state->g, state->ncolumn,
                               state->x, instance->lower, instance->upper),
                       norm_2(state->violation, instance->nrhs));
        return value + offset;
}

int cd_solve(double * x, size_t n, approx_t * approx, size_t niter,
             double max_pg, double max_value, double min_delta,
             FILE * file, size_t period, double * OUT_diagnosis,
             double offset, thread_pool_t * pool)
{
        (void)min_delta;
        (void)pool;

        period /= 10;
        assert(n == approx->nvars);
        struct cd_state state;
        init_cd_state(&state, approx);
        memcpy(state.x, x, n*sizeof(double));

        int reason = 0;
        size_t i;
        for (i = 0; i < niter; i++) {
                int print = ((i == 0)
                             || (period && ((i+1)%period == 0))
                             || (i+1 == niter));
                if (print && (file != NULL)) {
                        fprintf(file, "\t%6zu ", i+1);
                }
                double z = major_step(&state, approx, offset,
                                      print?file:NULL,
                                      .5);
                double pg = norm_pg(state.g, state.ncolumn,
                                    state.x, approx->lower, approx->upper);
                if (z < max_value) {
                        reason = 1;
                        break;
                }
                if (pg < max_pg) {
                        reason = 2;
                        break;
                }
        }

        if (OUT_diagnosis != NULL) {
                OUT_diagnosis[0] = (gradient(state.g, state.violation,
                                             state.x, approx)
                                    + offset);
                OUT_diagnosis[1] = norm_2(state.g, state.ncolumn);
                OUT_diagnosis[2] = norm_pg(state.g, state.ncolumn,
                                           state.x, approx->lower,
                                           approx->upper);
                OUT_diagnosis[3] = 0;
                OUT_diagnosis[4] = i+1;
        }

        memcpy(x, state.x, n*sizeof(double));
        clear_cd_state(&state);
        return reason;
}
