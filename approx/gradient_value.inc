#include "../spmv/spmv_internal.h"

#ifndef APPROX_SPMV_CHUNK
# define APPROX_SPMV_CHUNK 128
#endif


struct compute_violation_state
{
        struct block_mult_subrange_info mult;
        const v2d * rhs, * weight;
        v2d * viol;
};

static double compute_violation_1(size_t begin, size_t end,
                                  void * info, unsigned id)
{
        (void)id;
        struct compute_violation_state * state = info;
        const v2d * rhs = state->rhs,
                * weight = state->weight;
        v2d * viol = state->viol;
        v2d value = {0, 0};

        int last = 0;
        while (!last) {
                size_t begin2 = begin;
                size_t end2 = begin += APPROX_SPMV_CHUNK;
                if ((end2 < begin2) || (end <= end2)) {
                        end2 = end;
                        last = 1;
                }
                block_mult_subrange_1(begin2, end2, &state->mult,
                                      &begin2, &end2);
                assert(0 == begin2%2);
                assert((end2 == state->mult.matrix->nrows) || (0 == end2%2));

                size_t n = (end2+1)/2;
                if (rhs != NULL) {
                        for (size_t i = begin2/2; i < n; i++) {
                                v2d v = viol[i]-rhs[i];
                                v2d scaled = weight[i]*v;
                                viol[i] = scaled;
                                value += scaled*v;
                        }
                } else {
                        for (size_t i = begin2/2; i < n; i++) {
                                v2d v = viol[i];
                                v2d scaled = weight[i]*v;
                                viol[i] = scaled;
                        }
                }
        }
        return value[0]+value[1];
}

static void compute_violation(struct vector * xv, approx_t * approx,
                              thread_pool_t * pool)
{
        size_t nvars = xv->n,
                nrows = xv->nviolation;

        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);

        struct compute_violation_state state
                = {.mult = {.out = xv->violation,
                            .matrix = &approx->matrix->block,
                            .x = xv->x},
                   .rhs = (v2d*)approx->rhs,
                   .weight = (v2d*)approx->weight,
                   .viol = (v2d*)xv->violation};

        xv->quad_value = .5*thread_pool_map_reduce(pool,
                                                   0, approx->matrix->block.nblocks, 8,
                                                   compute_violation_1,
                                                   &state,
                                                   THREAD_POOL_REDUCE_SUM,
                                                   0);
        xv->violationp = 1;
}

static void compute_wax(struct vector * xv, approx_t * approx,
                        thread_pool_t * pool)
{
        size_t nvars = xv->n,
                nrows = xv->nviolation;

        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);

        struct compute_violation_state state
                = {.mult = {.out = xv->violation,
                            .matrix = &approx->matrix->block,
                            .x = xv->x},
                   .rhs = NULL,
                   .weight = (v2d*)approx->weight,
                   .viol = (v2d*)xv->violation};

        thread_pool_map_reduce(pool,
                               0, approx->matrix->block.nblocks, 8,
                               compute_violation_1,
                               &state,
                               THREAD_POOL_REDUCE_SUM,
                               0);

        xv->quad_value = nan("");
        xv->violationp = 1;
}

static double compute_value(approx_t * approx, struct vector * xv,
                            thread_pool_t * pool)
{
        size_t nvars = xv->n;
        size_t nrows = xv->nviolation;

        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);

        {
                double value = xv->value;
                if (!isnan(value)) return value;
        }

        double linear = dot(approx->linear, xv, pool);

#ifndef NO_CACHING
        if (isnan(xv->quad_value))
#endif
                compute_violation(xv, approx, pool);

        return xv->value = xv->quad_value+linear;
}

struct gradient_state
{
        struct block_mult_subrange_info mult;
        const v2d * linear;
        v2d * grad;
};

static void gradient_1(size_t begin, size_t end, void * info, unsigned id)
{
        (void)id;
        struct gradient_state * state = info;
        block_mult_subrange_1(begin, end, &state->mult,
                              &begin, &end);
        assert(0 == begin%2);
        assert((end == state->mult.matrix->nrows) || (0 == end%2));

        const v2d * linear = state->linear;
        v2d * grad = state->grad;
        size_t n = (end+1)/2;
        for (size_t i = begin/2; i < n; i++)
                grad[i] += linear[i];
}

static void gradient(struct vector * OUT_grad, approx_t * approx,
                     struct vector * xv, thread_pool_t * pool)
{
        size_t nvars = OUT_grad->n,
                nrows = xv->nviolation;
        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);
        assert(nvars == xv->n);

#ifndef NO_CACHING
        if (!xv->violationp)
#endif
                compute_violation(xv, approx, pool);

        struct gradient_state state
                = {.mult = {.out = OUT_grad->x,
                            .matrix = &approx->matrix->block_transpose,
                            .x = xv->violation},
                   .linear = (v2d*)approx->linear,
                   .grad = (v2d*)OUT_grad->x};

        thread_pool_for(pool, 0, approx->matrix->block_transpose.nblocks, 8,
                        gradient_1, &state);
}

/* A'd(W)Ax */
static void curve_gradient(struct vector * OUT_curved, approx_t * approx,
                           struct vector * xv, thread_pool_t * pool)
{
        size_t nvars = OUT_curved->n,
                nrows = xv->nviolation;
        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);
        assert(nvars == xv->n);

        compute_wax(xv, approx, pool);
        int ret = sparse_matrix_multiply(OUT_curved->x, nvars,
                                         approx->matrix,
                                         xv->violation, xv->nviolation,
                                         1, pool);
        assert(!ret);
}

struct gradient2_state
{
        struct block_mult2_subrange_info mult;
        const v2d * linear;
        v2d * grad[2];
};

static void gradient2_1(size_t begin, size_t end, void * info, unsigned id)
{
        (void)id;
        struct gradient2_state * state = info;
        const v2d * linear = state->linear;
        v2d * g0 = state->grad[0], * g1 = state->grad[1];

        int last = 0;
        while (!last) {
                size_t begin2 = begin;
                size_t end2 = begin += APPROX_SPMV_CHUNK;
                if ((end2 < begin2) || (end <= end2)) {
                        end2 = end;
                        last = 1;
                }
                block_mult2_subrange_1(begin2, end2, &state->mult,
                                       &begin2, &end2);
                assert(0 == begin2%2);
                assert((end2 == state->mult.matrix->nrows) || (0 == end2%2));

                size_t n = (end2+1)/2;
                for (size_t i = begin2/2; i < n; i++) {
                        v2d li = linear[i];
                        g0[i] += li;
                        g1[i] += li;
                }
        }
}

static void gradient2(struct vector ** OUT_grad, approx_t * approx,
                      struct vector ** xv, thread_pool_t * pool)
{
        size_t nvars = OUT_grad[0]->n,
                nrows = xv[0]->nviolation;
        assert(OUT_grad[1]->n == nvars);
        assert(xv[1]->nviolation == nrows);
        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);
        for (size_t i = 0; i < 2; i++)
                assert(nvars == xv[i]->n);

        for (size_t i = 0; i < 2; i++) {
#ifndef NO_CACHING
                if (!xv[i]->violationp)
#endif
                        compute_violation(xv[i], approx, pool);
        }

        double * grad[2] = {OUT_grad[0]->x, OUT_grad[1]->x};
        const double * x[2] = {xv[0]->violation, xv[1]->violation};
        struct gradient2_state state
                = {.mult = {.out = grad,
                            .matrix = &approx->matrix->block_transpose,
                            .x = x},
                   .linear = (v2d*)approx->linear,
                   .grad = {(v2d*)grad[0], (v2d*)grad[1]}};

        thread_pool_for(pool,
                        0, approx->matrix->block_transpose.nblocks, 8,
                        gradient2_1, &state);
}
