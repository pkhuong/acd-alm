static void compute_violation(struct vector * xv, approx_t * approx,
                              thread_pool_t * pool)
{
        size_t nvars = xv->n,
                nrows = xv->nviolation;

        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);

        assert(0 == sparse_matrix_multiply(xv->violation, nrows,
                                           approx->matrix, xv->x, nvars, 0,
                                           pool));
        const v2d * rhs = (v2d*)approx->rhs,
                * weight = (v2d*)approx->weight;
        v2d * viol = (v2d*)xv->violation;
        v2d value = {0, 0};
        size_t n = (nrows+1)/2;
        for (size_t i = 0; i < n; i++) {
                v2d v = viol[i]-rhs[i];
                v2d scaled = weight[i]*v;
                viol[i] = scaled;
                value += scaled*v;
        }
        xv->quad_value = .5*(value[0]+value[1]);
        xv->violationp = 1;
}

static double value(approx_t * approx, struct vector * xv, thread_pool_t * pool)
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
        if (isnan(xv->quad_value))
#endif
                compute_violation(xv, approx, pool);

        return xv->value = xv->quad_value+dot(approx->linear, xv);
}

static void gradient(struct vector * OUT_grad,
                     approx_t * approx, struct vector * OUT_scaled,
                     struct vector * xv, double * OUT_value,
                     thread_pool_t * pool)
{
        (void)OUT_scaled;
        size_t nvars = OUT_grad->n,
                nrows = xv->nviolation;
        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);
        assert(nrows == OUT_scaled->n);
        assert(nvars == xv->n);

#ifndef NO_CACHING
        if (!xv->violationp)
#endif
                compute_violation(xv, approx, pool);

        assert(0 == sparse_matrix_multiply(OUT_grad->x, nvars,
                                           approx->matrix,
                                           xv->violation, nrows,
                                           1, pool));

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

struct increment_linear_term_info
{
        v2d * grad[2];
        const v2d * linear;
};

static void increment_linear_term_1 (size_t begin, size_t end, void * thunk,
                                     unsigned id)
{
        (void)id;
        struct increment_linear_term_info * info = thunk;
        v2d * g0 = info->grad[0],
                * g1 = info->grad[1];
        const v2d * linear = info->linear;
        for (size_t i = begin; i < end; i++) {
                v2d li = linear[i];
                g0[i] += li;
                g1[i] += li;
        }
}

static void gradient2(struct vector ** OUT_grad,
                      approx_t * approx, struct vector ** OUT_scaled,
                      struct vector ** xv, double ** OUT_value,
                      thread_pool_t * pool)
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
                        compute_violation(xv[i], approx, pool);
        }

        {
                double * grad[2] = {OUT_grad[0]->x, OUT_grad[1]->x};
                const double * scaled[2] = {xv[0]->violation,
                                            xv[1]->violation};
                assert(0 == sparse_matrix_multiply_2(grad, nvars,
                                                     approx->matrix,
                                                     scaled, nrows,
                                                     1, pool));
        }

        {
                struct increment_linear_term_info info
                        = {.grad = {(v2d*)OUT_grad[0]->x,
                                    (v2d*)OUT_grad[1]->x},
                           .linear = (v2d*)approx->linear};
                thread_pool_for(pool, 0, (nvars+1)/2, 64,
                                increment_linear_term_1, &info);
        }

        for (size_t i = 0; i < 2; i++) {
                if (OUT_value[i] != NULL)
                        *OUT_value[i] += dot(approx->linear, xv[i]);
        }
}
