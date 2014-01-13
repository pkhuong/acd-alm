static void compute_violation(struct vector * xv, approx_t approx,
                              thread_pool_t pool)
{
        size_t nvars = xv->n,
                nrows = xv->nviolation;

        assert(nvars == approx->nvars);
        assert(nrows == approx->nrhs);

        assert(0 == sparse_matrix_multiply(xv->violation, nrows,
                                           approx->matrix, xv->x, nvars, 0,
                                           pool));
        const v2d * rhs = (v2d*)approx->rhs;
        v2d * viol = (v2d*)xv->violation;
        size_t n = (nrows+1)/2;
        for (size_t i = 0; i < n; i++)
                viol[i] -= rhs[i];
        xv->violationp = 1;
}

static double value(approx_t approx, struct vector * xv, thread_pool_t pool)
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
                compute_violation(xv, approx, pool);

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
                     struct vector * xv, double * OUT_value,
                     thread_pool_t pool)
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
                compute_violation(xv, approx, pool);

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

static void gradient2(struct vector ** OUT_grad,
                      approx_t approx, struct vector ** OUT_scaled,
                      struct vector ** xv, double ** OUT_value,
                      thread_pool_t pool)
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
                                                     1, pool));
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