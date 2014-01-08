#include "approx.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <strings.h>

struct approx {
        size_t nrhs, nvar;
        sparse_matrix_t constraints;
        double * rhs;
        double * scale; /* \sum_i scale_i [(Ax-b)_i]^2 */
        double * linear; /* linear x + [LS] */
        
        double * lower, * upper; /* box */
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
                     const double * scale,
                     size_t nvar,
                     const double * linear,
                     const double * lower, const double * upper)
{
        assert(nrhs == sparse_matrix_nrows(constraints));
        assert(nvar == sparse_matrix_ncolumns(constraints));

        approx_t approx = calloc(1, sizeof(struct approx));
        approx->nrhs = nrhs;
        approx->nvar = nvar;
        approx->constraints = constraints;
        approx->rhs = copy_double(rhs, nrhs);
        approx->scale = copy_double_default(scale, nrhs, 1);
        approx->linear = copy_double_default(linear, nvar, 0);
        approx->lower = copy_double_default(lower, nvar, -HUGE_VAL);
        approx->upper = copy_double_default(upper, nvar, HUGE_VAL);

        return approx;
}

#define DEF(TYPE, FIELD)                                        \
        TYPE approx_##FIELD(approx_t approx)                    \
        {                                                       \
                return approx->FIELD;                           \
        }

DEF(sparse_matrix_t, constraints)
DEF(size_t, nrhs)
DEF(double *, rhs)
DEF(double *, scale)
DEF(size_t, nvar)
DEF(double *, linear)
DEF(double *, lower)
DEF(double *, upper)

#undef DEF

int approx_free(approx_t approx)
{
        free(approx->rhs);
        free(approx->scale);
        free(approx->linear);
        free(approx->lower);
        free(approx->upper);
        memset(approx, 0, sizeof(struct approx));
        free(approx);

        return 0;
}
