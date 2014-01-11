#ifndef ALM_H
#define ALM_H
#include "spmv/spmv.h"
#include "approx/approx.h"
#include <stddef.h>
#include <stdio.h>

typedef struct alm * alm_t;

alm_t alm_make(sparse_matrix_t constraints,
               size_t nrhs, const double * rhs,
               size_t nvars, const double * linear,
               const double * lower, const double * upper,
               const double * lambda_lower, const double * lambda_upper);

sparse_matrix_t alm_matrix(alm_t);
size_t alm_nrhs(alm_t);
double * alm_rhs(alm_t);
size_t alm_nvars(alm_t);
double * alm_linear(alm_t);
double * alm_lower(alm_t);
double * alm_upper(alm_t);
double * alm_lambda_lower(alm_t);
double * alm_lambda_upper(alm_t);

int alm_free(alm_t);

int alm_solve(alm_t, size_t niter,
              double * x, size_t nvars,
              double * lambda, size_t nconstraints,
              FILE * log, double * OUT_diagnosis);

alm_t alm_read(FILE * stream);
#endif
