#ifndef APPROX_H
#define APPROX_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "../spmv/spmv.h"

typedef struct approx * approx_t;

approx_t approx_make(sparse_matrix_t constraints, /* Must remain alive */
                     size_t nrhs, const double * rhs,
                     const double * weight,
                     size_t nvars,
                     const double * linear,
                     const double * lower, const double * upper);

sparse_matrix_t approx_matrix(approx_t);
size_t approx_nrhs(approx_t);
double * approx_rhs(approx_t);
double * approx_weight(approx_t);
size_t approx_nvars(approx_t);
double * approx_linear(approx_t);
double * approx_lower(approx_t);
double * approx_upper(approx_t);

int approx_free(approx_t);

int approx_update_step_sizes(approx_t); /* Call after mods to objective */

int approx_solve(double * x, size_t n, approx_t approx, size_t niter,
                 double max_pg, double max_value, double min_delta,
                 FILE * log, size_t period,
                 double * OUT_diagnosis /* NULL or double[5] */,
                 double offset);
#endif
