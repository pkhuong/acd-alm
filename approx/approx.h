#ifndef APPROX_H
#define APPROX_H
#include <stdint.h>
#include <stddef.h>
#include "../sparse-gemv/sparse-gemv.h"

typedef struct approx * approx_t;

approx_t approx_make(sparse_matrix_t constraints, /* Must remain alive */
                     size_t nrhs, const double * rhs,
                     const double * scale,
                     size_t nvar,
                     const double * linear,
                     const double * lower, const double * upper);

sparse_matrix_t approx_matrix(approx_t);
size_t approx_nrhs(approx_t);
double * approx_rhs(approx_t);
double * approx_scale(approx_t);
size_t approx_nvar(approx_t);
double * approx_linear(approx_t);
double * approx_lower(approx_t);
double * approx_upper(approx_t);

int approx_free(approx_t);
#endif
