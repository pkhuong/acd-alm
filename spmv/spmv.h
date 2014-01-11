#ifndef SPMV_H
#define SPMV_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef struct sparse_matrix * sparse_matrix_t;

sparse_matrix_t sparse_matrix_make(size_t ncolumns, size_t nrows,
                                   size_t nnz,
                                   const uint32_t * rows,
                                   const uint32_t * columns,
                                   const double * values);

size_t sparse_matrix_ncolumns(sparse_matrix_t);
size_t sparse_matrix_nrows(sparse_matrix_t);
size_t sparse_matrix_nnz(sparse_matrix_t);
const uint32_t * sparse_matrix_rows(sparse_matrix_t);
const uint32_t * sparse_matrix_columns(sparse_matrix_t);
const double * sparse_matrix_values(sparse_matrix_t);

int sparse_matrix_free(sparse_matrix_t);

int sparse_matrix_multiply(double * OUT_y, size_t ny,
                           const sparse_matrix_t a,
                           const double * x, size_t nx,
                           int transpose);
int sparse_matrix_multiply_2(double ** OUT_y, size_t ny,
                             const sparse_matrix_t a,
                             const double ** x, size_t nx,
                             int transpose);

sparse_matrix_t sparse_matrix_read(FILE * stream);
#endif
