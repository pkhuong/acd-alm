#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <xmmintrin.h>
#include "../huge_alloc/huge_alloc.h"
#include "../thread_pool/thread_pool.h"
#include "spmv_internal.h"

/* FIXME: actually break these up in compilation units. */

#include "spmv.swizzle.inc"
#ifdef USE_OSKI
# include "spmv.oski.inc"
#endif

void sparse_matrix_init()
{
#ifdef USE_OSKI
        oski_Init();
#endif
}

sparse_matrix_t * sparse_matrix_make(size_t ncolumns, size_t nrows,
                                     size_t nnz,
                                     const uint32_t * rows,
                                     const uint32_t * columns,
                                     const double * values,
                                     int permute)
{
        sparse_matrix_t * matrix = calloc(1, sizeof(sparse_matrix_t));
        matrix->ncolumns = ncolumns;
        matrix->nrows = nrows;
        matrix->nnz = nnz;
        matrix->rows = huge_calloc(nnz+PREFETCH_DISTANCE,
                                   sizeof(uint32_t));
        matrix->columns = huge_calloc(nnz+PREFETCH_DISTANCE,
                                      sizeof(uint32_t));
        matrix->values = huge_calloc(nnz+PREFETCH_DISTANCE,
                                     sizeof(double));

        memcpy(matrix->rows, rows, nnz*sizeof(uint32_t));
        memcpy(matrix->columns, columns, nnz*sizeof(uint32_t));
        memcpy(matrix->values, values, nnz*sizeof(double));

        if (permute) {
                sparse_permutation_init(&matrix->row_permutation,
                                        matrix, 1);
                sparse_permutation_init(&matrix->col_permutation,
                                        matrix, 0);
                const uint32_t * row_ridx = matrix->row_permutation.ridx,
                        * col_ridx = matrix->col_permutation.ridx;
                for (size_t i = 0; i < nnz; i++) {
                        matrix->rows[i] = row_ridx[matrix->rows[i]];
                        matrix->columns[i] = col_ridx[matrix->columns[i]];
                }
        } else {
                sparse_permutation_identity(&matrix->row_permutation,
                                            matrix->nrows);
                sparse_permutation_identity(&matrix->col_permutation,
                                            matrix->ncolumns);
        }

        csr_from_sparse_matrix(matrix, &matrix->matrix, 0);
        csr_from_sparse_matrix(matrix, &matrix->transpose, 1);
        sparse_matrix_swizzle(matrix);

#ifdef USE_OSKI
        matrix->oski_matrix
                = oski_CreateMatCSR((int32_t*)matrix->matrix.rows_indices,
                                    (int32_t*)matrix->matrix.columns,
                                    matrix->matrix.values,
                                    nrows, ncolumns,
                                    SHARE_INPUTMAT,
                                    1, INDEX_ZERO_BASED);
        oski_SetHintMatMult(matrix->oski_matrix, OP_NORMAL,
                            1, SYMBOLIC_VEC, 0, SYMBOLIC_VEC,
                            ALWAYS_TUNE_AGGRESSIVELY);
        oski_SetHintMatMult(matrix->oski_matrix, OP_TRANS,
                            1, SYMBOLIC_MULTIVEC, 0, SYMBOLIC_MULTIVEC,
                            ALWAYS_TUNE_AGGRESSIVELY);
        size_t n = ncolumns>nrows?ncolumns:nrows;
        matrix->flat_input = huge_calloc(n*2, sizeof(double));
        matrix->flat_result = huge_calloc(n*2, sizeof(double));
#endif
        return matrix;
}

sparse_matrix_t * sparse_matrix_copy(const sparse_matrix_t * src,
                                     int permute)
{
        return sparse_matrix_make(src->ncolumns, src->nrows, src->nnz,
                                  src->rows, src->columns, src->values,
                                  permute);
}

int sparse_matrix_free(sparse_matrix_t * matrix)
{
        if (matrix == NULL)
                return 0;
        huge_free(matrix->rows);
        huge_free(matrix->columns);
        huge_free(matrix->values);
        csr_clear(&matrix->matrix);
        csr_clear(&matrix->transpose);
#ifdef USE_OSKI
        oski_DestroyMat(matrix->oski_matrix);
        huge_free(matrix->flat_input);
        huge_free(matrix->flat_result);
#endif
        sparse_permutation_clear(&matrix->row_permutation);
        sparse_permutation_clear(&matrix->col_permutation);
        memset(matrix, 0, sizeof(sparse_matrix_t));
        free(matrix);

        return 0;
}

#define DEF(TYPE, FIELD)                        \
        TYPE sparse_matrix_##FIELD(sparse_matrix_t * matrix)      \
        {                                                       \
                return matrix->FIELD;                           \
        }

DEF(size_t, ncolumns)
DEF(size_t, nrows)
DEF(size_t, nnz)
DEF(const uint32_t *, rows)
DEF(const uint32_t *, columns)
DEF(const double *, values)

#undef DEF

int sparse_matrix_row_permute(sparse_matrix_t * matrix,
                              double * dest, size_t n,
                              const double * src, int direction)
{
        return sparse_permute_vector(dest, n,
                                     &matrix->row_permutation,
                                     src, direction);
}

int sparse_matrix_col_permute(sparse_matrix_t * matrix,
                              double * dest, size_t n,
                              const double * src, int direction)
{
        return sparse_permute_vector(dest, n,
                                     &matrix->col_permutation,
                                     src, direction);
}

int sparse_matrix_multiply(double * OUT_y, size_t ny,
                           const sparse_matrix_t * a,
                           const double * x, size_t nx,
                           int transpose, thread_pool_t * pool)
{
        (void)pool;
        size_t nrows = a->nrows,
                ncolumns = a->ncolumns;
        uint32_t * rows = a->rows,
                * columns = a->columns;
        if (transpose) {
                SWAP(nrows, ncolumns);
                SWAP(rows, columns);
        }

        assert(ny == nrows);
        assert(nx == ncolumns);
        (void)mult;
#ifdef SWIZZLED_MULT
        memset(OUT_y, 0, sizeof(double)*ny);
        mult(OUT_y, a->nnz, columns, rows, a->values, x);
#elif defined(USE_OSKI)
        mult_oski(OUT_y, ny, a->oski_matrix, x, nx, transpose);
#else
        {
                struct csr_mult_subrange_info info
                        = {.out = OUT_y,
                           .csr = transpose?&a->transpose:&a->matrix,
                           .x = x};
                thread_pool_for(pool, 0, info.csr->nrows, 16,
                                csr_mult_subrange, &info);
        }
#endif
        return 0;
}

int sparse_matrix_multiply_2(double ** OUT_y, size_t ny,
                             const sparse_matrix_t * a,
                             const double ** x, size_t nx,
                             int transpose, thread_pool_t * pool)
{
        (void)pool;
        size_t nrows = a->nrows,
                ncolumns = a->ncolumns;
        uint32_t * rows = a->rows,
                * columns = a->columns;
        if (transpose) {
                SWAP(nrows, ncolumns);
                SWAP(rows, columns);
        }

        assert(ny == nrows);
        assert(nx == ncolumns);
        (void)mult2;
#ifdef SWIZZLED_MULT
        memset(OUT_y[0], 0, sizeof(double)*ny);
        memset(OUT_y[1], 0, sizeof(double)*ny);
        mult2(OUT_y, a->nnz, columns, rows, a->values, x);
#elif defined(USE_OSKI)
        mult_oski2(OUT_y, ny, a->flat_result,
                   a->oski_matrix,
                   x, nx, a->flat_input,
                   transpose);
#else
        {
                struct csr_mult2_subrange_info info
                        = {.out = OUT_y,
                           .csr = transpose?&a->transpose:&a->matrix,
                           .x = x};
                thread_pool_for(pool, 0, info.csr->nrows, 16,
                                csr_mult2_subrange, &info);
        }
#endif
        return 0;
}

sparse_matrix_t * sparse_matrix_read(FILE * stream, int permute)
{
        size_t nrows, ncolumns, nnz;
        assert(stream != NULL);
        assert(3 == fscanf(stream, " %zu %zu %zu",
                           &nrows, &ncolumns, &nnz));

        uint32_t * rows = calloc(nnz, sizeof(uint32_t)),
                * columns = calloc(nnz, sizeof(uint32_t));
        double * values = calloc(nnz, sizeof(double));

        for (size_t i = 0; i < nnz; i++)
                assert(3 == fscanf(stream, " %u %u %lf",
                                   rows+i, columns+i, values+i));

        sparse_matrix_t * m = sparse_matrix_make(ncolumns, nrows,
                                                 nnz,
                                                 rows, columns, values,
                                                 permute);
        free(rows);
        free(columns);
        free(values);

        return m;
}

#ifdef TEST_SPMV
#include <stdlib.h>
void fill_random_matrix(double * matrix, size_t nrows, size_t ncolumns,
                        double density)
{
        memset(matrix, 0, nrows*ncolumns*sizeof(double));

        for(size_t i = 0; i < nrows*ncolumns; i++) {
                if (random()<density*RAND_MAX) {
                        matrix[i] = (random()&1)?-1.0:1.0;
                }
        }
}

void random_vector(double * vector, size_t n)
{
        for (size_t i = 0; i < n; i++)
                vector[i] = ((2.0*random())/RAND_MAX)-1;
}

sparse_matrix_t * sparsify_matrix(const double * matrix,
                                size_t nrows, size_t ncolumns)
{
        size_t total_size = nrows*ncolumns;
        uint32_t * rows = calloc(total_size, sizeof(uint32_t));
        uint32_t * columns = calloc(total_size, sizeof(uint32_t));
        double * values = calloc(total_size, sizeof(double));

        size_t nnz = 0;
        for (size_t i = 0, row = 0; row < nrows; row++) {
                for (size_t column = 0; column < ncolumns; column++, i++) {
                        double v = matrix[i];
                        if (v == 0) continue;
                        rows[nnz] = row;
                        columns[nnz] = column;
                        values[nnz] = v;
                        nnz++;
                }
        }

        sparse_matrix_t * m = sparse_matrix_make(ncolumns, nrows, nnz,
                                                 rows, columns, values,
                                                 0);
        free(rows);
        free(columns);
        free(values);

        return m;
}

void dense_mult(double * OUT_y,
                const double * matrix, size_t nrows, size_t ncolumns,
                const double * x)
{
        memset(OUT_y, 0, sizeof(double)*nrows);
        for (size_t i = 0, row = 0; row < nrows; row++)
                for (size_t column = 0; column < ncolumns; column++, i++)
                        OUT_y[row] += matrix[i]*x[column];
}

void dense_mult_t(double * OUT_y,
                  const double * matrix, size_t nrows, size_t ncolumns,
                  const double * x)
{
        memset(OUT_y, 0, sizeof(double)*ncolumns);
        for (size_t i = 0, row = 0; row < nrows; row++)
                for (size_t column = 0; column < ncolumns; column++, i++)
                        OUT_y[column] += matrix[i]*x[row];
}

double diff(const double * x, const double * y, size_t n)
{
        double acc = 0;
        for (size_t i = 0; i < n; i++) {
                double d = x[i]-y[i];
                acc += d*d;
        }
        return sqrt(acc);
}

void random_test(size_t ncolumns, size_t nrows, size_t repeat)
{
        double * dense = calloc(ncolumns*nrows, sizeof(double)),
                * x = calloc(ncolumns, sizeof(double)),
                * y1 = calloc(nrows, sizeof(double)),
                * y2 = calloc(nrows, sizeof(double)),
                * xt = calloc(nrows, sizeof(double)),
                * y1t = calloc(ncolumns, sizeof(double)),
                * y2t = calloc(ncolumns, sizeof(double));

        fill_random_matrix(dense, nrows, ncolumns, .5);
        sparse_matrix_t * m = sparsify_matrix(dense, nrows, ncolumns);

        for (size_t i = 0; i < repeat; i++) {
                random_vector(x, ncolumns);
                sparse_matrix_multiply(y1, nrows, m, x, ncolumns, 0,
                                       NULL);
                dense_mult(y2, dense, nrows, ncolumns, x);
                assert(diff(y1, y2, nrows) < 1e-4);

                random_vector(xt, nrows);
                sparse_matrix_multiply(y1t, ncolumns, m, xt, nrows, 1,
                                       NULL);
                dense_mult_t(y2t, dense, nrows, ncolumns, xt);
                assert(diff(y1t, y2t, ncolumns) < 1e-4);
        }
        free(dense);
        free(x);
        free(y1);
        free(y2);
        free(xt);
        free(y1t);
        free(y2t);
        sparse_matrix_free(m);
}

int main()
{
        sparse_matrix_init();
        for (size_t i = 0; i < 20; i++) {
                for (size_t j = 0; j < 20; j++) {
                        random_test(i, j, 10);
                }
        }
        return 0;
}
#endif
