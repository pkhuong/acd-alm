#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <xmmintrin.h>

#ifdef USE_OSKI
# include <oski/oski.h>
#endif

void sparse_matrix_init()
{
#ifdef USE_OSKI
        oski_Init();
#endif
}

#define SWAP(X, Y) do {                         \
                __typeof__(X) temp = (X);       \
                (X) = (Y);                      \
                (Y) = temp;                     \
        } while (0)

struct csr
{
        size_t nrows;
        uint32_t * rows_indices;
        uint32_t * columns;
        double * values;
};

#ifndef PREFETCH_DISTANCE
# define PREFETCH_DISTANCE 0
#endif

static int init_csr(struct csr * csr, size_t nrows, size_t nnz)
{
        csr->nrows = nrows;
        csr->rows_indices = calloc(nrows+1, sizeof(uint32_t));
        csr->columns = calloc(nnz+PREFETCH_DISTANCE, sizeof(uint32_t));
        csr->values = calloc(nnz+PREFETCH_DISTANCE, sizeof(double));
        return 0;
}

static void free_csr(struct csr * csr)
{
        free(csr->rows_indices);
        free(csr->columns);
        free(csr->values);
        memset(csr, 0, sizeof(struct csr));
}

struct sparse_matrix
{
        size_t ncolumns, nrows, nnz;
        uint32_t * rows, * columns;
        double * values;
        struct csr matrix;
        struct csr transpose;
#ifdef USE_OSKI
        oski_matrix_t oski_matrix;
        double * flat_input;
        double * flat_result;
#endif
};

#define DEF(TYPE, FIELD)                        \
        TYPE sparse_matrix_##FIELD(sparse_matrix_t matrix)      \
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

struct matrix_entry {
        uint32_t column;
        uint32_t row;
        double value;
        uint64_t swizzled;
};

static uint64_t swizzle(uint32_t x, uint32_t y)
{
        uint64_t acc = 0;
        for (unsigned i = 0; i < 32; i++) {
                uint64_t mask = 1ull<<i;
                uint64_t xi = (x&mask)>>i;
                uint64_t yi = (y&mask)>>i;
                acc |= xi << (2*i);
                acc |= yi << (2*i+1);
        }
        return acc;
}

static int compare_matrix_entries(const void * xp, const void * yp)
{
        const struct matrix_entry * x = xp,
                * y = yp;
        if (x->swizzled < y->swizzled)
                return -1;
        if (x->swizzled > y->swizzled)
                return 1;
        return 0;
}

static int sparse_matrix_csr(sparse_matrix_t matrix, struct csr * csr,
                             int transpose)
{
        uint32_t * columns = matrix->columns;
        uint32_t * rows = matrix->rows;
        double * values = matrix->values;

        const size_t nnz = matrix->nnz,
                ncolumns = matrix->ncolumns,
                nrows = matrix->nrows;

        /* QUite possible the stupidest way to construct a CSR matrix */
        struct matrix_entry * entries
                = calloc(nnz, sizeof(struct matrix_entry));
        for (size_t i = 0; i < nnz; i++) {
                uint32_t column = columns[i],
                        row = rows[i];
                assert(column < ncolumns);
                assert(row < nrows);
                if (transpose)
                        SWAP(row, column);
                entries[i] = (struct matrix_entry)
                        {.column = column,
                         .row = row,
                         .value = values[i],
                         .swizzled = ((uint64_t)row<<32
                                      | (uint64_t)column)};
        }

        qsort(entries,
              nnz, sizeof(struct matrix_entry),
              compare_matrix_entries);

        init_csr(csr, transpose?ncolumns:nrows, nnz);

        for (size_t i = 0; i < nnz; i++) {
                struct matrix_entry * entry = entries+i;
                csr->rows_indices[entry->row+1]=i+1;
                csr->columns[i] = entry->column;
                csr->values[i] = entry->value;
        }

        size_t max = 0;
        size_t n = transpose?ncolumns:nrows;
        for (size_t i = 0; i <= n; i++) {
                if (csr->rows_indices[i] > max)
                        max = csr->rows_indices[i];
                csr->rows_indices[i] = max;
        }

        free(entries);

        return 0;
}

static int sparse_matrix_swizzle(sparse_matrix_t matrix)
{
        uint32_t * columns = matrix->columns;
        uint32_t * rows = matrix->rows;
        double * values = matrix->values;

        const size_t nnz = matrix->nnz,
                ncolumns = matrix->ncolumns,
                nrows = matrix->nrows;

        struct matrix_entry * entries
                = calloc(nnz, sizeof(struct matrix_entry));
        for (size_t i = 0; i < nnz; i++) {
                uint32_t column = columns[i],
                        row = rows[i];
                assert(column < ncolumns);
                assert(row < nrows);
                entries[i] = (struct matrix_entry){.column = column,
                                                   .row = row,
                                                   .value = values[i],
                                                   .swizzled = swizzle(column,
                                                                       row)};
        }

        qsort(entries,
              nnz, sizeof(struct matrix_entry),
              compare_matrix_entries);

        for (size_t i = 0; i < nnz; i++) {
                struct matrix_entry * entry = entries+i;
                columns[i] = entry->column;
                rows[i] = entry->row;
                values[i] = entry->value;
        }

        free(entries);

        return 0;
}

sparse_matrix_t sparse_matrix_make(size_t ncolumns, size_t nrows,
                                   size_t nnz,
                                   const uint32_t * rows,
                                   const uint32_t * columns,
                                   const double * values)
{
        sparse_matrix_t matrix = calloc(1, sizeof(struct sparse_matrix));
        matrix->ncolumns = ncolumns;
        matrix->nrows = nrows;
        matrix->nnz = nnz;
        matrix->rows = calloc(nnz+PREFETCH_DISTANCE, sizeof(uint32_t));
        matrix->columns = calloc(nnz+PREFETCH_DISTANCE, sizeof(uint32_t));
        matrix->values = calloc(nnz+PREFETCH_DISTANCE, sizeof(double));

        memcpy(matrix->rows, rows, nnz*sizeof(uint32_t));
        memcpy(matrix->columns, columns, nnz*sizeof(uint32_t));
        memcpy(matrix->values, values, nnz*sizeof(double));

        sparse_matrix_csr(matrix, &matrix->matrix, 0);
        sparse_matrix_csr(matrix, &matrix->transpose, 1);
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
        matrix->flat_input = calloc(n*2, sizeof(double));
        matrix->flat_result = calloc(n*2, sizeof(double));
#endif

        return matrix;
}

int sparse_matrix_free(sparse_matrix_t matrix)
{
        free(matrix->rows);
        free(matrix->columns);
        free(matrix->values);
        free_csr(&matrix->matrix);
        free_csr(&matrix->transpose);
#ifdef USE_OSKI
        oski_DestroyMat(matrix->oski_matrix);
        free(matrix->flat_input);
        free(matrix->flat_result);
#endif
        memset(matrix, 0, sizeof(struct sparse_matrix));
        free(matrix);

        return 0;
}

#define PREFETCH_TYPE _MM_HINT_NTA

static void mult(double * out,
                 size_t nnz,
                 const uint32_t * columns, const uint32_t * rows,
                 const double * values,
                 const double * x)
{
        for (size_t i = 0; i < nnz; i++) {
                uint32_t col = columns[i],
                        row = rows[i];
                double ax = values[i]*x[col];
                out[row] += ax;
#if PREFETCH_DISTANCE
                _mm_prefetch(x+columns[i+PREFETCH_DISTANCE], PREFETCH_TYPE);
                _mm_prefetch(out+rows[i+PREFETCH_DISTANCE], PREFETCH_TYPE);
#endif
        }
}

static void mult2(double ** out,
                 size_t nnz,
                 const uint32_t * columns, const uint32_t * rows,
                 const double * values,
                 const double ** x)
{
        double * out0 = out[0], * out1 = out[1];
        const double * x0 = x[0], * x1 = x[1];
        for (size_t i = 0; i < nnz; i++) {
                uint32_t col = columns[i],
                        row = rows[i];
                double v = values[i];
                out0[row] += v*x0[col];
                out1[row] += v*x1[col];
#if PREFETCH_DISTANCE
                _mm_prefetch(x0+columns[i+PREFETCH_DISTANCE], PREFETCH_TYPE);
                _mm_prefetch(x1+columns[i+PREFETCH_DISTANCE], PREFETCH_TYPE);
                _mm_prefetch(out0+rows[i+PREFETCH_DISTANCE], PREFETCH_TYPE);
                _mm_prefetch(out1+rows[i+PREFETCH_DISTANCE], PREFETCH_TYPE);
#endif
        }
}

static void mult_csr(double * out, struct csr * csr, const double * x)
{
        size_t nrows = csr->nrows;
        const uint32_t * rows_indices = csr->rows_indices,
                * columns = csr->columns;
        const double * values = csr->values;
        size_t ncoupled_rows = nrows&(~1ull);
        size_t i;
        for (i = 0; i < ncoupled_rows; i+=2) {
                uint32_t begin = rows_indices[i],
                        middle = rows_indices[i+1],
                        end = rows_indices[i+2];
                double acc0 = 0, acc1 = 0;
                uint32_t n0 = middle-begin, n1 = end-middle;
                const double * v0 = values+begin,
                        * v1 = values+middle;
                const uint32_t * c0 = columns+begin,
                        * c1 = columns+middle;
                uint32_t j;
                if (n0 <= n1) {
                        for (j = 0; j < n0; j++) {
                                acc0 += v0[j]*x[c0[j]];
                                acc1 += v1[j]*x[c1[j]];
                        }
                        out[i] = acc0;
                        for (; j < n1; j++)
                                acc1 += v1[j]*x[c1[j]];
                        out[i+1] = acc1;
                } else {
                        for (j = 0; j < n1; j++) {
                                acc0 += v0[j]*x[c0[j]];
                                acc1 += v1[j]*x[c1[j]];
                        }
                        out[i+1] = acc1;
                        for (; j < n0; j++)
                                acc0 += v0[j]*x[c0[j]];
                        out[i] = acc0;
                }
        }
        if (i < nrows) {
                uint32_t begin = rows_indices[i],
                        end = rows_indices[i+1];
                double acc = 0;
                for (; begin < end; begin++)
                        acc += values[begin]*x[columns[begin]];
                out[i] = acc;
        }
}

static void mult_csr2(double ** out, struct csr * csr, const double ** x)
{
        size_t nrows = csr->nrows;
        const uint32_t * rows_indices = csr->rows_indices,
                * columns = csr->columns;
        const double * values = csr->values;
        double * out0 = out[0], * out1 = out[1];
        const double * x0 = x[0], * x1 = x[1];
        size_t ncoupled_rows = nrows&(~1ull);
        size_t i;
        for (i = 0; i < ncoupled_rows; i+=2) {
                uint32_t begin = rows_indices[i],
                        middle = rows_indices[i+1],
                        end = rows_indices[i+2];
                double acc0_0 = 0, acc1_0 = 0; /* x0 */
                double acc0_1 = 0, acc1_1 = 0; /* x1 */
                uint32_t n0 = middle-begin, n1 = end-middle;
                const double * v0 = values+begin,
                        * v1 = values+middle;
                const uint32_t * c0 = columns+begin,
                        * c1 = columns+middle;
                uint32_t j;
                if (n0 <= n1) {
                        for (j = 0; j < n0; j++) {
                                acc0_0 += v0[j]*x0[c0[j]];
                                acc1_0 += v1[j]*x0[c1[j]];
                                acc0_1 += v0[j]*x1[c0[j]];
                                acc1_1 += v1[j]*x1[c1[j]];
                        }
                        out0[i] = acc0_0;
                        out1[i] = acc0_1;
                        for (; j < n1; j++) {
                                acc1_0 += v1[j]*x0[c1[j]];
                                acc1_1 += v1[j]*x1[c1[j]];
                        }
                        out0[i+1] = acc1_0;
                        out1[i+1] = acc1_1;
                } else {
                        for (j = 0; j < n1; j++) {
                                acc0_0 += v0[j]*x0[c0[j]];
                                acc1_0 += v1[j]*x0[c1[j]];
                                acc0_1 += v0[j]*x1[c0[j]];
                                acc1_1 += v1[j]*x1[c1[j]];
                        }
                        out0[i+1] = acc1_0;
                        out1[i+1] = acc1_1;
                        for (; j < n0; j++) {
                                acc0_0 += v0[j]*x0[c0[j]];
                                acc0_1 += v0[j]*x1[c0[j]];
                        }
                        out0[i] = acc0_0;
                        out1[i] = acc0_1;
                }
        }

        if (i < nrows) {
                uint32_t begin = rows_indices[i],
                        end = rows_indices[i+1];
                double acc0 = 0, acc1 = 0;
                for (; begin < end; begin++) {
                        double v = values[begin];
                        uint32_t col = columns[begin];
                        acc0 += v*x0[col];
                        acc1 += v*x1[col];
                }
                out0[i] = acc0;
                out1[i] = acc1;
        }
}

#ifdef USE_OSKI
static void mult_oski(double * out, size_t nout,
                      oski_matrix_t matrix,
                      const double * x, size_t nx,
                      int transpose)
{
        oski_vecview_t x_view = oski_CreateVecView((double*)x, nx,
                                                   STRIDE_UNIT);
        oski_vecview_t y_view = oski_CreateVecView(out, nout,
                                                   STRIDE_UNIT);

        oski_MatMult(matrix, transpose?OP_TRANS:OP_NORMAL,
                     1, x_view, 0, y_view);
        oski_DestroyVecView(x_view);
        oski_DestroyVecView(y_view);
}

static void mult_oski2(double ** out, size_t nout, double * flat_out,
                       oski_matrix_t matrix,
                       const double ** x, size_t nx, double * flat_in,
                       int transpose)
{
        memcpy(flat_in, x[0], sizeof(double)*nx);
        memcpy(flat_in+nx, x[1], sizeof(double)*nx);

        oski_vecview_t x_view 
                = oski_CreateMultiVecView(flat_in, nx, 2,
                                          LAYOUT_COLMAJ, nx);
        oski_vecview_t y_view
                = oski_CreateMultiVecView(flat_out, nout, 2,
                                          LAYOUT_COLMAJ, nout);

        oski_MatMult(matrix, transpose?OP_TRANS:OP_NORMAL,
                     1, x_view, 0, y_view);
        oski_DestroyVecView(x_view);
        oski_DestroyVecView(y_view);

        memcpy(out[0], flat_out, sizeof(double)*nout);
        memcpy(out[1], flat_out+nout, sizeof(double)*nout);
}
#endif

int sparse_matrix_multiply(double * OUT_y, size_t ny,
                           const sparse_matrix_t a,
                           const double * x, size_t nx,
                           int transpose)
{
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
        (void)mult_csr;
        (void)mult;
#ifdef SWIZZLED_MULT
        memset(OUT_y, 0, sizeof(double)*ny);
        mult(OUT_y, a->nnz, columns, rows, a->values, x);
#elif defined(USE_OSKI)
        mult_oski(OUT_y, ny, a->oski_matrix, x, nx, transpose);
#else
        mult_csr(OUT_y, transpose?&a->transpose:&a->matrix, x);
#endif
        return 0;
}

int sparse_matrix_multiply_2(double ** OUT_y, size_t ny,
                            const sparse_matrix_t a,
                            const double ** x, size_t nx,
                            int transpose)
{
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
        (void)mult_csr2;
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
        mult_csr2(OUT_y, transpose?&a->transpose:&a->matrix, x);
#endif
        return 0;
}

#undef SWAP

sparse_matrix_t sparse_matrix_read(FILE * stream)
{
        size_t nrows, ncolumns, nnz;
        assert(3 == fscanf(stream, " %zu %zu %zu",
                           &nrows, &ncolumns, &nnz));

        uint32_t * rows = calloc(nnz, sizeof(uint32_t)),
                * columns = calloc(nnz, sizeof(uint32_t));
        double * values = calloc(nnz, sizeof(double));

        for (size_t i = 0; i < nnz; i++)
                assert(3 == fscanf(stream, " %u %u %lf",
                                   rows+i, columns+i, values+i));

        sparse_matrix_t m = sparse_matrix_make(ncolumns, nrows,
                                               nnz,
                                               rows, columns, values);
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

sparse_matrix_t sparsify_matrix(const double * matrix,
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

        sparse_matrix_t m = sparse_matrix_make(ncolumns, nrows, nnz,
                                               rows, columns, values);
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
        sparse_matrix_t m = sparsify_matrix(dense, nrows, ncolumns);

        for (size_t i = 0; i < repeat; i++) {
                random_vector(x, ncolumns);
                sparse_matrix_multiply(y1, nrows, m, x, ncolumns, 0);
                dense_mult(y2, dense, nrows, ncolumns, x);
                assert(diff(y1, y2, nrows) < 1e-4);

                random_vector(xt, nrows);
                sparse_matrix_multiply(y1t, ncolumns, m, xt, nrows, 1);
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