#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include "../huge_alloc/huge_alloc.h"
#include "../thread_pool/thread_pool.h"
#include "spmv_internal.h"


static int csr_init(struct csr * csr, size_t nrows, size_t nnz)
{
        csr->nrows = nrows;
        csr->rows_indices = calloc(nrows+1, sizeof(uint32_t));
        csr->columns = huge_calloc(nnz+PREFETCH_DISTANCE, sizeof(uint32_t));
        csr->values = huge_calloc(nnz+PREFETCH_DISTANCE, sizeof(double));
        return 0;
}

void csr_clear(struct csr * csr)
{
        free(csr->rows_indices);
        huge_free(csr->columns);
        huge_free(csr->values);
        memset(csr, 0, sizeof(struct csr));
}

int csr_from_sparse_matrix(struct sparse_matrix * matrix, struct csr * csr,
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

        csr_init(csr, transpose?ncolumns:nrows, nnz);

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

void csr_mult_subrange(size_t from, size_t end, void * info, 
                       unsigned id)
{
        (void)id;
        struct csr_mult_subrange_info * info_struct = info;
        double * out = info_struct->out;
        const struct csr * csr = info_struct->csr;
        const double * x = info_struct->x;

        struct csr subcsr = *csr;
        subcsr.nrows = end-from;
        subcsr.rows_indices = csr->rows_indices+from;
        mult_csr(out+from, &subcsr, x);
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

void csr_mult2_subrange(size_t from, size_t end, void * info, 
                        unsigned id)
{
        (void)id;
        struct csr_mult2_subrange_info * info_struct = info;
        double ** out = info_struct->out;
        const struct csr * csr = info_struct->csr;
        const double ** x = info_struct->x;

        double * out2[] = {out[0]+from, out[1]+from};
        struct csr subcsr = *csr;
        subcsr.nrows = end-from;
        subcsr.rows_indices = csr->rows_indices+from;
        mult_csr2(out2, &subcsr, x);
}

