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
