#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include "../huge_alloc/huge_alloc.h"
#include "../thread_pool/thread_pool.h"
#include "spmv_internal.h"

struct push_vector
{
        void * block;
        size_t size;
        size_t total_size;
};

static int push_vector_grow(struct push_vector * vector,
                            size_t min_alloc)
{
        size_t page_mask = sysconf(_SC_PAGE_SIZE)-1;
        size_t size = 2*vector->total_size,
                min_size = vector->size+min_alloc;

        if (min_size < vector->size) return -1;
        if (size < min_size)
                size = min_size;
        size = (size+page_mask)&(~page_mask);
        if (size < min_size) return -1;

        void * new_block = huge_calloc(size, 1);
        if (new_block == NULL) return -2;

        memcpy(new_block, vector->block, vector->size);
        huge_free(vector->block);
        vector->block = new_block;
        vector->total_size = size;

        return 0;
}

#define MIN_ALIGNMENT 64ul

static void * push_vector_alloc(struct push_vector * vector, size_t alloc)
{
        size_t mask = MIN_ALIGNMENT-1;
        alloc = (alloc+mask)&(~mask);

        if (vector->size+alloc > vector->total_size)
                assert(!push_vector_grow(vector, alloc));
        
        void * ret = (char*)(vector->block)+vector->size;
        vector->size += alloc;
        assert(vector->size <= vector->total_size);
        return ret;
}

static size_t make_single_block(struct push_vector * vector, size_t start_row, size_t nrows,
                                const uint32_t * row_indices, const uint32_t * col,
                                const double * value)
{
        assert(nrows <= BLOCK_SIZE);
        size_t nnz = row_indices[start_row+nrows]-row_indices[start_row];

        struct matrix_entry * entries = calloc(nnz, sizeof(struct matrix_entry));

        size_t alloc = 0;
        for (size_t row = start_row; row < start_row+nrows; row++) {
                for (size_t j = row_indices[row];
                     j < row_indices[row+1]; j++) {
                        size_t column = col[j];
                        entries[alloc++]
                                = (struct matrix_entry)
                                {.column = column,
                                 .row = row,
                                 .value = value[j],
                                 .swizzled = (column<<32)|row};
                }
        }
        assert(nnz == alloc);

        qsort(entries, nnz, sizeof(struct matrix_entry),
              compare_matrix_entries);
        
        size_t col_alloc = 0;
        uint32_t * columns = calloc(nnz, sizeof(uint32_t));
        double * values = calloc(nnz*BLOCK_SIZE, sizeof(double));

        if (nnz > 0) {
                columns[col_alloc++] = entries[0].column;
                for (size_t i = 0; i < nnz; i++) {
                        size_t current = columns[col_alloc-1];
                        size_t col = entries[i].column, row = entries[i].row;
                        if (col != current) {
                                assert(col > current);
                                columns[col_alloc++] = col;
                        }
                        values[8*(col_alloc-1)+row-start_row]
                                = entries[i].value;
                }
        }

        free(entries);

        struct matrix_subblock * subblock 
                = push_vector_alloc(vector,
                                    (sizeof(struct matrix_subblock)
                                     + BLOCK_SIZE*sizeof(double)*col_alloc
                                     + sizeof(uint32_t)*col_alloc));
        uint32_t * indices = (uint32_t *)(subblock->values+col_alloc);
        subblock->nindices = col_alloc;
        subblock->start_row = start_row;
        subblock->nrows = nrows;
        memcpy(subblock->values, values, col_alloc*BLOCK_SIZE*sizeof(double));
        memcpy(indices, columns, col_alloc*sizeof(uint32_t));

        free(columns);
        free(values);

        return (char*)subblock-(char*)(vector->block);
}

int block_from_csr(const struct csr * csr, struct block_matrix * block)
{
        size_t nrow = csr->nrows;
        size_t nblock = (nrow+BLOCK_SIZE-1)/BLOCK_SIZE;

        block->nrows = nrow;
        block->nblocks = nblock;

        size_t * offsets
                = block->block_offsets
                = calloc(nblock, sizeof(size_t));
        struct push_vector alloc = {0, 0, 0};
        for (size_t count = 0, i = 0; i < nrow; i+=BLOCK_SIZE, count++) {
                size_t end = i + BLOCK_SIZE;
                if (end > nrow) end = nrow;
                offsets[count]
                        = make_single_block(&alloc,
                                            i, end-i,
                                            csr->rows_indices, csr->columns,
                                            csr->values);
        }

        block->blocks = alloc.block;

        return 0;
}

void block_clear(struct block_matrix * block)
{
        huge_free(block->blocks);
        free(block->block_offsets);
        memset(block, 0, sizeof(struct block_matrix));
}

typedef double __attribute__((vector_size(BLOCK_SIZE*8))) block_row_t;

static void mult_subblock(const struct matrix_subblock * block,
                          double * out, const double * x)
{
        size_t nindices = block->nindices;
        const uint32_t * indices = (const uint32_t *)(block->values+nindices);

        block_row_t acc = (block_row_t){0.0};
        for (size_t i = 0; i < nindices; i++) {
                block_row_t xi = (block_row_t){x[indices[i]]};
                
                acc += block->values[i]*xi;
        }

        uint32_t start = block->start_row;
        uint32_t nrows = block->nrows;
        if (nrows == BLOCK_SIZE) {
                *(block_row_t*)(out+start) = acc;
        } else {
                for (unsigned i = 0; i < nrows; i++, start++)
                        out[start] = acc[i];
        }
}

static void mult2_subblock(const struct matrix_subblock * block,
                           double ** out, const double ** x)
{
        size_t nindices = block->nindices;
        const uint32_t * indices = (const uint32_t *)(block->values+nindices);
        const double * x0 = x[0], * x1 = x[1];

        block_row_t acc0 = (block_row_t){0.0}, acc1 = acc0;
        for (size_t i = 0; i < nindices; i++) {
                unsigned col = indices[i];
                {
                        block_row_t xi = (block_row_t){x0[col]};
                        acc0 += block->values[i]*xi;
                }
                {
                        block_row_t xi = (block_row_t){x1[col]};
                        acc1 += block->values[i]*xi;
                }
        }

        uint32_t start = block->start_row;
        uint32_t nrows = block->nrows;
        double * out0 = out[0], * out1 = out[1];
        if (nrows == BLOCK_SIZE) {
                *(block_row_t*)(out0+start) = acc0;
                *(block_row_t*)(out1+start) = acc1;
        } else {
                for (unsigned i = 0; i < nrows; i++, start++) {
                        out0[start] = acc0[i];
                        out1[start] = acc1[i];
                }
        }
}
                           
#define GET_SUBBLOCK(MATRIX, INDEX) ((struct matrix_subblock*)          \
                                     ((char*)((MATRIX)->blocks)+(MATRIX)->block_offsets[INDEX]))

void block_mult_subrange_1(size_t from, size_t end,
                           struct block_mult_subrange_info * info,
                           size_t * OUT_begin, size_t * OUT_end)
{
        double * out = info->out;
        const struct block_matrix * matrix = info->matrix;
        const double * x = info->x;

        {
                size_t begin = GET_SUBBLOCK(matrix, from)->start_row;
                if (OUT_begin != NULL)
                        *OUT_begin = begin;
                if (end == from) {
                        if (OUT_end != NULL)
                                *OUT_end = *OUT_begin;
                        return;
                }
        }

        for (size_t i = from; i < end; i++)
                mult_subblock(GET_SUBBLOCK(matrix, i), out, x);

        if (OUT_end != NULL) {
                const struct matrix_subblock * block = GET_SUBBLOCK(matrix, end-1);
                *OUT_end = block->start_row+block->nrows;
        }
}

void block_mult_subrange(size_t from, size_t end, void * info, unsigned id)
{
        (void)id;
        block_mult_subrange_1(from, end, info, NULL, NULL);
}

void block_mult2_subrange_1(size_t from, size_t end,
                            struct block_mult2_subrange_info * info,
                            size_t * OUT_begin, size_t * OUT_end)
{
        double ** out = info->out;
        const struct block_matrix * matrix = info->matrix;
        const double ** x = info->x;

        {
                size_t begin = GET_SUBBLOCK(matrix, from)->start_row;
                if (OUT_begin != NULL)
                        *OUT_begin = begin;
                if (end == from) {
                        if (OUT_end != NULL)
                                *OUT_end = *OUT_begin;
                        return;
                }
        }

        for (size_t i = from; i < end; i++)
                mult2_subblock(GET_SUBBLOCK(matrix, i), out, x);

        if (OUT_end != NULL) {
                const struct matrix_subblock * block = GET_SUBBLOCK(matrix, end-1);
                *OUT_end = block->start_row+block->nrows;
        }
}

void block_mult2_subrange(size_t from, size_t end, void * info, unsigned id)
{
        (void)id;
        block_mult2_subrange_1(from, end, info, NULL, NULL);
}
