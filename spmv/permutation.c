#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include "spmv_internal.h"

void sparse_permutation_identity(sparse_permutation_t * permutation,
                                 size_t n)
{
        assert(NULL == permutation->idx);
        assert(NULL == permutation->ridx);
        assert(0 == permutation->n);
        permutation->idx = calloc(n, sizeof(uint32_t));
        permutation->ridx = calloc(n, sizeof(uint32_t));
        permutation->n = n;
        for (size_t i = 0; i < n; i++)
                permutation->idx[i] = permutation->ridx[i] = i;
}

struct count_pair
{
        uint32_t count;
        uint32_t initial_index;
};

static int compare_count_pair(const void * x, const void * y)
{
        const struct count_pair * p1 = x,
                * p2 = y;
        if (p1->count < p2->count) return -1;
        if (p1->count > p2->count) return 1;

        if (p1->initial_index < p2->initial_index) return -1;
        if (p1->initial_index > p2->initial_index) return 1;
        return 0;
}

void sparse_permutation_init(sparse_permutation_t * destination,
                             const sparse_matrix_t * matrix,
                             int row)
{
        assert(NULL == destination->idx);
        assert(NULL == destination->ridx);
        assert(0 == destination->n);

        size_t n = row?matrix->nrows:matrix->ncolumns;
        struct count_pair * pairs = calloc(n, sizeof(struct count_pair));

        for (size_t i = 0; i < n; i++)
                pairs[i].initial_index = i;
        {
                const uint32_t * src = row?matrix->rows:matrix->columns;
                size_t nnz = matrix->nnz;
                for (size_t i = 0; i < nnz; i++) {
                        uint32_t c = src[i];
                        assert(c < n);
                        pairs[c].count++;
                }
        }

        qsort(pairs, n, sizeof(struct count_pair), compare_count_pair);

        destination->idx = calloc(n, sizeof(uint32_t));
        destination->ridx = calloc(n, sizeof(uint32_t));
        destination->n = n;

        for (size_t i = 0; i < n; i++) {
                uint32_t src = pairs[i].initial_index;
                destination->idx[i] = src;
                destination->ridx[src] = i;
        }

        free(pairs);
}

void sparse_permutation_clear(sparse_permutation_t * permutation)
{
        if (permutation == NULL) return;

        free(permutation->idx);
        free(permutation->ridx);
        memset(permutation, 0, sizeof(sparse_permutation_t));
}

int sparse_permute_vector(double * dest, size_t n,
                          const sparse_permutation_t * permutation,
                          const double * src, int direction)
{
        if (n != permutation->n) return 1;
        const uint32_t * idx = ((direction >= 0)
                                ? permutation->idx
                                : permutation->ridx);
        for (size_t i = 0; i < n; i++)
                dest[i] = src[idx[i]];

        return 0;
}
