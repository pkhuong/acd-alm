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

void sparse_permutation_init(sparse_permutation_t * destination,
                             const sparse_matrix_t * matrix,
                             int row)
{
        (void)destination;
        (void)matrix;
        (void)row;
        assert(0);
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
        (void)dest;
        (void)n;
        (void)permutation;
        (void)src;
        (void)direction;
        assert(0);
        return 0;
}
