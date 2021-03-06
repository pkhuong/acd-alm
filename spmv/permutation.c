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
        uint32_t least;
        uint32_t initial_index;
};

static int compare_count_pair(const void * x, const void * y)
{
        const struct count_pair * p1 = x,
                * p2 = y;
        if (p1->count < p2->count) return -1;
        if (p1->count > p2->count) return 1;

        if (p1->least < p2->least) return -1;
        if (p1->least > p2->least) return 1;

        if (p1->initial_index < p2->initial_index) return -1;
        if (p1->initial_index > p2->initial_index) return 1;
        return 0;
}

static void fisher_yates(void * data, size_t nmemb, size_t size)
{
        void * temp = calloc(size, 1);
        for (size_t i = 0, offset = 0; i < nmemb; i++, offset += size) {
                size_t j = (1.0*random()/RAND_MAX)*(nmemb-i);
                j += i;
                void * current = (char*)data+offset,
                        * pick = (char*)data+(j*size);
                memcpy(temp, current, size);
                memcpy(current, pick, size);
                memcpy(pick, temp, size);
        }
        free(temp);
}

#ifndef PERMUTATION_SORTING_SCOPE
# define PERMUTATION_SORTING_SCOPE 256
#endif

void sparse_permutation_init(sparse_permutation_t * destination,
                             const sparse_matrix_t * matrix,
                             int row)
{
        assert(NULL == destination->idx);
        assert(NULL == destination->ridx);
        assert(0 == destination->n);

        size_t n = row?matrix->nrows:matrix->ncolumns;
        struct count_pair * pairs = calloc(n, sizeof(struct count_pair));

        for (size_t i = 0; i < n; i++) {
                pairs[i].initial_index = i;
                pairs[i].least = -1u;
        }
        {
                const uint32_t * src = row?matrix->rows:matrix->columns;
                const uint32_t * other = !row?matrix->rows:matrix->columns;
                size_t nnz = matrix->nnz;
                for (size_t i = 0; i < nnz; i++) {
                        uint32_t c = src[i];
                        uint32_t r = other[i];
                        assert(c < n);
                        pairs[c].count++;
#ifdef PERMUTATION_FIRST_NZ
                        if (r < pairs[c].least)
                                pairs[c].least = r;
#else
                        (void)r;
#endif
                }
        }

        if (PERMUTATION_SORTING_SCOPE) {
                struct count_pair * to_sort = pairs;
                for (size_t i = 0; i < n;) {
                        size_t m = PERMUTATION_SORTING_SCOPE;
                        if ((n - i) < m)
                                m = n-i;
                        qsort(to_sort, m, sizeof(struct count_pair),
                              compare_count_pair);
                        i += m;
                        to_sort += m;
                }
        }

#ifdef PERMUTATION_SHUFFLE
        fisher_yates(pairs, n/8, 8*sizeof(struct count_pair));
#else
        (void)fisher_yates;
#endif
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
