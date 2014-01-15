#ifndef SPMV_INTERNAL_H
#define SPMV_INTERNAL_H
#ifdef USE_OSKI
# include <oski/oski.h>
#endif

#ifndef PREFETCH_DISTANCE
# define PREFETCH_DISTANCE 0
#endif

#define PREFETCH_TYPE _MM_HINT_NTA

typedef struct sparse_permutation
{
        /* forward permute: dest[i] = src[idx[i]] */
        /* reverse permute: dest[i] = src[ridx[i]] */
        /* i = idx[ridx[i]] */
        /* i = ridx[idx[i]] */
        uint32_t * idx, * ridx;
        size_t n;
} sparse_permutation_t;

struct csr
{
        size_t nrows;
        uint32_t * rows_indices;
        uint32_t * columns;
        double * values;
};

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
        sparse_permutation_t row_permutation;
        sparse_permutation_t col_permutation;
};

struct matrix_entry {
        uint32_t column;
        uint32_t row;
        double value;
        uint64_t swizzled;
};

static inline int
compare_matrix_entries(const void * xp, const void * yp)
{
        const struct matrix_entry * x = xp,
                * y = yp;
        if (x->swizzled < y->swizzled)
                return -1;
        if (x->swizzled > y->swizzled)
                return 1;
        return 0;
}

void sparse_permutation_identity(sparse_permutation_t * destination,
                                 size_t n);
void sparse_permutation_init(sparse_permutation_t * destination,
                             const sparse_matrix_t * matrix,
                             int row); /* otherwise, column */
void sparse_permutation_clear(sparse_permutation_t *);

int sparse_permute_vector(double * dest, size_t n,
                          const sparse_permutation_t * permutation,
                          /* direction >= 0: forward, < 0: inverse */
                          const double * src, int direction);

#define SWAP(X, Y) do {                         \
                __typeof__(X) temp = (X);       \
                (X) = (Y);                      \
                (Y) = temp;                     \
        } while (0)

#endif
