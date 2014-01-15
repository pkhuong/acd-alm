/** Parallel sparse matrix/dense vector multiplication.
 **/

#ifndef SPMV_H
#define SPMV_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "../thread_pool/thread_pool.h"

typedef struct sparse_matrix sparse_matrix_t;
/* Must be called once, before creating any sparse matrix */
void sparse_matrix_init();

/* Makes a sparse matrix from a coordinate representation (as a
 * triplet of arrays, instead of an array of triplets).  The entries
 * can be in any order, as long as they never repeat.  Will eventually
 * return NULL on error. */
sparse_matrix_t * sparse_matrix_make(size_t ncolumns, size_t nrows,
                                   size_t nnz,
                                   const uint32_t * rows,
                                   const uint32_t * columns,
                                   const double * values);
/* Free a sparse matrix; NULL are ignored.  Will eventually return
 * non-zero on error. */
int sparse_matrix_free(sparse_matrix_t *);

/* Read-only accessors for the matrix; there is no guarantee on the
 * order of the coordinates.
 */
size_t sparse_matrix_ncolumns(sparse_matrix_t *);
size_t sparse_matrix_nrows(sparse_matrix_t *);
size_t sparse_matrix_nnz(sparse_matrix_t *);
const uint32_t * sparse_matrix_rows(sparse_matrix_t *);
const uint32_t * sparse_matrix_columns(sparse_matrix_t *);
const double * sparse_matrix_values(sparse_matrix_t *);

/* Actual multiplication. OUT_y <- op(A) x.
 *  op is identity if transpose = 0, transpose otherwise.
 *
 * ny and nx are redundant and only used for error checking purposes.
 *
 * pool can be either a thread_pool or NULL for serial execution.
 *
 * if SWIZZLED_MULT is defined, the multiplication is a serial COO
 * loop in Z-order.  Otherwise, it USE_OSKI is defined, the matrix
 * goes through OSKI, after tuning for Ax and A'x.  Finally, the
 * default case is a unrolled-and-jammed parallel CSR/CSC loop.
 */
int sparse_matrix_multiply(double * OUT_y, size_t ny,
                           const sparse_matrix_t * a,
                           const double * x, size_t nx,
                           int transpose,
                           thread_pool_t * pool);
/* Multiply a pair of vectors. x and OUT_y are arrays of two pointers;
 * the arguments are otherwise as in the single-vector case above.
 * This is equivalent to calling sparse_matrix_multiply on OUT_y[0]
 * and x[0], and then OUT_y[1] and x[1], but more efficient (the
 * sparse matrix is only traversed once and fusing the loops helps
 * expose more memory-level parallelism).
 */
int sparse_matrix_multiply_2(double ** OUT_y, size_t ny,
                             const sparse_matrix_t * a,
                             const double ** x, size_t nx,
                             int transpose,
                             thread_pool_t * pool);

/* Read a sparse matrix in text format. The first three integers are #
 * rows, # columns, # nonzeros.
 *
 * Then follows # nonzeros triplets of: row, column, value.
 *
 * Returns NULL on failure.
 */
sparse_matrix_t * sparse_matrix_read(FILE * stream);
#endif
