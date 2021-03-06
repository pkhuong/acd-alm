#ifndef SPMV_BLOCK_H
#define SPMV_BLOCK_H
#ifndef SPMV_BLOCK_SIZE
# define SPMV_BLOCK_SIZE 4
#endif

struct sparse_matrix;

struct matrix_subblock
{
        uint32_t nnz;
        uint32_t start_row;
        uint32_t nrows; /* actual # to write */
        double __attribute__((vector_size(SPMV_BLOCK_SIZE*8),
                              aligned(SPMV_BLOCK_SIZE*8)))
               values[];
};

struct block_matrix
{
        size_t nrows;
        size_t nblocks;
        void * blocks;
        size_t * block_offsets;
};

int block_from_csr(const struct csr * csr,
                   struct block_matrix * block);
void block_clear(struct block_matrix * block);

struct block_mult_subrange_info
{
        double * out;
        const struct block_matrix * matrix;
        const double * x;
};

void block_mult_subrange_1(size_t from, size_t end,
                           struct block_mult_subrange_info * info, 
                           size_t * OUT_begin, size_t * OUT_end);

void block_mult_subrange(size_t from, size_t end, void * info, 
                         unsigned id);

struct block_mult2_subrange_info
{
        double ** out;
        const struct block_matrix * matrix;
        const double ** x;
};

void block_mult2_subrange_1(size_t from, size_t end,
                            struct block_mult2_subrange_info * info, 
                            size_t * OUT_begin, size_t * OUT_end);

void block_mult2_subrange(size_t from, size_t end, void * info, 
                          unsigned id);
#endif
