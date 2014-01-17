#ifndef SPMV_CSR_H
#define SPMV_CSR_H
struct sparse_matrix;

struct csr
{
        size_t nrows;
        uint32_t * rows_indices;
        uint32_t * columns;
        double * values;
};

int csr_from_sparse_matrix(struct sparse_matrix * matrix,
                           struct csr * csr,
                           int transpose);
void csr_clear(struct csr * csr);

struct csr_mult_subrange_info
{
        double * out;
        const struct csr * csr;
        const double * x;
};

void csr_mult_subrange(size_t from, size_t end, void * info, 
                       unsigned id);

struct csr_mult2_subrange_info
{
        double ** out;
        const struct csr * csr;
        const double ** x;
};

void csr_mult2_subrange(size_t from, size_t end, void * info, 
                        unsigned id);
#endif
