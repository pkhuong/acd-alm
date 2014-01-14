static void mult_oski(double * out, size_t nout,
                      oski_matrix_t matrix,
                      const double * x, size_t nx,
                      int transpose)
{
        oski_vecview_t x_view = oski_CreateVecView((double*)x, nx,
                                                   STRIDE_UNIT);
        oski_vecview_t y_view = oski_CreateVecView(out, nout,
                                                   STRIDE_UNIT);

        oski_MatMult(matrix, transpose?OP_TRANS:OP_NORMAL,
                     1, x_view, 0, y_view);
        oski_DestroyVecView(x_view);
        oski_DestroyVecView(y_view);
}

static void mult_oski2(double ** out, size_t nout, double * flat_out,
                       oski_matrix_t matrix,
                       const double ** x, size_t nx, double * flat_in,
                       int transpose)
{
        memcpy(flat_in, x[0], sizeof(double)*nx);
        memcpy(flat_in+nx, x[1], sizeof(double)*nx);

        oski_vecview_t x_view 
                = oski_CreateMultiVecView(flat_in, nx, 2,
                                          LAYOUT_COLMAJ, nx);
        oski_vecview_t y_view
                = oski_CreateMultiVecView(flat_out, nout, 2,
                                          LAYOUT_COLMAJ, nout);

        oski_MatMult(matrix, transpose?OP_TRANS:OP_NORMAL,
                     1, x_view, 0, y_view);
        oski_DestroyVecView(x_view);
        oski_DestroyVecView(y_view);

        memcpy(out[0], flat_out, sizeof(double)*nout);
        memcpy(out[1], flat_out+nout, sizeof(double)*nout);
}
