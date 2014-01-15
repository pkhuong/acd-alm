#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <assert.h>

void * huge_calloc(size_t n, size_t size)
{
#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
        void * ptr = 0;
        size_t nbytes = n*size;
        if (nbytes/size != n) return NULL;
        /* 16 bytes for SSE slop */
        if (nbytes + 16 <= nbytes) return NULL;
        nbytes += 16;
        int ret = posix_memalign(&ptr, 64, nbytes); /* align to cache */
        if (ret) return NULL;
        memset(ptr, 0, nbytes);
        return ptr;
#else
        /* This *should* align to 16 bytes (because of SIMD ops) */
        size_t slop = 1;
        if (size < 16)
                slop = (16+size-1)/size;
        if (n + slop < n) return NULL;
        return calloc(n+slop, size);
#endif
}

void huge_free(void * ptr)
{
        free(ptr);
}
