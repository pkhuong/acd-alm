#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <assert.h>

void * huge_calloc(size_t n, size_t size)
{
#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
        void * ptr = 0;
        size_t nbytes = n*size+16; /* 16 bytes for SSE slop */
        int ret = posix_memalign(&ptr, 64, nbytes); /* align to cache */
        assert(0 == ret);
        memset(ptr, 0, nbytes);
        return ptr;
#else
        /* This *should* align to 16 bytes (because of SIMD ops) */
        return calloc(n+(16+size-1)/size, size);
#endif
}

void huge_free(void * ptr)
{
        free(ptr);
}
