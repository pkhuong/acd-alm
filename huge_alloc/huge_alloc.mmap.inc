#include <assert.h>
#include <stdint.h>
#include <unistd.h>

/* Colouring counter. Avoid pathological aliasing issues by offsetting
 * allocations by a small number of cache lines. */
static unsigned offset = 0;

void * huge_calloc(size_t n, size_t size)
{
        /* header + colouring */
        size_t padding = 2*sizeof(size_t) + 64*((offset++)%8); 
        padding = (padding+63ul)&(~63ul); /* round up to 64 bytes */
        assert(padding < (size_t)sysconf(_SC_PAGE_SIZE));
        size_t bytes = n*size+padding+16; /* 16 byte slop for SIMD */
        void * ret = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                          MAP_ANONYMOUS|MAP_HUGETLB|MAP_PRIVATE,
                          -1, 0);
        if (ret == MAP_FAILED)
                ret = mmap(NULL, bytes, PROT_READ|PROT_WRITE,
                           MAP_ANONYMOUS|MAP_PRIVATE,
                           -1, 0);
        assert(ret != MAP_FAILED);
        {
                size_t * header = ret;
                header[0] = bytes;
                header[1] = padding;
        }
        return (char*)ret+padding;
}

void huge_free(void * ptr)
{
        uintptr_t mask = (uintptr_t)sysconf(_SC_PAGE_SIZE)-1;
        if (ptr == NULL) return;
        size_t * header = (size_t*)((uintptr_t)ptr & (~mask));
        size_t bytes = header[0], padding = header[1];
        assert((char*)ptr == (char*)header+padding);

        munmap(header, bytes);
}
