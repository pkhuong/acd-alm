#include <assert.h>
#include <stdint.h>

static unsigned offset = 0;

void * huge_calloc(size_t n, size_t size)
{
        size_t mask = (1ul<<21)-1;
        size_t padding = 64*(1+((offset++)%8));
        assert(2*sizeof(size_t) < 64); /* FIXME: static assert */
        size_t bytes = (n*size+padding+16+mask)&(~mask);
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
        uintptr_t mask = (1ul<<12)-1;
        if (ptr == NULL) return;
        size_t * header = (size_t*)((uintptr_t)ptr & (~mask));
        size_t bytes = header[0], padding = header[1];
        assert((char*)ptr == (char*)header+padding);

        munmap(header, bytes);
}
