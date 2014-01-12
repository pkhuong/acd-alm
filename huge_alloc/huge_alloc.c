#define _GNU_SOURCE
#include "huge_alloc.h"
#include <sys/mman.h>

/* Horribad hack, but I'll make this right later. */
#if defined(MAP_HUGETLB) && defined(USE_MMAP)
# include "huge_alloc.mmap.inc"
#else
# include "huge_alloc.default.inc"
#endif
