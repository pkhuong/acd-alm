#ifndef HUGE_ALLOC_H
#define HUGE_ALLOC_H
#include <stddef.h>

void * huge_calloc(size_t n, size_t size);
void huge_free(void * ptr);
#endif
