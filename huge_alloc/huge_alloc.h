/** Cache-line aligned and padded allocation of large vectors
 **
 ** Like calloc, except that the return value is aligned to 64 bytes
 ** when possible, and padded with 16 bytes of zeros at the end.
 **
 ** The alignment helps exploit caches fully, and the padding makes it
 ** easier to write SIMD loops.
 **
 ** If USE_MMAP is defined, allocation goes through mmap to try and
 ** get huge pages on Linux.  In that case, each allocation gets its
 ** own mmap, and they are offset by a small number (between 1 and 9,
 ** inclusively) of cache lines from the beginning of the page.
 **
 ** This allocator is trivial and not meant for small or even medium
 ** allocations... Only use it for a few objects that must be aligned
 ** and are usually large.
 */
#ifndef HUGE_ALLOC_H
#define HUGE_ALLOC_H
#include <stddef.h>

void * huge_calloc(size_t n, size_t size);
/* Safe to call on NULL. */
void huge_free(void * ptr);
#endif
