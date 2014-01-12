#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <stddef.h>

typedef struct thread_pool * thread_pool_t;

thread_pool_t thread_pool_init(unsigned nthreads);
void thread_pool_free(thread_pool_t);

typedef void (*thread_pool_function)(size_t begin, size_t end, void * info,
                                     unsigned worker_id);

void thread_pool_for(thread_pool_t,
                     size_t from, size_t end, size_t granularity,
                     thread_pool_function function, void * info);
#endif
