#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <stddef.h>

typedef struct thread_pool * thread_pool_t;

thread_pool_t thread_pool_init(unsigned nthreads);
void thread_pool_free(thread_pool_t);

size_t thread_pool_count();
void * const * thread_pool_worker_storage(size_t char_per_worker);
void * thread_pool_worker_storage_flat(size_t char_per_worker);

void thread_pool_sleep(thread_pool_t);
void thread_pool_wakeup(thread_pool_t);

typedef void (*thread_pool_function)(size_t begin, size_t end, void * info,
                                     unsigned worker_id);

void thread_pool_for(thread_pool_t,
                     size_t from, size_t end, size_t granularity,
                     thread_pool_function function, void * info);

typedef double (*thread_pool_map)(size_t begin, size_t end, void * info,
                                  unsigned worker_id);

enum thread_pool_reducer{REDUCE_SUM, REDUCE_MAX, REDUCE_MIN};

void thread_pool_map_reduce(thread_pool_t,
                            size_t from, size_t end, size_t granularity,
                            thread_pool_map function, void * info,
                            enum thread_pool_reducer reducer);
#endif
