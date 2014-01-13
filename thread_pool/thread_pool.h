#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <stddef.h>

typedef struct thread_pool * thread_pool_t;

thread_pool_t thread_pool_init(unsigned nthreads);
void thread_pool_free(thread_pool_t);

size_t thread_pool_count(thread_pool_t);
void * const * thread_pool_worker_storage(thread_pool_t,
                                          size_t char_per_worker);
void * thread_pool_worker_storage_flat(thread_pool_t,
                                       size_t char_per_worker,
                                       size_t * OUT_aligned_size);

void thread_pool_sleep(thread_pool_t);
void thread_pool_wakeup(thread_pool_t);

typedef void (*thread_pool_function)(size_t begin, size_t end, void * info,
                                     unsigned worker_id);

void thread_pool_for(thread_pool_t,
                     size_t from, size_t end, size_t granularity,
                     thread_pool_function function, void * info);

typedef double (*thread_pool_map)(size_t begin, size_t end, void * info,
                                  unsigned worker_id);

enum thread_pool_reducer{THREAD_POOL_REDUCE_SUM,
                         THREAD_POOL_REDUCE_MAX,
                         THREAD_POOL_REDUCE_MIN};

double thread_pool_map_reduce(thread_pool_t,
                              size_t from, size_t end, size_t granularity,
                              thread_pool_map function, void * info,
                              enum thread_pool_reducer reducer,
                              double initial_value);
#endif
