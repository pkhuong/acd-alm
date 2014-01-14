/** Spinlocky thread pool with parallel dotimes and dotimes/reduce
 **/
#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <stddef.h>

/* NULL means no thread pool, i.e., serial execution */
typedef struct thread_pool * thread_pool_t;

/* Allocate a new thread pool for n workers; the caller (master)
 * counts as a worker, so one fewer thread is spawned.  If nworker is
 * 0, it is treated as 1. */
thread_pool_t thread_pool_init(unsigned nworker);
/* Terminate worker threads and free the pool. Safe to call on NULL. */
void thread_pool_free(thread_pool_t);

/* Number of workers, including the master (i.e., one more than the
 * number of threads in the pool). */
size_t thread_pool_count(thread_pool_t);
/* Allocate char_per_worker space for each worker, and return an array
 * of nworker void pointers.  This array is reused on each call to
 * thread_pool_worker_storage and is freed by thread_pool_free.
 *
 * The function aligns and pads allocations to try and guarantee that
 * each worker get its own cache line (and thus avoid false sharing).
 * Each worker's data is zero-initialised.
 *
 * The storage is also always recycled; char_per_worker = 0 will
 * deallocate the internal backing vector.
 */
void * const * thread_pool_worker_storage(thread_pool_t,
                                          size_t char_per_worker);
/* Instead return the backing vector directly.  If OUT_aligned_size is
 * non-NULL, the padded allocation size per worker is stored there.
 */
void * thread_pool_worker_storage_flat(thread_pool_t,
                                       size_t char_per_worker,
                                       size_t * OUT_aligned_size);

/* The thread pool defaults to spin-loops, but workers can be
 * explicitly put to sleep (via pthread mutex/cv) if the thread pool
 * will go unused for a long period.
 *
 * Workers will be woken up if necessary when a new job is executed.
 */
void thread_pool_sleep(thread_pool_t);
void thread_pool_wakeup(thread_pool_t);

/* Parallel dotimes: calls a function for each size_t value in [from, end).
 *
 * The thread_pool_function is called with the half-open range to
 * cover, a user pointer, and the worker id (0 is the calling thread,
 * [1, nworker) are worker threads).  This helps for quick
 * worker-local caches, etc., in conjunction with worker_storage
 * functions above.
 *
 * The ranges never overlap and their sizes are multiples of
 * granularity, except for the last one.
 *
 * This granularity is a minimum: it is scaled up if possible, with a
 * goal of at least 10 work unit/worker.
 */
typedef void (*thread_pool_function)(size_t begin, size_t end, void * info,
                                     unsigned worker_id);
void thread_pool_for(thread_pool_t,
                     size_t from, size_t end, size_t granularity,
                     thread_pool_function function, void * info);

/* Parallel dotimes/reduce.
 *
 * Same as thread_pool_for, but the map function returns a double
 * float value.  These values are accumulated with the reducer
 * function and the final accumulator is returned.  Initial value
 * should be an identity element for the reducer (e.g. 0 for SUM, and
 * +/- infty for MAX/MIN).
 *
 * NOTE: map_reduce already uses worker_storage above, so the
 * thread_pool_map may not use it for its own purposes.
 */
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
