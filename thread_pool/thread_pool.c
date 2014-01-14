#include "thread_pool.h"
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include "../huge_alloc/huge_alloc.h"

struct job
{
        /* Beginning of the next work unit; atomically incremented. */
        size_t id __attribute__((aligned(64)));
        /* Exclusive end of the last work unit. */
        size_t limit;
        /* Size of work units. */
        size_t increment;

        thread_pool_function function;
        void * info;

        /* Barrier to determine when all workers have seen this job
         * and finished processing it.  Initialised to the number of
         * workers, and atomically decremented until it reaches 0.
         *
         * Alignment avoids false sharing with id.
         */
        unsigned barrier_waiting_for __attribute__((aligned(64)));
};

/* Pseudo jobs: STOP to terminate workers, and SLEEP to loop on
 * pthread_mutex_t/pthread_cond_t. */
#define STOP_JOB ((struct job*)-1ul)
#define SLEEP_JOB ((struct job*)-2ul)

struct thread_pool
{
        /* Sequence counter to determine when job has changed. Could
         * be much smaller without any ABA problem: the barrier in
         * jobs locks us out of unbounded increments. 
         *
         * Only incremented when a job is completely finished: all
         * workers have completed, and job is NULLed out.
         */
        unsigned long job_sequence;
        /* Current job, NULL if none. */
        struct job * job;
        pthread_t * threads;
        unsigned nthreads;
        /* Atomically incremented by threads on start-up: determines
         * their worker id (master is 0). */
        unsigned worker_id_counter;
        /* For sleep/wakeup */
        pthread_mutex_t lock;
        pthread_cond_t queue;
        /* Cached worker-local storage. */
        size_t allocated_bytes_per_worker;
        void * storage;
        void ** storage_vector;
};

static inline void maybe_wake_up(thread_pool_t * pool)
{
        if (pool->job == SLEEP_JOB)
                thread_pool_wakeup(pool);
}

static void * worker(void*); /* implementation of worker loop below */

thread_pool_t * thread_pool_init(unsigned nthreads)
{
        /* implicit worker: the caller */
        if (nthreads == 0) nthreads = 1;
        unsigned allocated_threads = nthreads-1;

        thread_pool_t * pool = calloc(1, sizeof(thread_pool_t));
        pool->threads = calloc(allocated_threads, sizeof(pthread_t));
        pool->nthreads = allocated_threads;
        pthread_mutex_init(&pool->lock, NULL);
        pthread_cond_init(&pool->queue, NULL);

        for (unsigned i = 0; i < allocated_threads; i++) {
                int ret = pthread_create(pool->threads+i, NULL,
                                         worker, pool);
                assert(0 == ret);
        }

        pool->storage = NULL;
        pool->storage_vector = calloc(nthreads, sizeof(void*));

        return pool;
}

static void set_job(thread_pool_t * pool, struct job * job)
{
        maybe_wake_up(pool);
        assert(pool->job == NULL);
        assert(job != NULL);
        if ((job != STOP_JOB) && (job != SLEEP_JOB))
                assert(job->barrier_waiting_for
                       == (pool->nthreads+1));
        int ret = __sync_bool_compare_and_swap(&pool->job, NULL, job);
        assert(ret && "concurrent use of thread pool");
}

void thread_pool_free(thread_pool_t * pool)
{
        if (pool == NULL) return;

        set_job(pool, STOP_JOB);

        for (unsigned i = 0; i < pool->nthreads; i++) {
                int ret = pthread_join(pool->threads[i], NULL);
                assert(0 == ret);
        }
        free(pool->threads);
        pthread_mutex_destroy(&pool->lock);
        pthread_cond_destroy(&pool->queue);
        huge_free(pool->storage);
        free(pool->storage_vector);
        memset(pool, 0, sizeof(thread_pool_t));
        free(pool);
}

size_t thread_pool_count(thread_pool_t * pool)
{
        return 1+pool->nthreads;
}

static size_t ensure_worker_storage(thread_pool_t * pool,
                                    size_t char_per_worker)
{
        size_t aligned_size = (char_per_worker+63)&(~63);
        size_t nworkers = thread_pool_count(pool);
        if (char_per_worker == 0) {
                if (pool->allocated_bytes_per_worker) {
                        huge_free(pool->storage);
                        pool->storage = NULL;
                        pool->allocated_bytes_per_worker = 0;
                }
        } else if (aligned_size > pool->allocated_bytes_per_worker) {
                huge_free(pool->storage);
                pool->storage = huge_calloc(nworkers, aligned_size);
                pool->allocated_bytes_per_worker = aligned_size;
        } else {
                memset(pool->storage, 0, aligned_size*nworkers);
        }

        for (size_t i = 0, offset = 0;
             i < nworkers;
             i++, offset += aligned_size)
                pool->storage_vector[i] = pool->storage+offset;
        return aligned_size;
}

void * const * thread_pool_worker_storage(thread_pool_t * pool,
                                          size_t char_per_worker)
{
        ensure_worker_storage(pool, char_per_worker);
        return pool->storage_vector;
}

void * thread_pool_worker_storage_flat(thread_pool_t * pool,
                                       size_t char_per_worker,
                                       size_t *OUT_aligned_size)
{
        size_t size = ensure_worker_storage(pool, char_per_worker);
        if (OUT_aligned_size != NULL)
                *OUT_aligned_size = size;
        return pool->storage;
}

void thread_pool_sleep(thread_pool_t * pool)
{
        if (pool->job == SLEEP_JOB) return;
        pthread_mutex_lock(&pool->lock);
        set_job(pool, SLEEP_JOB);
        pthread_mutex_unlock(&pool->lock);
}

void thread_pool_wakeup(thread_pool_t * pool)
{
        if (pool->job != SLEEP_JOB) return;

        pthread_mutex_lock(&pool->lock);
        pool->job = NULL;
        pthread_cond_broadcast(&pool->queue);
        pthread_mutex_unlock(&pool->lock);
}

static struct job * get_job(thread_pool_t * pool)
{
        struct job * job = NULL;
        while (NULL == (job = pool->job))
                __asm__("pause":::"memory");

        return job;
}

static void release_job(thread_pool_t * pool, struct job * job, int master)
{
        (void)master;
        assert(job->barrier_waiting_for > 0);
        assert(pool->job == job);
        unsigned sequence = pool->job_sequence;
        unsigned nactive
                = __sync_sub_and_fetch(&job->barrier_waiting_for, 1);
        if (0 == nactive) {
                /* We're the last to reach the barrier. Clear job
                 * out and change the sequence number.
                 *
                 * No overflow issue: if all other workers have
                 * reached the barrier, they're seen the latest
                 * pool->job and previous sequence number!
                 */
                pool->job = NULL;
                __sync_fetch_and_add(&pool->job_sequence, 1);
                return;
        }

        while (pool->job_sequence == sequence)
                __asm__("pause":::"memory");
}

static void do_job(struct job * job, unsigned self)
{
        size_t limit = job->limit, increment = job->increment;
        thread_pool_function function = job->function;
        void * info = job->info;

        while (1) {
                size_t begin = job->id;
                if (begin >= limit) break;
                /* End of the work unit is the least of limit and
                 * begin+increment */
                size_t end = (((limit-begin) <= increment)
                              ? limit
                              : begin+increment);
                /* Try and acquire work unit. */
                if (__sync_bool_compare_and_swap(&job->id, begin, end))
                        function(begin, end, info, self);
        }
}

static void worker_sleep(thread_pool_t * pool)
{
        pthread_mutex_lock(&pool->lock);
        while (pool->job == SLEEP_JOB)
                pthread_cond_wait(&pool->queue, &pool->lock);
        pthread_mutex_unlock(&pool->lock);
}

static void * worker(void * thunk)
{
        thread_pool_t * pool = thunk;
        unsigned worker_id
                = __sync_add_and_fetch(&pool->worker_id_counter, 1);

        while (1) {
                struct job * job = get_job(pool);
                if (job == STOP_JOB)
                        break;
                if (job == SLEEP_JOB) {
                        worker_sleep(pool);
                } else {
                        do_job(job, worker_id);
                        release_job(pool, job, 0);
                }
        }

        return NULL;
}

static void init_job(struct job * job, thread_pool_t * pool,
                     size_t begin, size_t end, size_t granularity,
                     thread_pool_function function, void * info)
{
        if (granularity == 0) granularity = 1;
        job->barrier_waiting_for = pool->nthreads+1;
        job->id = begin;
        job->limit = end;
        job->increment = granularity;
        job->function = function;
        job->info = info;
}

static void execute_job(thread_pool_t * pool, struct job * job)
{
        maybe_wake_up(pool);
        assert(NULL == pool->job);
        set_job(pool, job);
        do_job(job, 0);
        release_job(pool, job, 1);
}

static size_t ideal_granularity(size_t n, size_t minimum, unsigned nthreads)
{
        n = (n+minimum-1)/minimum;
        size_t nchunks = nthreads*10;
        if (n < nchunks) return minimum;
        return minimum*((n+nchunks-1)/nchunks);
}

void thread_pool_for(thread_pool_t * pool,
                     size_t from, size_t end, size_t granularity,
                     thread_pool_function function, void * info)
{
        assert(from <= end);
        if ((pool == NULL)
            || (0 == pool->nthreads)
            || ((end - from) <= granularity)) {
                function(from, end, info, 0);
                return;
        }

        struct job job;
        init_job(&job, pool, from, end, 
                 ideal_granularity(end-from, granularity,
                                   pool->nthreads+1),
                 function, info);
        execute_job(pool, &job);
}

struct map_reduce_info
{
        thread_pool_map function;
        void * info;
        enum thread_pool_reducer reducer;
        double * const * storage;
};

static void map_reduce_worker(size_t begin, size_t end, void * thunk,
                              unsigned id)
{
        struct map_reduce_info * info = thunk;
        double value = info->function(begin, end, info->info, id);
        double * accumulator = info->storage[id];
        switch (info->reducer)
        {
        case THREAD_POOL_REDUCE_SUM:
                *accumulator += value;
                break;
        case THREAD_POOL_REDUCE_MAX:
                *accumulator = fmax(*accumulator, value);
                break;
        case THREAD_POOL_REDUCE_MIN:
                *accumulator = fmin(*accumulator, value);
                break;
        default:
                assert(0 && "Unknown reducer type");
        }
}

double thread_pool_map_reduce(thread_pool_t * pool,
                              size_t from, size_t end, size_t granularity,
                              thread_pool_map function, void * info,
                              enum thread_pool_reducer reducer,
                              double initial_value)
{
        if (pool == NULL)
                return function(from, end, info, 0);

        assert(reducer >= THREAD_POOL_REDUCE_SUM);
        assert(reducer <= THREAD_POOL_REDUCE_MIN);
        size_t n = thread_pool_count(pool);
        double * const * storage
                = ((double * const *)
                   thread_pool_worker_storage(pool, sizeof(double)));
        if (initial_value != 0) {
                for (size_t i = 0; i < n; i++)
                        *storage[i] = initial_value;
        }

        struct map_reduce_info mr_info
                = {.function = function,
                   .info = info,
                   .reducer = reducer,
                   .storage = storage};
        thread_pool_for(pool, from, end, granularity, 
                        map_reduce_worker, &mr_info);

        double accumulator = initial_value;
        switch (reducer)
        {
        case THREAD_POOL_REDUCE_SUM:
                for (size_t i = 0; i < n; i++)
                        accumulator += *storage[i];
                break;
        case THREAD_POOL_REDUCE_MAX:
                for (size_t i = 0; i < n; i++)
                        accumulator = fmax(accumulator, *storage[i]);
                break;
        case THREAD_POOL_REDUCE_MIN:
                for (size_t i = 0; i < n; i++)
                        accumulator = fmin(accumulator, *storage[i]);
                break;
        default:
                assert(0 && "Unknown reducer type");
        }
        return accumulator;
}

#ifdef TEST_THREAD_POOL
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

void sleep_test(size_t from, size_t end, void * info, unsigned id)
{
        (void)id;
        size_t n = *(size_t*)info;
        for (size_t i = from; i < end; i++)
                for (size_t j = 0; j < n; j++)
                        __asm__("":::"memory");
}

double map_reduce_test(size_t from, size_t end, void * info, unsigned id)
{
        (void)info;
        (void)id;

        double acc = 0;
        for (size_t i = from; i < end; i++)
                acc += i;
        return acc;
}

int main (int argc, char **argv)
{
        unsigned nthread = 0;
        if (argc > 1)
                nthread = atoi(argv[1]);
        thread_pool_t * pool = thread_pool_init(nthread);
        size_t n = 5000000;
        if (argc > 2)
                n = atoi(argv[2]);

        struct job job;
        printf("barrier offset: %zu %p %zu\n",
               __offsetof(struct job, barrier_waiting_for),
               &job, n);
        thread_pool_for(pool, 0, 1000, 1, sleep_test, &n);
        thread_pool_sleep(pool);
        printf("sleeping for 5 seconds\n");
        sleep(5);
        double sum = thread_pool_map_reduce(pool, 0, n, 1,
                                            map_reduce_test, NULL,
                                            THREAD_POOL_REDUCE_SUM, 0);
        printf("sum: %f %zu\n", sum, (n*(n-1)/2));
        thread_pool_free(pool);
        return 0;
}
#endif
