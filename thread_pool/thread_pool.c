#include "thread_pool.h"
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include "../huge_alloc/huge_alloc.h"

struct job
{
        size_t id __attribute__ ((aligned(64)));
        size_t limit;
        size_t increment;

        thread_pool_function function;
        void * info;

        unsigned barrier_waiting_for __attribute__ ((aligned(64)));
};

struct thread_pool
{
        unsigned job_sequence;
        unsigned worker_id_counter;
        struct job * job;
        pthread_t * threads;
        unsigned nthreads;
        pthread_mutex_t lock;
        pthread_cond_t queue;
        int sleeping;
        size_t allocated_bytes;
        void * storage;
        void ** storage_vector;
};

static inline void maybe_wake_up(thread_pool_t pool)
{
        if (pool->sleeping)
                thread_pool_wakeup(pool);
}

static void * worker(void*);

thread_pool_t thread_pool_init(unsigned nthreads)
{
        /* implicit worker: the caller */
        if (nthreads == 0) nthreads = 1;
        unsigned allocated_threads = nthreads-1;

        thread_pool_t pool = calloc(1, sizeof(struct thread_pool));
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

static void set_job(thread_pool_t pool, struct job * job)
{
        maybe_wake_up(pool);
        assert(pool->job == NULL);
        assert(job != NULL);
        if ((job != (struct job*)-1ul) && (job != (struct job*)-2ul))
                assert(job->barrier_waiting_for
                       == (pool->nthreads+1));
        __sync_fetch_and_add(&pool->job_sequence, 1);
        pool->job = job;
        __sync_synchronize();
}

void thread_pool_free(thread_pool_t pool)
{
        if (pool == NULL) return;

        set_job(pool, (struct job*)-1ul);

        void * scratch;
        for (unsigned i = 0; i < pool->nthreads; i++) {
                int ret = pthread_join(pool->threads[i], &scratch);
                assert(0 == ret);
        }
        free(pool->threads);
        pthread_mutex_destroy(&pool->lock);
        pthread_cond_destroy(&pool->queue);
        if (pool->storage != NULL)
                huge_free(pool->storage);
        free(pool->storage_vector);
        memset(pool, 0, sizeof(struct thread_pool));
        free(pool);
}

void thread_pool_sleep(thread_pool_t pool)
{
        if (pool->sleeping) return;
        pthread_mutex_lock(&pool->lock);
        set_job(pool, (struct job*)-2ul);
        /* otherwise set_job will wake them back up! */
        pool->sleeping = 1;
        pthread_mutex_unlock(&pool->lock);
}

void thread_pool_wakeup(thread_pool_t pool)
{
        if (!pool->sleeping) return;
        assert((struct job*)-2ul == pool->job);

        pthread_mutex_lock(&pool->lock);
        pool->sleeping = 0;
        pool->job = NULL;
        pthread_cond_broadcast(&pool->queue);
        pthread_mutex_unlock(&pool->lock);
}

static struct job * get_job(thread_pool_t pool)
{
        struct job * job = NULL;
        while (NULL == (job = pool->job))
                __asm__("":::"memory");

        return job;
}

static void release_job(thread_pool_t pool, struct job * job, int master)
{
        assert(job->barrier_waiting_for > 0);
        assert(pool->job == job);
        unsigned sequence = pool->job_sequence;
        unsigned nactive
                = __sync_sub_and_fetch(&job->barrier_waiting_for, 1);
        if (0 == nactive) {
                pool->job = NULL;
                __sync_fetch_and_add(&pool->job_sequence, 1);
                return;
        }

        if (master) {
                while (job->barrier_waiting_for)
                        __asm__("":::"memory");
        } else {
                while (pool->job_sequence == sequence)
                        __asm__("":::"memory");
        }
}

static void do_job(struct job * job, unsigned self)
{
        size_t limit = job->limit, increment = job->increment;
        thread_pool_function function = job->function;
        void * info = job->info;

        while (1) {
                size_t begin = job->id;
                if (begin >= limit) break;
                begin = __sync_fetch_and_add(&job->id, increment);
                if (begin >= limit) break;
                size_t n = limit-begin;
                if (n > increment) n = increment;
                function(begin, begin+n, info, self);
        }
}

static void worker_sleep(thread_pool_t pool)
{
        pthread_mutex_lock(&pool->lock);
        while ((pool->sleeping) || (pool->job == (struct job*)-2ul))
                pthread_cond_wait(&pool->queue, &pool->lock);
        pthread_mutex_unlock(&pool->lock);
}

static void * worker(void * thunk)
{
        thread_pool_t pool = thunk;
        unsigned worker_id
                = __sync_add_and_fetch(&pool->worker_id_counter, 1);

        while (1) {
                struct job * job = get_job(pool);
                if (job == (struct job*)-1UL)
                        break;
                if (job == (struct job*)-2ul) {
                        worker_sleep(pool);
                } else {
                        do_job(job, worker_id);
                        release_job(pool, job, 0);
                }
        }

        return NULL;
}

static void init_job(struct job * job, thread_pool_t pool,
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

static void execute_job(thread_pool_t pool, struct job * job)
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
        size_t nchunks = nthreads*100;
        if (n < nchunks) return minimum;
        return minimum*(n/nchunks);
}

void thread_pool_for(thread_pool_t pool,
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

int main (int argc, char **argv)
{
        unsigned nthread = 0;
        if (argc > 1)
                nthread = atoi(argv[1]);
        thread_pool_t pool = thread_pool_init(nthread);
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
        thread_pool_for(pool, 0, 1000, 1, sleep_test, &n);
        thread_pool_free(pool);
        return 0;
}
#endif
