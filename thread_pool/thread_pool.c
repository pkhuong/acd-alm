#include "thread_pool.h"
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>
#include <strings.h>

struct job;

struct thread_pool
{
        pthread_t * threads;
        unsigned nthreads;
        unsigned nactive;
        pthread_mutex_t lock;
        pthread_cond_t new_job_queue;
        pthread_cond_t job_done_queue;
        struct job * job;
        unsigned job_sequence;
        unsigned worker_id_counter;
};

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
        pthread_cond_init(&pool->new_job_queue, NULL);
        pthread_cond_init(&pool->job_done_queue, NULL);
        pool->worker_id_counter = 1;

        for (unsigned i = 0; i < allocated_threads; i++) {
                int ret = pthread_create(pool->threads+i, NULL,
                                         worker, pool);
                assert(0 == ret);
        }

        return pool;
}

static void set_job(thread_pool_t, struct job *);

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
        pthread_cond_destroy(&pool->new_job_queue);
        pthread_cond_destroy(&pool->job_done_queue);
        memset(pool, 0, sizeof(struct thread_pool));
        free(pool);
}

static void set_job(thread_pool_t pool, struct job * job)
{
        assert(pool->job == NULL);
        assert(pool->nactive == 0);

        __sync_fetch_and_add(&pool->job_sequence, 1);
        pool->job = job;
        __sync_synchronize();
}

static struct job * get_job(thread_pool_t pool)
{
        struct job * job = NULL;
        while (NULL == (job = pool->job))
                __asm__("":::"memory");
        __sync_fetch_and_add(&pool->nactive, 1);

        return job;
}

static void release_job(thread_pool_t pool, struct job * job, int master)
{
        (void)master;
        assert(pool->nactive > 0);
        assert(pool->job == job);
        unsigned sequence = pool->job_sequence;
        unsigned nactive = __sync_sub_and_fetch(&pool->nactive, 1);
        if (0 == nactive) {
                pool->job = NULL;
                __sync_fetch_and_add(&pool->job_sequence, 1);
                return;
        }

        while (pool->job_sequence == sequence)
                __asm__("":::"memory");
}

struct job
{
        size_t id;
        size_t limit;
        size_t increment;

        thread_pool_function function;
        void * info;
};

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

static void * worker(void * thunk)
{
        thread_pool_t pool = thunk;
        unsigned worker_id;
        {
                int ret = pthread_mutex_lock(&pool->lock);
                assert(0 == ret);
                worker_id = pool->worker_id_counter++;
                ret = pthread_mutex_unlock(&pool->lock);
                assert(0 == ret);
        }

        while (1) {
                struct job * job = get_job(pool);
                if (job == (struct job*)-1UL)
                        break;
                do_job(job, worker_id);
                release_job(pool, job, 0);
        }

        return NULL;
}

static void init_job(struct job * job,
                     size_t begin, size_t end, size_t granularity,
                     thread_pool_function function, void * info)
{
        if (granularity == 0) granularity = 1;
        job->id = begin;
        job->limit = end;
        job->increment = granularity;
        job->function = function;
        job->info = info;
}

static void execute_job(thread_pool_t pool, struct job * job)
{
        assert(NULL == pool->job);
        assert(0 == pool->nactive);
        __sync_fetch_and_add(&pool->job_sequence, 1);
        pool->job = job;
        __sync_fetch_and_add(&pool->nactive, 1);

        do_job(job, 0);
        release_job(pool, job, 1);
}

static size_t ideal_granularity(size_t n, size_t minimum, unsigned nthreads)
{
        size_t scale = nthreads*nthreads;
        size_t m = (n+scale-1)/scale;
        size_t grains = ((m+minimum-1)/minimum);
        if (grains == 0) grains = 1;
        return minimum*grains;
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
        init_job(&job, from, end, 
                 ideal_granularity(end-from, granularity, pool->nthreads),
                 function, info);
        execute_job(pool, &job);
}
