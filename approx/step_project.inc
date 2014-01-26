static inline double min(double x, double y)
{
        return (x<y)?x:y;
}

static inline double max(double x, double y)
{
        return (x>y)?x:y;
}

static void project(struct vector * xv,
                    const double * lower, const double * upper)
{
        size_t n = (xv->n+1)/2;
        v2d * x = (v2d*)xv->x;
        const v2d * l = (const v2d*)lower, * u = (const v2d*)upper;
        for (size_t i = 0; i < n; i++) {
                v2d clamp_low = __builtin_ia32_maxpd(l[i], x[i]);
                v2d clamp_high = __builtin_ia32_minpd(clamp_low, u[i]);
                x[i] = clamp_high;
        }

        xv->violationp = 0;
        xv->quad_value = xv->value = nan("");
}

struct step_state
{
        v2d * zp;
        const v2d * g, * z, * l, * u, * iv;
        v2d itheta;
};

static double step_1(size_t begin, size_t end, void * info,
                     unsigned id)
{
        (void)id;
        struct step_state * state = info;
        v2d * zp = state->zp;
        const v2d * g = state->g, * z = state->z,
                * l = state->l, * u = state->u,
                * iv = state->iv;
        v2d itheta = state->itheta;
        v2ul mask = {~(1ull<<63), ~(1ull<<63)};

        v2d max_z = {0,0};
        for (size_t i = begin; i < end; i++) {
                v2d gi = g[i], zi = z[i],
                        li = l[i], ui = u[i],
                        inv_vi = iv[i];
                v2d step = itheta*inv_vi;
                v2d trial = zi - gi*step;
                trial = __builtin_ia32_maxpd(li, trial);
                trial = __builtin_ia32_minpd(ui, trial);
                zp[i] = trial;
                max_z = __builtin_ia32_maxpd(max_z,
                                             (v2d)((v2ul)trial&mask));
        }
        return max(max_z[0], max_z[1]);
}

static void step(struct vector * zpv, double theta,
                 const struct vector * gv, const struct vector * zv,
                 const double * restrict lower, const double * restrict upper,
                 const double * restrict inv_v,
                 thread_pool_t * pool)
{
        size_t n = zpv->n;
        assert(gv->n == n);
        assert(zv->n == n);

        size_t vector_n = (n+1)/2;
        v2d * zp = (v2d*)zpv->x;
        const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                * l = (const v2d*)lower, * u = (const v2d*)upper,
                * iv = (const v2d*)inv_v;

        double inv_theta = (1-1e-6)/theta;   /* protect vs rounding */
        v2d itheta = {inv_theta, inv_theta}; /* errors. */
        struct step_state state
                = {.zp = zp, 
                   .g = g, .z = z,
                   .l = l, .u = u,
                   .iv = iv,
                   .itheta = itheta};
        double max_z = thread_pool_map_reduce(pool,
                                              0, vector_n, 512,
                                              step_1, &state,
                                              THREAD_POOL_REDUCE_MAX, 0);
        assert(max_z < HUGE_VAL);
        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
        zpv->quad_value = nan("");
}

struct long_step_state
{
        v2d itheta, theta2;
        v2d * zp;
        const v2d * g, * z, * l, * u, * iv, * v;
};

static double
long_step_1(size_t from, size_t end, void * thunk, unsigned id)
{
        (void)id;
        struct long_step_state * state = thunk;
        v2d itheta = state->itheta, theta2 = state->theta2;
        v2d * zp = state->zp;
        const v2d * g = state->g, * z = state->z,
                * l = state->l, * u = state->u,
                * iv = state->iv, * v = state->v;
        v2d linear_estimate = {0, 0};
        v2d quad_estimate = {0, 0};
        for (size_t i = from; i < end; i++) {
                v2d gi = g[i], zi = z[i],
                        li = l[i], ui = u[i],
                        inv_vi = iv[i],
                        vi = v[i];
                v2d step = itheta*inv_vi;
                v2d trial = zi - gi*step;
                trial = __builtin_ia32_maxpd(li, trial);
                trial = __builtin_ia32_minpd(ui, trial);
                zp[i] = trial;
                v2d delta = trial - zi;
                linear_estimate += gi*delta;
                quad_estimate += delta*delta*theta2*vi;
        }
        return ((linear_estimate[0]+linear_estimate[1])
                +.5*(quad_estimate[0]+quad_estimate[1]));        
}
/*
 * f(x) ~= f(z) + g'x + \sum_i (theta v_i)/2 (x-z)^2_i
 *   min in z_i - 1/(theta v_i) g_i
 */
static double __attribute__((unused))
long_step(struct vector * zpv, double theta, double length,
          const struct vector * gv, const struct vector * zv,
          const double * restrict lower, const double * restrict upper,
          const double * restrict inv_v, const double * restrict vs,
          thread_pool_t * pool)
{
        size_t n = zpv->n;
        assert(gv->n == n);
        assert(zv->n == n);

        size_t vector_n = (n+1)/2;
        v2d * zp = (v2d*)zpv->x;
        const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                * l = (const v2d*)lower, * u = (const v2d*)upper,
                * iv = (const v2d*)inv_v, *v = (const v2d*)vs;
        assert(length >= 1);
        double inv_theta = (length-1e-6)/theta;
        struct long_step_state state 
                = { .itheta = {inv_theta, inv_theta},
                    .theta2 = {theta/length, theta/length},
                    .zp = zp,
                    .g = g, .z = z,
                    .l = l, .u = u,
                    .iv = iv, .v = v};
        double estimate
                = thread_pool_map_reduce(pool,
                                         0, vector_n, 64,
                                         long_step_1, &state,
                                         THREAD_POOL_REDUCE_SUM,
                                         0);
        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
        zpv->quad_value = nan("");
        return estimate;
}

struct gradient2_long_step_state
{
        struct gradient2_state gradient2;
        struct long_step_state step;
};

static double
gradient2_long_step_1(size_t begin, size_t end, void * info,
                      unsigned id)
{
        (void)id;
        struct gradient2_long_step_state * state = info;
        const v2d * linear = state->gradient2.linear;
        v2d * g0 = state->gradient2.grad[0],
                * g1 = state->gradient2.grad[1];

        v2d itheta = state->step.itheta, theta2 = state->step.theta2;
        v2d * zp = state->step.zp;
        const v2d * g = state->step.g, * z = state->step.z,
                * l = state->step.l, * u = state->step.u,
                * iv = state->step.iv, * v = state->step.v;
        v2d linear_estimate = {0, 0};
        v2d quad_estimate = {0, 0};

        assert(g1 == g);

        int last = 0;
        while (!last) {
                size_t begin2 = begin;
                size_t end2 = begin += APPROX_SPMV_CHUNK;
                if ((end2 < begin2) || (end <= end2)) {
                        end2 = end;
                        last = 1;
                }
                block_mult2_subrange_1(begin2, end2,
                                       &state->gradient2.mult,
                                       &begin2, &end2);
                assert(0 == begin2%2);
                assert((end2 == state->gradient2.mult.matrix->nrows)
                       || (0 == end2%2));

                size_t n = (end2+1)/2;
                for (size_t i = begin2/2; i < n; i++) {
                        v2d lineari = linear[i];
                        g0[i] += lineari;
                        v2d gi = g1[i] += lineari;
                        v2d zi = z[i],
                                li = l[i], ui = u[i],
                                inv_vi = iv[i],
                                vi = v[i];
                        v2d step = itheta*inv_vi;
                        v2d trial = zi - gi*step;
                        trial = __builtin_ia32_maxpd(li, trial);
                        trial = __builtin_ia32_minpd(ui, trial);
                        zp[i] = trial;
                        v2d delta = trial - zi;
                        linear_estimate += gi*delta;
                        quad_estimate += delta*delta*theta2*vi;
                }
        }

        return ((linear_estimate[0]+linear_estimate[1])
                +.5*(quad_estimate[0]+quad_estimate[1]));        
}

static double
gradient2_long_step(approx_t * approx, thread_pool_t * pool,
                    struct vector ** OUT_grad, struct vector ** zy,
                    struct vector * zpv, double theta, double length,
                    const double * lower, const double * upper,
                    const double * inv_v, const double * vs)
{
        struct gradient2_long_step_state state;
        double * grad[2] = {OUT_grad[0]->x, OUT_grad[1]->x};
        const double * x[2] = {zy[0]->violation,
                               zy[1]->violation};

        size_t n;

        {
                size_t nvars = OUT_grad[0]->n,
                        nrows = zy[0]->nviolation;
                n = nvars;
                assert(OUT_grad[1]->n == nvars);
                assert(zy[1]->nviolation == nrows);
                assert(nvars == approx->nvars);
                assert(nrows == approx->nrhs);
                for (size_t i = 0; i < 2; i++)
                        assert(nvars == zy[i]->n);
                for (size_t i = 0; i < 2; i++) {
#ifndef NO_CACHING
                        if (!zy[i]->violationp)
#endif
                                compute_violation(zy[i], approx, pool);
                }
                
                state.gradient2 = 
                        (struct gradient2_state)
                        {.mult = {.out = grad,
                                  .matrix = &approx->matrix->block_transpose,
                                  .x = x},
                         .linear = (v2d*)approx->linear,
                         .grad = {(v2d*)grad[0], (v2d*)grad[1]}};
        }

        {
                assert(n == zpv->n);
                struct vector * gv = OUT_grad[1];
                struct vector * zv = zy[0];
                assert(gv->n == n);
                assert(zv->n == n);
                
                v2d * zp = (v2d*)zpv->x;
                const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                        * l = (const v2d*)lower, * u = (const v2d*)upper,
                        * iv = (const v2d*)inv_v, *v = (const v2d*)vs;
                assert(length >= 1);
                double inv_theta = (length-1e-6)/theta;
                state.step =
                        (struct long_step_state)
                        { .itheta = {inv_theta, inv_theta},
                          .theta2 = {theta/length, theta/length},
                          .zp = zp,
                          .g = g, .z = z,
                          .l = l, .u = u,
                          .iv = iv, .v = v};
        }

        double estimate
                = thread_pool_map_reduce
                (pool,
                 0, approx->matrix->block_transpose.nblocks, 8,
                 gradient2_long_step_1, &state,
                 THREAD_POOL_REDUCE_SUM, 0);

        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
        zpv->quad_value = nan("");
        return estimate;
}

struct gradient_long_step_state
{
        struct gradient_state gradient;
        struct long_step_state step;
};

static double
gradient_long_step_1(size_t begin, size_t end, void * info,
                      unsigned id)
{
        (void)id;
        struct gradient_long_step_state * state = info;
        const v2d * linear = state->gradient.linear;
        v2d * g = state->gradient.grad;

        v2d itheta = state->step.itheta, theta2 = state->step.theta2;
        v2d * zp = state->step.zp;
        const v2d  * z = state->step.z,
                * l = state->step.l, * u = state->step.u,
                * iv = state->step.iv, * v = state->step.v;
        v2d linear_estimate = {0, 0};
        v2d quad_estimate = {0, 0};

        assert(g == state->step.g);

        int last = 0;
        while (!last) {
                size_t begin2 = begin;
                size_t end2 = begin += APPROX_SPMV_CHUNK;
                if ((end2 < begin2) || (end <= end2)) {
                        end2 = end;
                        last = 1;
                }
                block_mult_subrange_1(begin2, end2,
                                      &state->gradient.mult,
                                      &begin2, &end2);
                assert(0 == begin2%2);
                assert((end2 == state->gradient.mult.matrix->nrows)
                       || (0 == end2%2));

                size_t n = (end2+1)/2;
                for (size_t i = begin2/2; i < n; i++) {
                        v2d lineari = linear[i];
                        v2d gi = g[i] += lineari;
                        v2d zi = z[i],
                                li = l[i], ui = u[i],
                                inv_vi = iv[i],
                                vi = v[i];
                        v2d step = itheta*inv_vi;
                        v2d trial = zi - gi*step;
                        trial = __builtin_ia32_maxpd(li, trial);
                        trial = __builtin_ia32_minpd(ui, trial);
                        zp[i] = trial;
                        v2d delta = trial - zi;
                        linear_estimate += gi*delta;
                        quad_estimate += delta*delta*theta2*vi;
                }
        }

        return ((linear_estimate[0]+linear_estimate[1])
                +.5*(quad_estimate[0]+quad_estimate[1]));        
}

static double
gradient_long_step(approx_t * approx, thread_pool_t * pool,
                   struct vector * OUT_grad, struct vector * yv,
                   struct vector * zv,
                   struct vector * zpv, double theta, double length,
                   const double * lower, const double * upper,
                   const double * inv_v, const double * vs)
{
        struct gradient_long_step_state state;

        size_t n;

        {
                size_t nvars = OUT_grad->n,
                        nrows = yv->nviolation;
                n = nvars;
                assert(OUT_grad->n == nvars);
                assert(nvars == approx->nvars);
                assert(nrows == approx->nrhs);
                assert(nvars == yv->n);
#ifndef NO_CACHING
                if (!yv->violationp)
#endif
                        compute_violation(yv, approx, pool);

                state.gradient = 
                        (struct gradient_state)
                        {.mult = {.out = OUT_grad->x,
                                  .matrix = &approx->matrix->block_transpose,
                                  .x = yv->violation},
                         .linear = (v2d*)approx->linear,
                         .grad = (v2d*)OUT_grad->x};
        }

        {
                assert(n == zpv->n);
                struct vector * gv = OUT_grad;
                assert(gv->n == n);
                assert(zv->n == n);
                
                v2d * zp = (v2d*)zpv->x;
                const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                        * l = (const v2d*)lower, * u = (const v2d*)upper,
                        * iv = (const v2d*)inv_v, *v = (const v2d*)vs;
                assert(length >= 1);
                double inv_theta = (length-1e-6)/theta;
                state.step =
                        (struct long_step_state)
                        { .itheta = {inv_theta, inv_theta},
                          .theta2 = {theta/length, theta/length},
                          .zp = zp,
                          .g = g, .z = z,
                          .l = l, .u = u,
                          .iv = iv, .v = v};
        }

        double estimate
                = thread_pool_map_reduce
                (pool,
                 0, approx->matrix->block_transpose.nblocks, 16,
                 gradient_long_step_1, &state,
                 THREAD_POOL_REDUCE_SUM, 0);

        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
        zpv->quad_value = nan("");
        return estimate;
}

struct gradient2_step_state
{
        struct gradient2_state gradient;
        struct step_state step;
};

double gradient2_step_1(size_t begin, size_t end, void * info,
                        unsigned id)
{
        (void)id;
        struct gradient2_step_state * state = info;
        
        const v2d * linear = state->gradient.linear;
        v2d * g0 = state->gradient.grad[0], 
                * g1 = state->gradient.grad[1];

        v2d * zp = state->step.zp;
        const v2d * g = state->step.g, * z = state->step.z,
                * l = state->step.l, * u = state->step.u,
                * iv = state->step.iv;
        v2d itheta = state->step.itheta;
        v2ul mask = {~(1ull<<63), ~(1ull<<63)};

        assert(g == g1);

        v2d max_z = {0,0};
        int last = 0;
        while (!last) {
                size_t begin2 = begin;
                size_t end2 = begin += APPROX_SPMV_CHUNK;
                if ((end2 < begin2) || (end <= end2)) {
                        end2 = end;
                        last = 1;
                }
                block_mult2_subrange_1(begin2, end2,
                                       &state->gradient.mult,
                                       &begin2, &end2);
                assert(0 == begin2%2);
                assert((end2 == state->gradient.mult.matrix->nrows)
                       || (0 == end2%2));

                size_t n = (end2+1)/2;
                for (size_t i = begin2/2; i < n; i++) {
                        v2d gi;
                        {
                                v2d li = linear[i];
                                g0[i] += li;
                                gi = g1[i] += li;
                        }
                        v2d zi = z[i],
                                li = l[i], ui = u[i],
                                inv_vi = iv[i];
                        v2d step = itheta*inv_vi;
                        v2d trial = zi - gi*step;
                        trial = __builtin_ia32_maxpd(li, trial);
                        trial = __builtin_ia32_minpd(ui, trial);
                        zp[i] = trial;
                        max_z = __builtin_ia32_maxpd(max_z,
                                                     (v2d)((v2ul)trial&mask));
                }
        }
        return max(max_z[0], max_z[1]);
}

static void
gradient2_step(approx_t * approx, thread_pool_t * pool,
               struct vector ** OUT_grad, struct vector ** xv,
               struct vector * zpv, double theta,
               const struct vector * gv, const struct vector * zv,
               const double * restrict lower, const double * restrict upper,
               const double * restrict inv_v)
{
        struct gradient2_step_state state;

        double * grad[2] = {OUT_grad[0]->x, OUT_grad[1]->x};
        const double * x[2] = {xv[0]->violation, xv[1]->violation};
        size_t n = zpv->n;

        {
                size_t nvars = OUT_grad[0]->n,
                        nrows = xv[0]->nviolation;
                assert(n == nvars);
                assert(OUT_grad[1]->n == nvars);
                assert(xv[1]->nviolation == nrows);
                assert(nvars == approx->nvars);
                assert(nrows == approx->nrhs);
                for (size_t i = 0; i < 2; i++)
                        assert(nvars == xv[i]->n);
                
                for (size_t i = 0; i < 2; i++) {
#ifndef NO_CACHING
                        if (!xv[i]->violationp)
#endif
                                compute_violation(xv[i], approx, pool);
                }


                state.gradient
                        = (struct gradient2_state)
                        {.mult = {.out = grad,
                                  .matrix = &approx->matrix->block_transpose,
                                  .x = x},
                         .linear = (v2d*)approx->linear,
                         .grad = {(v2d*)grad[0], (v2d*)grad[1]}};
        }
        {
                assert(gv->n == n);
                assert(zv->n == n);
                
                v2d * zp = (v2d*)zpv->x;
                const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                        * l = (const v2d*)lower, * u = (const v2d*)upper,
                        * iv = (const v2d*)inv_v;
                
                double inv_theta = (1-1e-6)/theta;   /* protect vs rounding */
                v2d itheta = {inv_theta, inv_theta}; /* errors. */
                state.step
                        = (struct step_state)
                        {.zp = zp, 
                         .g = g, .z = z,
                         .l = l, .u = u,
                         .iv = iv,
                         .itheta = itheta};
        }

        double max_z
                = thread_pool_map_reduce
                (pool,
                 0, approx->matrix->block_transpose.nblocks, 16,
                 gradient2_step_1, &state,
                 THREAD_POOL_REDUCE_MAX, 0);
        assert(max_z < HUGE_VAL);
        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
        zpv->quad_value = nan("");        
}
