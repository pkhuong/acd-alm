struct vector {
        double * x;
        double * violation; /* Ax-b */
        size_t n;
        size_t nviolation;
        int violationp;
        double value;
};

static void init_vector(struct vector * x, size_t n, size_t nviolation)
{
        x->n = n;
        x->x = huge_calloc(n, sizeof(double));
        if (nviolation) {
                x->violation = huge_calloc(nviolation, sizeof(double));
                x->nviolation = nviolation;
        } else {
                x->violation = NULL;
                x->nviolation = 0;
        }
        x->violationp = 0;
        x->value = nan("");
}

static void copy_vector(struct vector * x, const struct vector * y)
{
        size_t n = x->n;
        assert(n == y->n);
        memcpy(x->x, y->x, n*sizeof(double));
        if (y->violationp
            && (y->violation != NULL)
            && (x->violation != NULL)) {
                size_t m = x->nviolation;
                assert(m == y->nviolation);
                memcpy(x->violation, y->violation, m*sizeof(double));
                x->violationp = 1;
        } else {
                x->violationp = 0;
        }
        x->value = y->value;
}

static void project(struct vector * xv,
                    const double * lower, const double * upper);

static void set_vector(struct vector * x, const double * src,
                       approx_t * approx)
{
        memcpy(x->x, src, x->n*sizeof(double));
        project(x, approx->lower, approx->upper);
}

static void destroy_vector(struct vector * x)
{
        huge_free(x->x);
        huge_free(x->violation);
        memset(x, 0, sizeof(struct vector));
}

typedef double v2d __attribute__ ((vector_size (16)));
typedef uint64_t v2ul __attribute__ ((vector_size (16)));

struct linterp_state
{
        v2d scale2, theta2;
        v2d * OUT_y;
        const v2d * x, * z;
};

static void linterp_1(size_t from, size_t to, void * data, unsigned id)
{
        (void)id;
        struct linterp_state * state = data;
        v2d scale2 = state->scale2, theta2 = state->theta2;
        v2d * OUT_y = state->OUT_y;
        const v2d * x = state->x, * z = state->z;
        for (size_t i = from; i < to; i++)
                OUT_y[i] = scale2*x[i]+theta2*z[i];
}

struct linterp2_state
{
        v2d scale2, theta2;
        v2d * OUT_y[2];
        const v2d * x[2], * z[2];
        size_t n0; /* n0 <= n1 */
};

static void linterp_2(size_t from, size_t to, void * data, unsigned id)
{
        (void)id;
        struct linterp2_state * state = data;
        v2d scale2 = state->scale2, theta2 = state->theta2;
        v2d * OUT_y0 = state->OUT_y[0],
                * OUT_y1 = state->OUT_y[1];
        const v2d * x0 = state->x[0],
                * x1 = state->x[1],
                * z0 = state->z[0],
                * z1 = state->z[1];
        size_t n0 = state->n0;

        if (to <= n0) {
                for (size_t i = from; i < to; i++) {
                        OUT_y0[i] = scale2*x0[i]+theta2*z0[i];
                        OUT_y1[i] = scale2*x1[i]+theta2*z1[i];
                }
        } else if (from >= n0) {
                for (size_t i = from; i < to; i++)
                        OUT_y1[i] = scale2*x1[i]+theta2*z1[i];
        } else {
                size_t i;
                for (i = from; i < n0; i++) {
                        OUT_y0[i] = scale2*x0[i]+theta2*z0[i];
                        OUT_y1[i] = scale2*x1[i]+theta2*z1[i];
                }
                for (; i < to; i++)
                        OUT_y1[i] = scale2*x1[i]+theta2*z1[i];
        }
}

#define SWAP(X, Y) do {                         \
                __typeof__(X) temp = (X);       \
                (X) = (Y);                      \
                (Y) = temp;                     \
        } while (0)


static void linterp_both(thread_pool_t * pool,
                         double scale, double theta,
                         double * OUT_y0, size_t n0,
                         const double * x0, const double * z0,
                         double * OUT_y1, size_t n1,
                         const double * x1, const double * z1)
{
        if (n0 > n1) {
                SWAP(OUT_y0, OUT_y1);
                SWAP(n0, n1);
                SWAP(x0, x1);
                SWAP(z0, z1);
        }

        struct linterp2_state state
                = {.scale2 = {scale, scale},
                   .theta2 = {theta, theta},
                   .OUT_y = {(v2d*)OUT_y0, (v2d*)OUT_y1},
                   .x = {(const v2d*)x0, (const v2d*)x1},
                   .z = {(const v2d*)z0, (const v2d*)z1},
                   .n0 = (n0+1)/2};
        thread_pool_for(pool, 0, (n1+1)/2, 64,
                        linterp_2, &state);
}
                         
#undef SWAP

static void linterp_single(thread_pool_t * pool,
                           double scale, double theta,
                           double * OUT_y, size_t n,
                           const double * x, const double * z)
{
        struct linterp_state state
                = {.scale2 = {scale, scale},
                   .theta2 = {theta, theta},
                   .OUT_y = (v2d*)OUT_y,
                   .x = (const v2d*)x, .z = (const v2d*)z};
        thread_pool_for(pool, 0, (n+1)/2, 64, linterp_1, &state);
}

/* y <- (1-theta)x + theta z */
static void linterp(struct vector * OUT_yv, double theta,
                    const struct vector * xv, const struct vector * zv,
                    thread_pool_t * pool, int violation_only)
{
        assert(theta >= 0);
        assert(theta <= 1);
        double scale = 1-theta;
        size_t nvars = OUT_yv->n;
        assert(xv->n == nvars);
        assert(zv->n == nvars);

        size_t nviolation = OUT_yv->nviolation;
        if (nviolation && xv->violationp && zv->violationp) {
                assert(xv->nviolation == nviolation);
                assert(zv->nviolation == nviolation);
                if (violation_only) {
                        linterp_single(pool, scale, theta,
                                       OUT_yv->violation, nviolation,
                                       xv->violation, zv->violation);
                } else {
                        linterp_both(pool, scale, theta,
                                     OUT_yv->x, nvars, xv->x, zv->x,
                                     OUT_yv->violation, nviolation,
                                     xv->violation, zv->violation);
                }
                OUT_yv->violationp = 1;
        } else {
                assert(!violation_only);
                linterp_single(pool, scale, theta,
                               OUT_yv->x, nvars, xv->x, zv->x);
                OUT_yv->violationp = 0;
        }
        OUT_yv->value = nan("");
}

static double dot(const double * xp, const struct vector * yv)
{
        size_t n = (yv->n+1)/2;
        const v2d * x = (v2d*)xp;
        const v2d * y = (v2d*)yv->x;
        v2d acc = {0,0};
        for (size_t i = 0; i < n; i++)
                acc += x[i]*y[i];
        return acc[0]+acc[1];
}

/* OUT_x <- scale*OUT_x + theta*z
 * OUT_x_viol <- scale*OUT_x_viol + theta*z_viol
 * OUT_y_viol <- next_scale*OUT_x_viol + next_theta*z_viol
 */
struct linterp_xy_state
{
        v2d scale2, theta2;
        v2d next_scale2, next_theta2;
        v2d * OUT_y_viol, * OUT_x, * OUT_x_viol;
        const v2d * z, * z_viol;
        size_t n, nviol;
};

static void linterp_xy_1(size_t begin, size_t end, void * info, unsigned id)
{
        (void)id;
        struct linterp_xy_state * state = info;
        v2d scale2 = state->scale2, theta2 = state->theta2,
                next_scale2 = state->next_scale2,
                next_theta2 = state->next_theta2;
        v2d * y_viol = state->OUT_y_viol,
                * x = state->OUT_x,
                * x_viol = state->OUT_x_viol;
        const v2d * z = state->z, * z_viol = state->z_viol;
        size_t n = state->n, nviol = state->nviol;
        size_t max = end;
        if (max > n) max = n;
        if (max > nviol) max = nviol;

        size_t i = begin;
        if (i < max) {
                for (; i < max; i++) {
                        v2d zi = z[i], xi = x[i];
                        v2d zvi = z_viol[i], xvi = x_viol[i];
                        xi = scale2*xi+theta2*zi;
                        xvi = scale2*xvi+theta2*zvi;
                        y_viol[i] = next_scale2*xvi+next_theta2*zvi;
                        x[i] = xi;
                        x_viol[i] = xvi;
                }
        }

        if (i == end) return;
        if (i < n) {
                assert(i >= nviol);
                for (; i < end; i++) {
                        v2d zi = z[i], xi = x[i];
                        xi = scale2*xi+theta2*zi;
                        x[i] = xi;
                }
        } else if (i < nviol) {
                assert(i >= n);
                for (; i < end; i++) {
                        v2d zvi = z_viol[i], xvi = x_viol[i];
                        xvi = scale2*xvi+theta2*zvi;
                        y_viol[i] = next_scale2*xvi+next_theta2*zvi;
                        x_viol[i] = xvi;
                }
        }
}

static void linterp_xy(struct vector * OUT_yv, struct vector * OUT_xv,
                       const struct vector * zv,
                       double theta, double next_theta,
                       thread_pool_t * pool)
{
        assert(theta >= 0);
        assert(theta <= 1);
        assert(next_theta >= 0);
        assert(next_theta <= 1);

        size_t nvars = OUT_yv->n;
        assert(OUT_xv->n == nvars);
        assert(zv->n == nvars);

        size_t nviolation = OUT_yv->nviolation;
        assert(nviolation && OUT_xv->violationp && zv->violationp);
        assert(OUT_xv->nviolation == nviolation);
        assert(zv->nviolation == nviolation);

        double scale = 1-theta;
        double next_scale = 1-next_theta;
        struct linterp_xy_state state
                = {.scale2 = {scale, scale}, .theta2 = {theta, theta},
                   .next_scale2 = {next_scale, next_scale},
                   .next_theta2 = {next_theta, next_theta},
                   .OUT_y_viol = (v2d *)OUT_yv->violation,
                   .OUT_x = (v2d *)OUT_xv->x,
                   .OUT_x_viol = (v2d*)OUT_xv->violation,
                   .z = (v2d *)zv->x, .z_viol = (v2d *)zv->violation,
                   .n = (nvars+1)/2, .nviol = (nviolation+1)/2};
        size_t n = (nvars > nviolation)?nvars:nviolation;
        thread_pool_for(pool, 0, (n+1)/2, 64,
                        linterp_xy_1, &state);
}
