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
                       approx_t approx)
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

/* y <- (1-theta)x + theta z */
static void linterp(struct vector * OUT_yv, double theta,
                    const struct vector * xv, const struct vector * zv,
                    thread_pool_t * pool)
{
        assert(theta >= 0);
        assert(theta <= 1);
        double scale = 1-theta;
        size_t nvars = OUT_yv->n;
        assert(xv->n == nvars);
        assert(zv->n == nvars);

        v2d scale2 = {scale, scale}, theta2 = {theta, theta};

        {
                v2d * OUT_y = (v2d*)OUT_yv->x;
                const v2d * x = (v2d*)xv->x, * z = (v2d*)zv->x;
                size_t n = (nvars+1)/2;
                struct linterp_state state
                        = {.scale2 = scale2,
                           .theta2 = theta2,
                           .OUT_y = OUT_y,
                           .x = x, .z = z};
                thread_pool_for(pool, 0, n, 64, linterp_1, &state);
        }

        size_t nviolation = OUT_yv->nviolation;
        if (nviolation && xv->violationp && zv->violationp) {
                assert(xv->nviolation == nviolation);
                assert(zv->nviolation == nviolation);
                v2d * OUT_y = (v2d*)OUT_yv->violation;
                const v2d * x = (v2d*)xv->violation,
                        * z = (v2d*)zv->violation;
                size_t n = (nviolation+1)/2;

                struct linterp_state state
                        = {.scale2 = scale2,
                           .theta2 = theta2,
                           .OUT_y = OUT_y,
                           .x = x, .z = z};
                thread_pool_for(pool, 0, n, 64, linterp_1, &state);
                OUT_yv->violationp = 1;
        } else {
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
