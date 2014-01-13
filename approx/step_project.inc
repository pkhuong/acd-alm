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
}

static void step(struct vector * zpv, double theta,
                 const struct vector * gv, const struct vector * zv,
                 const double * restrict lower, const double * restrict upper,
                 const double * restrict inv_v)
{
        size_t n = zpv->n;
        assert(gv->n == n);
        assert(zv->n == n);

        size_t vector_n = (n+1)/2;
        v2d * zp = (v2d*)zpv->x;
        const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                * l = (const v2d*)lower, * u = (const v2d*)upper,
                * iv = (const v2d*)inv_v;
        v2d max_z = {0,0};
        double inv_theta = (1-1e-6)/theta;   /* protect vs rounding */
        v2d itheta = {inv_theta, inv_theta}; /* errors. */
        v2ul mask = {~(1ull<<63), ~(1ull<<63)};
        for (size_t i = 0; i < vector_n; i++) {
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
        assert(max(max_z[0], max_z[1]) < HUGE_VAL);
        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
}

/*
 * f(x) ~= f(z) + g'x + \sum_i (theta v_i)/2 (x-z)^2_i
 *   min in z_i - 1/(theta v_i) g_i
 */
static double
long_step(struct vector * zpv, double theta, double length,
          const struct vector * gv, const struct vector * zv,
          const double * restrict lower, const double * restrict upper,
          const double * restrict inv_v, const double * restrict vs)
{
        size_t n = zpv->n;
        assert(gv->n == n);
        assert(zv->n == n);

        size_t vector_n = (n+1)/2;
        v2d * zp = (v2d*)zpv->x;
        const v2d * g = (const v2d*)gv->x, * z = (v2d*)zv->x,
                * l = (const v2d*)lower, * u = (const v2d*)upper,
                * iv = (const v2d*)inv_v, *v = (const v2d*)vs;
        v2d max_z = {0,0};
        assert(length >= 1);
        double inv_theta = (length-1e-6)/theta;
        v2d itheta = {inv_theta, inv_theta};
        v2d theta2 = {theta/length, theta/length};
        v2ul mask = {~(1ull<<63), ~(1ull<<63)};
        v2d linear_estimate = {0, 0};
        v2d quad_estimate = {0, 0};
        for (size_t i = 0; i < vector_n; i++) {
                v2d gi = g[i], zi = z[i],
                        li = l[i], ui = u[i],
                        inv_vi = iv[i],
                        vi = v[i];
                v2d step = itheta*inv_vi;
                v2d trial = zi - gi*step;
                trial = __builtin_ia32_maxpd(li, trial);
                trial = __builtin_ia32_minpd(ui, trial);
                zp[i] = trial;
                max_z = __builtin_ia32_maxpd(max_z,
                                             (v2d)((v2ul)trial&mask));
                v2d delta = trial - zi;
                linear_estimate += gi*delta;
                quad_estimate += delta*delta*theta2*vi;
        }
        zpv->violationp = 0; /* cache is now invalid */
        zpv->value = nan("");
        if (isfinite(max(max_z[0], max_z[1])))
                return ((linear_estimate[0]+linear_estimate[1])
                        +.5*(quad_estimate[0]+quad_estimate[1]));
        return -HUGE_VAL; /* Might be a finite precision problem. fail */
}
