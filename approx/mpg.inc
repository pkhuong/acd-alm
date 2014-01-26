static void mpg_project_gradient(struct vector * OUT_projected,
                                 const struct vector * x,
                                 const struct vector * g,
                                 const double * lower,
                                 const double * upper)
{
        size_t n = OUT_projected->n;
        assert(n == x->n);
        assert(n == g->n);

        for (size_t i = 0; i < n; i++) {
                double d = -g->x[i];
                double xi = x->x[i];
                if ((xi <= lower[i]) && (d < 0))
                        d = 0;
                if ((xi >= upper[i]) && (d > 0))
                        d = 0;
                OUT_projected->x[i] = d;
        }

        OUT_projected->violationp = 0;
        OUT_projected->quad_value = OUT_projected->value = nan("");
}

static double kappa = .55;
static double delta = .5;
static double sigma = 1e-4;

static int mpg_iter(approx_t * approx, struct approx_state * state,
                    double * OUT_pg, thread_pool_t * pool)
{
        size_t n = approx->nvars;

        state->x.violationp = 0;
        state->x.quad_value = state->x.value = nan("");

        project(&state->x, approx->lower, approx->upper);
        gradient(&state->g, approx, &state->x, pool);
        if (OUT_pg != NULL)
                    *OUT_pg = project_gradient_norm(&state->g,
                                                    &state->x,
                                                    approx->lower,
                                                    approx->upper);

        mpg_project_gradient(&state->u, &state->x, &state->g,
                             approx->lower, approx->upper);
        double lambda;
        {
                /* g2 <- Hu */
                curve_gradient(&state->g2, approx, &state->u, pool);
                double alpha_sd = -(dot(state->g.x, &state->u, pool)
                                    /dot(state->u.x, &state->g2, pool));
                double alpha_mg = (dot(state->g2.x, &state->u, pool)
                                   /dot(state->g2.x, &state->g2, pool));
                
                lambda = ((alpha_mg >= kappa*alpha_sd)
                          ? alpha_mg
                          : alpha_sd-delta*alpha_mg);
                if (lambda > 1e8) lambda = 1e8;
                if (lambda < 1e-8) lambda = 1e-8;
        }

        /* y = x + lambda u */
        for (size_t i = 0; i < n; i++)
                state->y.x[i] = state->x.x[i] + lambda*state->u.x[i];
        project(&state->y, approx->lower, approx->upper);

        /* y = step */
        for (size_t i = 0; i < n; i++)
                state->y.x[i] -= state->x.x[i];

        double value = compute_value(approx, &state->x, pool);
        double estimate = dot(state->g.x, &state->y, pool);
        assert(estimate <= 0);
        double min_alpha = 1e-12, max_alpha = 1;
        for (size_t i = 0; max_alpha-min_alpha > 1e-12; i++) {
                assert(min_alpha < max_alpha);
                double alpha = (min_alpha+max_alpha)/2;
                if (i == 0) alpha = 1;
                /* z = x + alpha step */
                for (size_t i = 0; i < n; i++)
                        state->z.x[i]
                                = state->x.x[i]+alpha*state->y.x[i];
                state->z.violationp = 0;
                state->z.quad_value = state->z.value = nan("");
                double trial = compute_value(approx, &state->z, pool);
                if (trial <= value + sigma*alpha*estimate) {
                        if (!i) goto done;
                        min_alpha = alpha;
                } else {
                        max_alpha = alpha;
                }
        }
        for (size_t i = 0; i < n; i++)
                state->z.x[i]
                        = state->x.x[i]+min_alpha*state->y.x[i];
        state->z.violationp = 0;
        state->z.quad_value = state->z.value = nan("");

done:
        {
                /* swap */
                struct vector temp = state->z;
                state->z = state->x;
                state->x = temp;
        }

        return 0;
}

static int pbb_iter(approx_t * approx, struct approx_state * state,
                    double * OUT_pg, thread_pool_t * pool)
{
        size_t n = approx->nvars;

        state->x.violationp = 0;
        state->x.quad_value = state->x.value = nan("");

        project(&state->x, approx->lower, approx->upper);
        gradient(&state->g, approx, &state->x, pool);
        if (OUT_pg != NULL)
                    *OUT_pg = project_gradient_norm(&state->g,
                                                    &state->x,
                                                    approx->lower,
                                                    approx->upper);

        double step = state->step_length;

        /* z = x - step g */
        for (size_t i = 0; i < n; i++)
                state->z.x[i] = state->x.x[i] - step*state->g.x[i];
        project(&state->z, approx->lower, approx->upper);

        /* y = z - x */
        for (size_t i = 0; i < n; i++)
                state->y.x[i] = state->z.x[i] - state->x.x[i];

        state->z.violationp = 0;
        state->z.quad_value = state->z.value = nan("");
        gradient(&state->g2, approx, &state->z, pool);
        for (size_t i = 0; i < n; i++)
                state->g2.x[i] -= state->g.x[i];
        state->step_length = (dot(state->y.x, &state->y, pool)
                              /dot(state->y.x, &state->g2, pool));
        {
                /* swap */
                struct vector temp = state->z;
                state->z = state->x;
                state->x = temp;
        }

        return 0;
}

static int spg_iter(approx_t * approx, struct approx_state * state,
                    double * OUT_pg, thread_pool_t * pool)
{
        size_t n = approx->nvars;

        /* z = prev x, g2 = prev g */
        project(&state->x, approx->lower, approx->upper);
        gradient(&state->g, approx, &state->x, pool);
        if (OUT_pg != NULL)
                    *OUT_pg = project_gradient_norm(&state->g,
                                                    &state->x,
                                                    approx->lower,
                                                    approx->upper);
        for (size_t i = 0; i < n; i++) {
                state->y.x[i] = state->x.x[i] - state->z.x[i];
                state->u.x[i] = state->g.x[i] - state->g2.x[i];
        }

        double step = (dot(state->y.x, &state->y, pool)
                       /dot(state->y.x, &state->u, pool));

        if (isnan(step)) step = 1;
        if (step > 1e8) return 1;

        /* z = x - step g */
        for (size_t i = 0; i < n; i++)
                state->z.x[i] = state->x.x[i] - step*state->g.x[i];
        project(&state->z, approx->lower, approx->upper);

        {
                /* swap */
                struct vector temp = state->z;
                state->z = state->x;
                state->x = temp;
        }

        {
                /* swap */
                struct vector temp = state->g2;
                state->g2 = state->g;
                state->g = temp;
        }

        return 0;
}
