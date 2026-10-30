#ifndef COLIBRI_HYPER_CONNECTIONS_H
#define COLIBRI_HYPER_CONNECTIONS_H

/*
 * Manifold-Constrained Hyper-Connections (mHC), shared by every engine whose
 * checkpoint carries them.
 *
 * mHC replaces the plain residual with `hc_mult` parallel residual streams. At
 * each site a learned map turns the incoming streams into three things:
 *
 *   pre   [H]      how to collapse the H streams into the block's input
 *   post  [H]      where in the streams the block's output is written back
 *   comb  [H,H]    how the streams mix with each other
 *
 * `comb` is projected onto the doubly-stochastic manifold by Sinkhorn
 * normalization, which is the "manifold-constrained" part: the mixing matrix
 * keeps rows and columns summing to one, so the residual pathway cannot
 * silently gain or lose scale layer after layer.
 *
 * The functions here are the arithmetic only. Weight layout, per-layer wiring
 * and any batched or accelerated variant stay with the engine.
 *
 * Two checkpoints use this today and agree down to the config key names
 * (hc_mult, hc_eps, hc_sinkhorn_iters): DeepSeek V4 and GLM-5.3-Flash. This
 * header exists so they share the arithmetic rather than each carrying a copy —
 * copies of shared mechanisms drifting apart is the defect class that keeps
 * recurring in this tree (#720/#748 being the clearest example).
 */

#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <time.h>

/* V4_HC_PAR (1): how the two steps use OpenMP. 0 = every loop on the
 * calling thread; 1 = only the mix (the ~400k-MAC row loop) on a team no
 * larger than its row count, the column loops on the calling thread (a
 * decode token's 4 x 4096 column work is a few microseconds, a 32-thread
 * fork and join several times that); 2 = every loop on the full team.
 * Each row's and each column's summation order is the same in all three:
 * bitwise. */
static inline int coli_hc_par(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("V4_HC_PAR");
        mode = e && *e ? atoi(e) : 1;
        if (mode < 0) mode = 0;
        if (mode > 2) mode = 2;
    }
    return mode;
}
/* V4_HC_MIX_GROUP (8): coli_hc_pre's mix rows in groups of this many, one
 * thread per group running its rows' chains interleaved in one column
 * loop (an accumulator per row: each row's own serial order, bitwise), and
 * one more task for the mean square (V4_HC_MIX_TEAM: on the calling
 * thread by default). DSV4_HC_PROF: the 24-thread row-per-thread mix
 * took 145 us per call, of which the work ~20 and the rest the wake and
 * join of 24 threads; the grouped mix on one thread 108 us, a thread's
 * eight chains pipelining where one chain was latency-bound. 0 = the
 * row-per-thread loop. */
static inline int coli_hc_mix_group(void) {
    static int group = -1;
    if (group < 0) {
        const char *e = getenv("V4_HC_MIX_GROUP");
        group = e && *e ? atoi(e) : 8;
        if (group < 0) group = 0;
        if (group > 8) group = 8;
    }
    return group;
}
/* V4_HC_MIX_TEAM (1): the grouped mix's team; 1 = the calling thread
 * alone, no OpenMP region (measured equal to a team of groups + 1: the
 * fork and join cost what the threads saved, and the single thread's
 * three groups of eight chains are memory-bound at ~108 us per call);
 * 0 = groups + 1 threads. */
static inline int coli_hc_mix_team(int groups) {
    static int team = -1;
    if (team < 0) {
        const char *e = getenv("V4_HC_MIX_TEAM");
        team = e && *e ? atoi(e) : 1;
        if (team < 0) team = 0;
    }
    return team ? team : groups + 1;
}
static inline int coli_hc_team(int rows) {
#ifdef _OPENMP
    int team = omp_get_max_threads();
    if (coli_hc_par() == 1 && rows < team) team = rows;
    return team > 0 ? team : 1;
#else
    (void)rows;
    return 1;
#endif
}

static inline float coli_hc_sigmoid(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

/* Split the mix logits into pre/post/comb and project comb onto the
 * doubly-stochastic manifold. `mixes` is [(2+hc)*hc], `scale` is the three
 * learned scales, `base` the matching biases. Returns 0, or -1 on bad
 * arguments or allocation failure. */
static inline int coli_hc_split_sinkhorn(float *pre, float *post, float *comb,
                                         const float *mixes, const float scale[3],
                                         const float *base, int hc, int iterations,
                                         float eps) {
    if (!pre || !post || !comb || !mixes || !scale || !base ||
        hc < 1 || iterations < 1 || eps < 0.0f)
        return -1;
    for (int index = 0; index < hc; index++) {
        pre[index] = coli_hc_sigmoid(
            mixes[index] * scale[0] + base[index]) + eps;
        post[index] = 2.0f * coli_hc_sigmoid(
            mixes[hc + index] * scale[1] + base[hc + index]);
    }
    int matrix_offset = 2 * hc;
    for (int row = 0; row < hc; row++) {
        float maximum = -INFINITY;
        for (int column = 0; column < hc; column++) {
            int index = matrix_offset + row * hc + column;
            float value = mixes[index] * scale[2] + base[index];
            comb[row * hc + column] = value;
            if (value > maximum) maximum = value;
        }
        float sum = 0.0f;
        for (int column = 0; column < hc; column++) {
            float value = expf(comb[row * hc + column] - maximum);
            comb[row * hc + column] = value;
            sum += value;
        }
        for (int column = 0; column < hc; column++)
            comb[row * hc + column] = comb[row * hc + column] / sum + eps;
    }
    float *sums = malloc((size_t)hc * sizeof(*sums));
    if (!sums) return -1;
    for (int column = 0; column < hc; column++) {
        float sum = 0.0f;
        for (int row = 0; row < hc; row++)
            sum += comb[row * hc + column];
        sums[column] = sum;
    }
    for (int row = 0; row < hc; row++)
        for (int column = 0; column < hc; column++)
            comb[row * hc + column] /= sums[column] + eps;

    for (int iteration = 1; iteration < iterations; iteration++) {
        for (int row = 0; row < hc; row++) {
            float sum = 0.0f;
            for (int column = 0; column < hc; column++)
                sum += comb[row * hc + column];
            sums[row] = sum;
        }
        for (int row = 0; row < hc; row++)
            for (int column = 0; column < hc; column++)
                comb[row * hc + column] /= sums[row] + eps;
        for (int column = 0; column < hc; column++) {
            float sum = 0.0f;
            for (int row = 0; row < hc; row++)
                sum += comb[row * hc + column];
            sums[column] = sum;
        }
        for (int row = 0; row < hc; row++)
            for (int column = 0; column < hc; column++)
                comb[row * hc + column] /= sums[column] + eps;
    }
    free(sums);
    return 0;
}

/* Entering a block: RMS-rescale the flattened streams, map them to the mix
 * logits through `hc_fn` [(2+hc)*hc, hc*dimension], split, and collapse the
 * streams into `output` [dimension] with the pre weights. `post` and `comb`
 * are handed back for the matching coli_hc_post(). */
/* DSV4_HC_PROF=1: coli_hc_pre's five parts timed, totals printed every
 * 1000 calls (`hcprof`). */
static inline int coli_hc_prof_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("DSV4_HC_PROF"); on = e && *e && *e != '0'; }
    return on;
}
static inline double coli_hc_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static inline void coli_hc_prof_add(int part, double sec) {
    static double acc[5];
    static long long calls;
    acc[part] += sec;
    if (part == 4 && ++calls % 1000 == 0)
        fprintf(stderr, "hcprof calls=%lld rms=%.1f mix=%.1f sinkhorn=%.1f collapse=%.1f alloc=%.1f us/call\n",
                calls, acc[0] / calls * 1e6, acc[1] / calls * 1e6, acc[2] / calls * 1e6,
                acc[3] / calls * 1e6, acc[4] / calls * 1e6);
}

static inline int coli_hc_pre(float *output, float *post, float *comb,
                              const float *input, const float *hc_fn,
                              const float scale[3], const float *base,
                              int hc, int dimension, int iterations,
                              float norm_eps, float hc_eps) {
    if (!output || !post || !comb || !input || !hc_fn || !scale || !base ||
        hc < 1 || dimension < 1 || norm_eps < 0.0f)
        return -1;
    int flattened = hc * dimension;
    int mix_count = (2 + hc) * hc;
    int prof = coli_hc_prof_on();
    double t0 = prof ? coli_hc_now() : 0, t1;
    int group = coli_hc_par() > 0 ? coli_hc_mix_group() : 0;
    float mean_square = 0.0f;
    if (!group)
        for (int index = 0; index < flattened; index++)
            mean_square += input[index] * input[index];
    float inverse_rms = group ? 0.0f : 1.0f / sqrtf(mean_square / flattened + norm_eps);
    if (prof) { t1 = coli_hc_now(); coli_hc_prof_add(0, t1 - t0); t0 = t1; }
    /* These two mallocs look like free money to remove: hc_mult is bounded to
     * 8 by every loader, so fixed-size stack arrays would save an allocation
     * at ~90 call sites per token. It was tried and it is measurably NOT
     * numerics-neutral. Once hc's range is knowable at compile time -- whether
     * from the array size itself or from an `hc <= 8` check added defensively
     * -- GCC re-rounds the reductions below. Bisected across
     * -fno-tree-vectorize, -ffp-contract=off, #pragma GCC unroll 1, an
     * aliasing barrier and __attribute__((noipa)); none restored bit-identity,
     * while keeping the mallocs and adding only the parallelism is
     * bit-identical (checked directly). So the mallocs stay. */
    float *mixes = malloc((size_t)mix_count * sizeof(*mixes));
    float *pre = malloc((size_t)hc * sizeof(*pre));
    if (!mixes || !pre) {
        free(mixes);
        free(pre);
        return -1;
    }
    if (prof) { t1 = coli_hc_now(); coli_hc_prof_add(4, t1 - t0); t0 = t1; }
    /* The ~400k-MAC mix (mix_count independent dot products, 24 rows at
     * hc=4): each row's own untouched scalar reduction, so parallelising
     * over rows changes nothing about any one row's summation order --
     * this distributes whole rows across threads and is not a SIMD reduction
     * reorder. Nested parallelism was checked separately and is safe here. */
    if (group) {
        /* V4_HC_MIX_GROUP: see coli_hc_mix_group */
        int groups = (mix_count + group - 1) / group;
        float mean_square_task = 0.0f;
        int team = coli_hc_mix_team(groups);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static, 1) if(team > 1) num_threads(team)
#endif
        for (int task = 0; task <= groups; task++) {
            if (task == groups) {
                float ms = 0.0f;
                for (int index = 0; index < flattened; index++)
                    ms += input[index] * input[index];
                mean_square_task = ms;
                continue;
            }
            int r0 = task * group;
            int rn = mix_count - r0 < group ? mix_count - r0 : group;
            const float *rows[8];
            float sums[8];
            for (int r = 0; r < 8; r++) {
                rows[r] = hc_fn + (size_t)(r0 + (r < rn ? r : 0)) * flattened;
                sums[r] = 0.0f;
            }
            for (int column = 0; column < flattened; column++) {
                float x = input[column];
                for (int r = 0; r < 8; r++)
                    sums[r] += rows[r][column] * x;
            }
            for (int r = 0; r < rn; r++) mixes[r0 + r] = sums[r];
        }
        mean_square = mean_square_task;
        inverse_rms = 1.0f / sqrtf(mean_square / flattened + norm_eps);
        for (int row = 0; row < mix_count; row++)
            mixes[row] = mixes[row] * inverse_rms;
    } else {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(coli_hc_par() > 0) num_threads(coli_hc_team(mix_count))
#endif
    for (int row = 0; row < mix_count; row++) {
        float sum = 0.0f;
        for (int column = 0; column < flattened; column++)
            sum += hc_fn[(size_t)row * flattened + column] * input[column];
        mixes[row] = sum * inverse_rms;
    }
    }
    if (prof) { t1 = coli_hc_now(); coli_hc_prof_add(1, t1 - t0); t0 = t1; }
    if (coli_hc_split_sinkhorn(pre, post, comb, mixes, scale, base,
                               hc, iterations, hc_eps) != 0) {
        free(pre);
        free(mixes);
        return -1;
    }
    if (prof) { t1 = coli_hc_now(); coli_hc_prof_add(2, t1 - t0); t0 = t1; }
    /* V4_HC_PAR <= 1: a plain loop, so the compiler vectorizes across
     * columns (each column's own short chain, in its order: bitwise); the
     * OpenMP form, even with its `if` false, is outlined and stays scalar. */
#ifdef _OPENMP
    if (coli_hc_par() > 1) {
        #pragma omp parallel for schedule(static)
        for (int column = 0; column < dimension; column++) {
            float sum = 0.0f;
            for (int copy = 0; copy < hc; copy++)
                sum += pre[copy] * input[copy * dimension + column];
            output[column] = sum;
        }
    } else
#endif
    for (int column = 0; column < dimension; column++) {
        float sum = 0.0f;
        for (int copy = 0; copy < hc; copy++)
            sum += pre[copy] * input[copy * dimension + column];
        output[column] = sum;
    }
    if (prof) { t1 = coli_hc_now(); coli_hc_prof_add(3, t1 - t0); }
    free(pre);
    free(mixes);
    return 0;
}

/* Leaving a block: mix the incoming streams through `comb` and add the block's
 * output where `post` says it goes. `output` and `residual` are [hc*dimension]
 * and may not overlap. */
static inline int coli_hc_post(float *output, const float *branch,
                               const float *residual, const float *post,
                               const float *comb, int hc, int dimension) {
    if (!output || !branch || !residual || !post || !comb ||
        hc < 1 || dimension < 1)
        return -1;
    /* destination x column is embarrassingly parallel (disjoint output);
     * collapse(2) gives full-width parallelism regardless of hc's small
     * size. The inner source-reduction (hc terms) keeps its own order. */
#ifdef _OPENMP
    if (coli_hc_par() > 1) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int destination = 0; destination < hc; destination++) {
            for (int column = 0; column < dimension; column++) {
                float value = 0.0f;
                for (int source = 0; source < hc; source++)
                    value += comb[source * hc + destination] *
                             residual[source * dimension + column];
                value += post[destination] * branch[column];
                output[destination * dimension + column] = value;
            }
        }
        return 0;
    }
#endif
    /* V4_HC_PAR <= 1: the plain loops (see coli_hc_pre): vectorized across
     * columns, each element's chain in its order. */
    for (int destination = 0; destination < hc; destination++) {
        for (int column = 0; column < dimension; column++) {
            float value = 0.0f;
            for (int source = 0; source < hc; source++)
                value += comb[source * hc + destination] *
                         residual[source * dimension + column];
            value += post[destination] * branch[column];
            output[destination * dimension + column] = value;
        }
    }
    return 0;
}

#endif /* COLIBRI_HYPER_CONNECTIONS_H */
