/* V4_PREFILL_RESIDENT: dsv4_cuda_input_alias names a host range as held by a
 * device activation, and the bf16 projections (dsv4_cuda_matmul_bf16_rows /
 * _multi) read their input there in place. The host array named here holds
 * different values than the activation (a decoy): a result equal to the plain
 * call's on the real input proves the kernel read the device copy, and the
 * alias's count says it was used. Also: a sub-range at an offset, a range that
 * runs past the alias (the host is read, nothing is counted), and the alias
 * cleared. The outputs must match bit for bit. Needs a GPU. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

enum { HIDDEN = 4096, TOKENS = 128 };

static unsigned g_seed = 2027;
static unsigned urand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}
static float frand(void) { return (urand() & 0xffff) / 65536.0f - 0.5f; }

static uint16_t *bf16_matrix(int rows) {
    uint16_t *w = malloc((size_t)rows * HIDDEN * sizeof(*w));
    for (size_t i = 0; w && i < (size_t)rows * HIDDEN; i++)
        w[i] = (uint16_t)(((urand() % 2048) + 0x3900) | (urand() & 0x8000));
    return w;
}

static long long g_differ, g_values;
static int compare(const char *what, float *const *want, float *const *got, const int *rows, int n, int tokens) {
    long long before = g_differ;
    for (int m = 0; m < n; m++) {
        size_t count = (size_t)tokens * rows[m];
        for (size_t i = 0; i < count; i++)
            if (memcmp(&want[m][i], &got[m][i], sizeof(float))) {
                if (g_differ == before)
                    fprintf(stderr, "%s: first difference in matrix %d at %zu: %.9g vs %.9g\n",
                            what, m, i, want[m][i], got[m][i]);
                g_differ++;
            }
        g_values += (long long)count;
    }
    return g_differ != before;
}

int main(void) {
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) { puts("dsv4 prefill resident: no CUDA device, skipped"); return 77; }
    static const int rows[4] = {1024, 1024, 256, 256};
    const int n = 4;
    size_t xn = (size_t)TOKENS * HIDDEN;
    float *x = malloc(xn * sizeof(*x)), *decoy = malloc(xn * sizeof(*decoy));
    float *want[4] = {0}, *got[4] = {0};
    Dsv4CudaTensor *ts[4] = {0};
    int failed = !x || !decoy;
    for (size_t i = 0; !failed && i < xn; i++) { x[i] = frand() * 4.0f; decoy[i] = frand() * 4.0f; }
    for (int m = 0; !failed && m < n; m++) {
        uint16_t *w = bf16_matrix(rows[m]);
        want[m] = calloc((size_t)TOKENS * rows[m], sizeof(float));
        got[m] = calloc((size_t)TOKENS * rows[m], sizeof(float));
        if (!w || !want[m] || !got[m] || !dsv4_cuda_upload_compressor(&ts[m], w, rows[m], HIDDEN, dev))
            failed = 1;
        free(w);
    }
    Dsv4CudaActivation *act = failed ? NULL : dsv4_cuda_activation_create(dev, (long long)xn);
    if (!act || !dsv4_cuda_activation_upload(act, x, (long long)xn) || !dsv4_cuda_activation_sync(act))
        failed = 1;
    long long hits;

    /* the whole chunk: the decoy names the activation, the result is x's */
    if (!failed && (!dsv4_cuda_matmul_bf16_multi(ts, n, x, TOKENS, want) ||
                    (dsv4_cuda_input_alias(act, decoy, (long long)xn),
                     !dsv4_cuda_matmul_bf16_rows(ts, n, decoy, TOKENS, got)))) failed = 1;
    hits = dsv4_cuda_input_alias(NULL, NULL, 0);
    if (!failed && (compare("whole chunk", want, got, rows, n, TOKENS) || hits != 1)) {
        fprintf(stderr, "whole chunk: the alias served %lld calls (want 1)\n", hits);
        failed = 1;
    }

    /* 77 tokens from row 16 of the alias, through _multi */
    if (!failed && (!dsv4_cuda_matmul_bf16_multi(ts, n, x + (size_t)16 * HIDDEN, 77, want) ||
                    (dsv4_cuda_input_alias(act, decoy, (long long)xn),
                     !dsv4_cuda_matmul_bf16_multi(ts, n, decoy + (size_t)16 * HIDDEN, 77, got)))) failed = 1;
    hits = dsv4_cuda_input_alias(NULL, NULL, 0);
    if (!failed && (compare("offset rows", want, got, rows, n, 77) || hits != 1)) {
        fprintf(stderr, "offset rows: the alias served %lld calls (want 1)\n", hits);
        failed = 1;
    }

    /* a range past the alias (64 of 128 rows named): the host's decoy is read */
    if (!failed && (!dsv4_cuda_matmul_bf16_multi(ts, n, decoy, TOKENS, want) ||
                    (dsv4_cuda_input_alias(act, decoy, (long long)64 * HIDDEN),
                     !dsv4_cuda_matmul_bf16_rows(ts, n, decoy, TOKENS, got)))) failed = 1;
    hits = dsv4_cuda_input_alias(NULL, NULL, 0);
    if (!failed && (compare("past the alias", want, got, rows, n, TOKENS) || hits != 0)) {
        fprintf(stderr, "past the alias: the alias served %lld calls (want 0)\n", hits);
        failed = 1;
    }

    /* cleared: the host is read again */
    if (!failed && !dsv4_cuda_matmul_bf16_rows(ts, n, decoy, TOKENS, got)) failed = 1;
    if (!failed && compare("cleared", want, got, rows, n, TOKENS)) failed = 1;

    if (act) dsv4_cuda_activation_free(act);
    for (int m = 0; m < n; m++) { if (ts[m]) dsv4_cuda_tensor_free(ts[m]); free(want[m]); free(got[m]); }
    free(decoy); free(x);
    dsv4_cuda_shutdown();
    printf("dsv4 prefill resident: %lld of %lld outputs differ\n", g_differ, g_values);
    if (failed || g_differ) return 1;
    puts("dsv4 prefill resident: OK (the in-place reads give the uploaded input's bits)");
    return 0;
}
