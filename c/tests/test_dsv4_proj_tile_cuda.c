/* DSV4_PROJ_TILE: the register-tiled bf16 projection kernel (mm_bf16_tile) against
 * the per-matrix kernels it replaces (mm_bf16_batch through
 * dsv4_cuda_matmul_bf16_multi, mm_bf16_batch_rows through
 * dsv4_cuda_matmul_bf16_rows), on the shapes of the prompt's compressor and
 * indexer-compressor projections (rows 1024, 512, 256; 1022 is not a multiple
 * of 16 and keeps the old kernels), on chunks of 128, 77, 16, 9, 3, 2 and 1
 * tokens, with plain and compressor-packed uploads. The outputs must match bit
 * for bit. The switch is read at every call, so the test flips it between
 * calls. Needs a GPU. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../backend_cuda_dsv4.h"

enum { HIDDEN = 4096 };

static unsigned g_seed = 2027;
static unsigned urand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed >> 8;
}
static float frand(void) { return (urand() & 0xffff) / 65536.0f - 0.5f; }
static double now_us(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
static void set_tile(const char *value) {
#ifdef _WIN32
    _putenv_s("DSV4_PROJ_TILE", value);
#else
    setenv("DSV4_PROJ_TILE", value, 1);
#endif
}

static uint16_t *bf16_matrix(int rows) {
    uint16_t *w = malloc((size_t)rows * HIDDEN * sizeof(*w));
    for (size_t i = 0; w && i < (size_t)rows * HIDDEN; i++)
        w[i] = (uint16_t)(((urand() % 2048) + 0x3900) | (urand() & 0x8000));
    return w;
}

typedef int (*proj_fn)(Dsv4CudaTensor *const *, int, const float *, int, float *const *);

int main(void) {
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) { puts("dsv4 proj tile: no CUDA device, skipped"); return 77; }
    static const int shapes[][4] = {{1024, 1024, 256, 256}, {512, 512, 256, 256}, {1024, 1024, 0, 0},
                                    {1022, 256, 0, 0}};
    static const int chunks[] = {128, 77, 16, 9, 3, 2, 1};
    static const proj_fn calls[2] = {dsv4_cuda_matmul_bf16_multi, dsv4_cuda_matmul_bf16_rows};
    float *x = malloc((size_t)128 * HIDDEN * sizeof(*x));
    for (size_t i = 0; x && i < (size_t)128 * HIDDEN; i++) x[i] = frand() * 4.0f;
    long long differ = 0, values = 0;
    int failed = !x;
    for (int s = 0; !failed && s < (int)(sizeof(shapes) / sizeof(shapes[0])); s++)
        for (int packed = 0; !failed && packed < 2; packed++) {
            Dsv4CudaTensor *ts[4] = {0};
            int n = 0;
            for (int m = 0; m < 4 && shapes[s][m]; m++, n++) {
                uint16_t *w = bf16_matrix(shapes[s][m]);
                int up = w && (packed && shapes[s][m] % 32 == 0
                    ? dsv4_cuda_upload_compressor(&ts[m], w, shapes[s][m], HIDDEN, dev)
                    : dsv4_cuda_upload_bf16(&ts[m], w, shapes[s][m], HIDDEN, dev));
                free(w);
                if (!up) { fprintf(stderr, "upload refused (rows %d)\n", shapes[s][m]); failed = 1; }
            }
            for (int c = 0; !failed && c < (int)(sizeof(chunks) / sizeof(chunks[0])); c++)
                for (int k = 0; !failed && k < 2; k++) {
                    int T = chunks[c];
                    float *want[4] = {0}, *got[4] = {0};
                    for (int m = 0; m < n; m++) {
                        want[m] = calloc((size_t)T * shapes[s][m], sizeof(float));
                        got[m] = calloc((size_t)T * shapes[s][m], sizeof(float));
                        if (!want[m] || !got[m]) failed = 1;
                    }
                    set_tile("0");
                    int a = !failed && calls[k](ts, n, x, T, want);
                    set_tile("1");
                    int b = !failed && calls[k](ts, n, x, T, got);
                    set_tile("0");
                    if (!failed && (!a || !b)) {
                        fprintf(stderr, "a projection call refused (shape %d, %d tokens)\n", s, T);
                        failed = 1;
                    }
                    for (int m = 0; !failed && m < n; m++) {
                        size_t count = (size_t)T * shapes[s][m];
                        for (size_t i = 0; i < count; i++)
                            if (memcmp(&want[m][i], &got[m][i], sizeof(float))) {
                                if (!differ)
                                    fprintf(stderr, "first difference: %s rows %d matrix %d tokens %d at %zu: %.9g vs %.9g\n",
                                            k ? "rows" : "multi", shapes[s][m], m, T, i, want[m][i], got[m][i]);
                                differ++;
                            }
                        values += (long long)count;
                    }
                    if (!failed && k && !packed && (T == 128 || T == 16)) {
                        /* one chunk's call, both ways (informative; copies and sync included) */
                        double t0 = now_us();
                        for (int i = 0; i < 50; i++) calls[k](ts, n, x, T, want);
                        double t1 = now_us();
                        set_tile("1");
                        for (int i = 0; i < 50; i++) calls[k](ts, n, x, T, got);
                        set_tile("0");
                        double t2 = now_us();
                        printf("rows %d x%d, %d tokens: per-matrix kernels %.0f us, tiled %.0f us per call\n",
                               shapes[s][0], n, T, (t1 - t0) / 50, (t2 - t1) / 50);
                    }
                    for (int m = 0; m < n; m++) { free(want[m]); free(got[m]); }
                }
            for (int m = 0; m < 4; m++) if (ts[m]) dsv4_cuda_tensor_free(ts[m]);
        }
    free(x);
    dsv4_cuda_shutdown();
    printf("dsv4 proj tile: %lld of %lld outputs differ\n", differ, values);
    if (failed || differ) return 1;
    puts("dsv4 proj tile: OK (bit-identical to the per-matrix kernels)");
    return 0;
}
