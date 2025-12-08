// reconstructor.c
// Parallel reconstruction: loads ASCII model produced by compressor.c and reconstructs an image.
// Uses OpenMP to parallelize model evaluation over the output grid for faster decoding.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

/* ----------------- Hardcoded settings ----------------- */
const char *IN_MODEL = "./model.txt";
const char *OUT_PNG   = "./reconstructed.png";
const int OUT_W = 2048;   // target width (customize)
const int OUT_H = 2048;   // target height (customize)
/* ----------------------------------------------------- */

// Load ASCII model saved by compressor.c
nn load_model_txt(const char *path, int **arch_out, int *arch_count_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open model: %s\n", path);
        exit(1);
    }
    int arch_count = 0;
    if (fscanf(f, "%d", &arch_count) != 1) {
        fprintf(stderr, "Model parse error (arch_count)\n");
        fclose(f); exit(1);
    }

    int *arch = malloc(sizeof(int) * arch_count);
    for (int i = 0; i < arch_count; ++i) {
        if (fscanf(f, "%d", &arch[i]) != 1) { fprintf(stderr, "Model parse arch\n"); fclose(f); exit(1); }
    }

    nn net = nn_alloc(arch, arch_count);

    for (int l = 0; l < net.count; ++l) {
        int rows, cols;
        if (fscanf(f, "%d %d", &rows, &cols) != 2) { fprintf(stderr, "Model parse w dims\n"); fclose(f); exit(1); }
        if (rows != net.w[l].rows || cols != net.w[l].cols) {
            fprintf(stderr, "Model shape mismatch for layer %d weights\n", l); fclose(f); exit(1);
        }
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                float v;
                if (fscanf(f, "%g", &v) != 1) { fprintf(stderr, "Model parse w vals\n"); fclose(f); exit(1); }
                MAT_AT(net.w[l], r, c) = v;
            }

        int brow, bcol;
        if (fscanf(f, "%d %d", &brow, &bcol) != 2) { fprintf(stderr, "Model parse b dims\n"); fclose(f); exit(1); }
        if (brow != net.b[l].rows || bcol != net.b[l].cols) { fprintf(stderr, "Model shape mismatch biases\n"); fclose(f); exit(1); }
        for (int r = 0; r < brow; ++r)
            for (int c = 0; c < bcol; ++c) {
                float v;
                if (fscanf(f, "%g", &v) != 1) { fprintf(stderr, "Model parse b vals\n"); fclose(f); exit(1); }
                MAT_AT(net.b[l], r, c) = v;
            }
    }

    fclose(f);
    *arch_out = arch;
    *arch_count_out = arch_count;
    return net;
}

int main(void) {
    int *arch = NULL;
    int arch_count = 0;
    nn net = load_model_txt(IN_MODEL, &arch, &arch_count);
    printf("Loaded model (arch_count=%d):", arch_count);
    for (int i = 0; i < arch_count; ++i) printf(" %d", arch[i]);
    printf("\n");

    unsigned char *out = malloc((size_t)OUT_W * (size_t)OUT_H);
    if (!out) { fprintf(stderr, "Failed alloc out\n"); return 1; }

    // Parallel evaluation: each thread computes a stripe of rows
    int num_threads = omp_get_max_threads();
    printf("Decoding with %d threads\n", num_threads);

#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < OUT_H; ++y) {
        for (int x = 0; x < OUT_W; ++x) {
            // set input coordinates
            MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (float)(OUT_W - 1);
            MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (float)(OUT_H - 1);
            nn_forward(net);
            float v = MAT_AT(NN_OUTPUT_MAT(net), 0, 0);
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            out[y * OUT_W + x] = (unsigned char)(v * 255.0f);
        }
    }

    if (!stbi_write_png(OUT_PNG, OUT_W, OUT_H, 1, out, OUT_W)) {
        fprintf(stderr, "Failed to write %s\n", OUT_PNG);
        free(out);
        return 1;
    }
    printf("Saved %s (%d x %d)\n", OUT_PNG, OUT_W, OUT_H);
    free(out);
    return 0;
}
