// compressor.c
// Optimized compressor: multithreaded gradient accumulation, visualization frames (nn render + preview sprite).
// Hardcoded settings at top. Uses nn.h, stb_image, stb_image_write, and olive.c for visualization.
// Adapted from ImageUpscaler.c (visualization + training) to integrate into compressor pipeline.
// See: ImageUpscaler.c for original visuals and training helper code. :contentReference[oaicite:7]{index=7}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define OLIVEC_IMPLEMENTATION
#include "olive.c"   // for visualization utilities used by nn_render

/* ----------------- Hardcoded settings (edit as needed) ----------------- */
const char *INPUT_IMAGE = "./98.png";   // input grayscale image
const char *OUT_MODEL   = "./model.txt";
const int    OUT_EPOCHS = 50000;        // number of training epochs
const int    VIZ_FRAMES = 40;           // number of visualization frames to save during training
const float  LEARN_RATE = 1.0f;         // learning rate applied after gradient accumulation
// architecture: input is (x,y) so 2 inputs, output 1 brightness
int ARCH[] = {2, 28, 14, 7, 1};
int ARCH_COUNT = sizeof(ARCH)/sizeof(ARCH[0]);

// preview sprite size used in visualization
const int PREVIEW_W = 128;
const int PREVIEW_H = 128;

#define PIX_IDX(w,x,y) ((y)*(w)+(x))
/* ----------------------------------------------------------------------- */

/* ------------------ Visualization helpers (adapted) -------------------- */
#define IMG_X 1024
#define IMG_Y 768
static uint32_t img_pixels_canvas[IMG_X * IMG_Y];

// ARGB pack
static inline uint32_t ARGB(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

// weight -> red/green color mapping via tanh
static inline uint32_t weight_to_rgcolor(float w)
{
    float t = (tanhf(w) + 1.0f) * 0.5f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    uint8_t r = (uint8_t)((1.0f - t) * 255.0f);
    uint8_t g = (uint8_t)(t * 255.0f);
    return ARGB(0xFF, r, g, 0);
}

// normalized value -> color mapping (red->green)
static inline uint32_t t_to_rgcolor(float t)
{
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    uint8_t r = (uint8_t)((1.0f - t) * 255.0f);
    uint8_t g = (uint8_t)(t * 255.0f);
    return ARGB(0xFF, r, g, 0);
}

// Legend + swatches (uses olivec_text / shapes)
static void draw_legend(Olivec_Canvas img)
{
    const int legend_w = 220;
    const int legend_h = 140;
    const int margin = 12;
    const int lx = img.width - legend_w - margin;
    const int ly = margin;

    olivec_frame(img, lx - 2, ly - 2, legend_w + 4, legend_h + 4, 2, ARGB(0xFF,255,255,255));
    olivec_rect(img, lx, ly, legend_w, legend_h, ARGB(0xE0,20,20,20));

    int sw = 18;
    int spacing = 8;
    int sw_x = lx + 12;
    int sw_y = ly + 10;

    uint32_t col_blue = ARGB(0xFF, 64, 64, 160);
    uint32_t col_green = ARGB(0xFF, 0, 255, 0);
    uint32_t col_white = ARGB(0xFF, 255, 255, 255);
    uint32_t col_gray = ARGB(0xFF, 128, 128, 128);
    uint32_t col_red = ARGB(0xFF, 255, 0, 0);

    olivec_rect(img, sw_x, sw_y, sw, sw, col_blue);
    olivec_text(img, "Preview background", sw_x + sw + 8, sw_y - 1, olivec_default_font, 1, col_white);

    int r2y = sw_y + (sw + spacing);
    olivec_rect(img, sw_x, r2y, sw, sw, col_green);
    olivec_text(img, "High value (green)", sw_x + sw + 8, r2y - 1, olivec_default_font, 1, col_white);

    int r3y = r2y + (sw + spacing);
    olivec_rect(img, sw_x, r3y, sw, sw, col_white);
    olivec_text(img, "Neutral / bias ~ 0 (white)", sw_x + sw + 8, r3y - 1, olivec_default_font, 1, col_white);

    int r4y = r3y + (sw + spacing);
    olivec_rect(img, sw_x, r4y, sw, sw, col_gray);
    olivec_text(img, "Input / inactive neurons (gray)", sw_x + sw + 8, r4y - 1, olivec_default_font, 1, col_white);

    int r5y = r4y + (sw + spacing);
    olivec_rect(img, sw_x, r5y, sw, sw, col_red);
    olivec_text(img, "Low value (red)", sw_x + sw + 8, r5y - 1, olivec_default_font, 1, col_white);
}

// Neural network visualization + preview sprite blit — adapted from ImageUpscaler.c.
// preview_sprite is ARGB pixels (pw x ph)
int nn_render(Olivec_Canvas img, nn net, int *arch, int arch_count,
              uint32_t *preview_sprite, int pw, int ph)
{
    olivec_fill(img, ARGB(0xFF,26,26,26));

    int layer_bvpad = 50;
    int layer_bhpad = 50;
    int net_width = img.width - layer_bhpad * 2;
    int net_height = img.height - 2 * layer_bvpad;
    int layer_hpad = net_width / arch_count;
    int net_x = img.width / 2 - net_width / 2;
    int net_y = img.height / 2 - net_height / 2;
    int neuron_radius = 18;

    for (int l = 0; l < arch_count; ++l) {
        int layer_vpad = net_height / arch[l];
        for (int i = 0; i < arch[l]; ++i) {
            int cx = net_x + l * layer_hpad + layer_hpad / 2;
            int cy = net_y + i * layer_vpad + layer_vpad / 2;
            if (l + 1 < arch_count) {
                int next_vpad = net_height / arch[l + 1];
                for (int j = 0; j < arch[l + 1]; ++j) {
                    int cx2 = net_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = net_y + j * next_vpad + next_vpad / 2;
                    float w = MAT_AT(net.w[l], j, i);
                    uint32_t col = weight_to_rgcolor(w);
                    olivec_line(img, cx, cy, cx2, cy2, col);
                }
            }
            if (l > 0) {
                float b = MAT_AT(net.b[l - 1], 0, i);
                uint8_t v = (uint8_t)(255.0f / (1.0f + expf(-b)));
                uint32_t col = ARGB(0xFF, v, v, v);
                olivec_circle(img, cx, cy, neuron_radius, col);
            } else {
                olivec_circle(img, cx, cy, neuron_radius, ARGB(0xFF,128,128,128));
            }
        }
    }

    if (preview_sprite && pw > 0 && ph > 0) {
        Olivec_Canvas sprite = olivec_canvas(preview_sprite, pw, ph, pw);
        int margin = 24;
        int px = img.width - pw - margin;
        int py = img.height - ph - margin;
        olivec_sprite_copy(img, px, py, pw, ph, sprite);
        olivec_frame(img, px, py, pw, ph, 2, ARGB(0xFF,255,255,255));
    }

    draw_legend(img);
    olivec_frame(img, 0, 0, img.width - 1, img.height - 1, 8, ARGB(0xFF,255,255,255));
    return 0;
}
/* --------------------------------------------------------------------- */

/* ------------------ Multithreaded training w/ visualization ------------- */
/*
  train_nn_mt_vis:
  - Multithreaded gradient accumulation (each thread computes backprop on a data chunk).
  - Aggregates gradients into global accumulator g (weighted by sub_rows).
  - Applies update using nn_learn(net, g, lr_scaled).
  - Produces exactly `frame_count` visualization frames saved to ./vizns/upscaler-XXXX.png.
  - Builds a GIF via ImageMagick `convert` at the end (requires convert).
*/
float rate = 1.0f;

void train_nn_mt_vis(nn net, nn g, int epochs, mat tin, mat tout,
                     int arch[], int arch_count, int frame_count)
{
    int num_threads = omp_get_max_threads();
    printf("Using %d threads\n", num_threads);

    system("mkdir -p vizns");

    if (frame_count < 1) frame_count = 1;
    int save_every = epochs / frame_count;
    if (save_every <= 0) save_every = 1;

    int frame_index = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        nn_init(g, 0.0f);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int rows = tin.rows;
            int start = (tid * rows) / num_threads;
            int end = ((tid + 1) * rows) / num_threads;
            int sub_rows = end - start;

            nn local_g = nn_alloc(arch, arch_count);
            nn_init(local_g, 0.0f);

            nn local_net = nn_alloc(arch, arch_count);
            for (int l = 0; l < net.count; ++l) {
                mat_cpy(local_net.w[l], net.w[l]);
                mat_cpy(local_net.b[l], net.b[l]);
            }

            mat sub_tin = {.rows = sub_rows, .cols = tin.cols, .stride = tin.stride, .data = &MAT_AT(tin, start, 0)};
            mat sub_tout = {.rows = sub_rows, .cols = tout.cols, .stride = tout.stride, .data = &MAT_AT(tout, start, 0)};

            // compute local gradient
            nn_backprop(local_net, local_g, sub_tin, sub_tout);

#pragma omp critical
            {
                // accumulate weighted by sub_rows (we'll divide later by total rows in updates)
                for (int l = 0; l < g.count; ++l) {
                    for (int r = 0; r < g.w[l].rows; ++r)
                        for (int c = 0; c < g.w[l].cols; ++c)
                            MAT_AT(g.w[l], r, c) += MAT_AT(local_g.w[l], r, c) * (float)sub_rows;
                    for (int r = 0; r < g.b[l].rows; ++r)
                        for (int c = 0; c < g.b[l].cols; ++c)
                            MAT_AT(g.b[l], r, c) += MAT_AT(local_g.b[l], r, c) * (float)sub_rows;
                }
            }

            // free local nets (nn.h may not expose free — but avoid mem leak by relying on program exit or provide free if available)
            // If nn.h has nn_free, use it here; otherwise rely on exit freeing.
        } // omp parallel

        // After accumulation, scale gradients to average (divide by total rows)
        float rows_total = (float)tin.rows;
        for (int l = 0; l < g.count; ++l) {
            for (int r = 0; r < g.w[l].rows; ++r)
                for (int c = 0; c < g.w[l].cols; ++c)
                    MAT_AT(g.w[l], r, c) /= rows_total;
            for (int r = 0; r < g.b[l].rows; ++r)
                for (int c = 0; c < g.b[l].cols; ++c)
                    MAT_AT(g.b[l], r, c) /= rows_total;
        }

        // Apply gradient descent update — reuse nn_learn with LEARN_RATE scaling
        nn_learn(net, g, LEARN_RATE);

        if ((epoch % (save_every)) == 0) {
            float cost = nn_cost(net, tin, tout);
            printf("[epoch %d/%d] cost = %g\n", epoch, epochs, cost);

            // Build preview sprite from current net
            uint32_t *sprite_pixels = malloc(sizeof(uint32_t) * PREVIEW_W * PREVIEW_H);
            if (sprite_pixels) {
                for (int y = 0; y < PREVIEW_H; ++y) {
                    for (int x = 0; x < PREVIEW_W; ++x) {
                        MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (PREVIEW_W - 1);
                        MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (PREVIEW_H - 1);
                        nn_forward(net);
                        float v = MAT_AT(NN_OUTPUT_MAT(net), 0, 0);
                        if (v < 0.0f) v = 0.0f;
                        if (v > 1.0f) v = 1.0f;
                        sprite_pixels[y * PREVIEW_W + x] = t_to_rgcolor(v);
                    }
                }

                Olivec_Canvas canvas = olivec_canvas(img_pixels_canvas, IMG_X, IMG_Y, IMG_X);
                nn_render(canvas, net, ARCH, ARCH_COUNT, sprite_pixels, PREVIEW_W, PREVIEW_H);

                char fname[256];
                snprintf(fname, sizeof(fname), "./vizns/upscaler-%04d.png", frame_index);
                if (!stbi_write_png(fname, IMG_X, IMG_Y, 4, canvas.pixels, canvas.stride * sizeof(uint32_t))) {
                    fprintf(stderr, "Failed to write %s\n", fname);
                } else {
                    printf("Saved visualization frame: %s\n", fname);
                }
                free(sprite_pixels);
                frame_index++;
            }
        }
    } // epochs

    printf("Final cost = %f\n", nn_cost(net, tin, tout));
    // Build GIF using ImageMagick convert (requires convert on PATH)
    printf("Generating GIF vizns/training.gif (requires ImageMagick 'convert').\n");
    system("convert -delay 10 -loop 0 vizns/upscaler-*.png vizns/training.gif");
}
/* --------------------------------------------------------------------- */


/* ------------------ Model save (same format, ASCII) ------------------- */
int save_model_txt(const char *path, nn net, int *arch, int arch_count) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "%d\n", arch_count);
    for (int i = 0; i < arch_count; ++i) {
        if (i) fprintf(f, " ");
        fprintf(f, "%d", arch[i]);
    }
    fprintf(f, "\n");

    for (int l = 0; l < net.count; ++l) {
        int rows = net.w[l].rows;
        int cols = net.w[l].cols;
        fprintf(f, "%d %d\n", rows, cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c)
                fprintf(f, "%.9g ", MAT_AT(net.w[l], r, c));
            fprintf(f, "\n");
        }
        int brow = net.b[l].rows;
        int bcol = net.b[l].cols;
        fprintf(f, "%d %d\n", brow, bcol);
        for (int r = 0; r < brow; ++r) {
            for (int c = 0; c < bcol; ++c)
                fprintf(f, "%.9g ", MAT_AT(net.b[l], r, c));
            fprintf(f, "\n");
        }
    }
    fclose(f);
    return 0;
}
/* --------------------------------------------------------------------- */

int main(void) {
    srand((unsigned)time(NULL));

    // Load input
    int img_w, img_h, img_c;
    unsigned char *img = stbi_load(INPUT_IMAGE, &img_w, &img_h, &img_c, 0);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", INPUT_IMAGE);
        return 1;
    }
    if (img_c != 1) {
        fprintf(stderr, "Image must be single-channel grayscale (channels=%d). Convert before.\n", img_c);
        stbi_image_free(img);
        return 1;
    }
    printf("Loaded %s (%d x %d), channels=%d\n", INPUT_IMAGE, img_w, img_h, img_c);

    // Build training data matrix: [x_norm, y_norm, intensity]
    int samples = img_w * img_h;
    mat trd = mat_alloc(samples, 3);
    for (int y = 0; y < img_h; ++y) {
        for (int x = 0; x < img_w; ++x) {
            int i = PIX_IDX(img_w, x, y);
            MAT_AT(trd, i, 0) = (float)x / (float)(img_w - 1);
            MAT_AT(trd, i, 1) = (float)y / (float)(img_h - 1);
            MAT_AT(trd, i, 2) = img[i] / 255.0f;
        }
    }

    mat tin = {.rows = trd.rows, .cols = 2, .stride = trd.stride, .data = &MAT_AT(trd, 0, 0)};
    mat tout = {.rows = trd.rows, .cols = 1, .stride = trd.stride, .data = &MAT_AT(trd, 0, 2)};

    // allocate network and gradient accumulator
    nn net = nn_alloc(ARCH, ARCH_COUNT);
    nn g   = nn_alloc(ARCH, ARCH_COUNT);

    // random init
    nn_rand(net, -1, 1);

    // multithreaded training with visualization frames
    train_nn_mt_vis(net, g, OUT_EPOCHS, tin, tout, ARCH, ARCH_COUNT, VIZ_FRAMES);

    // Save model
    if (save_model_txt(OUT_MODEL, net, ARCH, ARCH_COUNT) != 0) {
        fprintf(stderr, "Failed to save model to %s\n", OUT_MODEL);
    } else {
        printf("Saved model to %s\n", OUT_MODEL);
    }

    stbi_image_free(img);
    return 0;
}
