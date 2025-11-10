//to Run:  
//   gcc ImageUpscaler.c -o ImageUpscaler -fopenmp -lm
//   ./ImageUpscaler

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define OLIVEC_IMPLEMENTATION
#include "olive.c"

#define IMG_X 1024
#define IMG_Y 768
static uint32_t img_pixels_canvas[IMG_X * IMG_Y];

#define pix(x, y) ((y) * img_width + (x))

// Helper: pack color with alpha in top byte (consistent with earlier code)
// Format used in original project: 0xAARRGGBB where R is bits16..23, G bits8..15, B bits0..7
static inline uint32_t ARGB(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

// Map weight to t in [0,1] via tanh then produce ARGB color red->green (blue=0)
static inline uint32_t weight_to_rgcolor(float w)
{
    float t = (tanhf(w) + 1.0f) * 0.5f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    uint8_t r = (uint8_t)((1.0f - t) * 255.0f);
    uint8_t g = (uint8_t)(t * 255.0f);
    return ARGB(0xFF, r, g, 0);
}

// Map normalized t [0,1] to red->green color
static inline uint32_t t_to_rgcolor(float t)
{
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    uint8_t r = (uint8_t)((1.0f - t) * 255.0f);
    uint8_t g = (uint8_t)(t * 255.0f);
    return ARGB(0xFF, r, g, 0);
}

// Draw legend (top-right), using only olive.c functions.
// Legend shows gradient bar and text labels using olivec_text.
static void draw_legend(Olivec_Canvas img)
{
    const int legend_w = 220;
    const int legend_h = 140;
    const int margin = 12;
    const int lx = img.width - legend_w - margin;
    const int ly = margin;

    // Border
    olivec_frame(img, lx - 2, ly - 2, legend_w + 4, legend_h + 4, 2, ARGB(0xFF, 255, 255, 255));

    // Background (slightly transparent black)
    olivec_rect(img, lx, ly, legend_w, legend_h, ARGB(0xE0, 20, 20, 20));

    // Gradient bar top area (left-to-right red->green)
    const int gbar_x = lx + 10;
    const int gbar_y = ly + 10;
    const int gbar_w = legend_w - 20;
    const int gbar_h = 18;
    // Fill pixels directly for gradient
    for (int gx = 0; gx < gbar_w; ++gx) {
        float t = (float)gx / (gbar_w - 1);
        uint32_t c = t_to_rgcolor(t);
        for (int gy = 0; gy < gbar_h; ++gy) {
            if ((gbar_x + gx) >= 0 && (gbar_x + gx) < (int)img.width && (gbar_y + gy) >= 0 && (gbar_y + gy) < (int)img.height)
                OLIVEC_PIXEL(img, gbar_x + gx, gbar_y + gy) = c;
        }
    }
    olivec_frame(img, gbar_x, gbar_y, gbar_w, gbar_h, 1, ARGB(0xFF, 255, 255, 255));

    // Legend swatches and labels below gradient
    int sw_x = gbar_x;
    int sw_y = gbar_y + gbar_h + 10;
    int sw = 12;
    int spacing = 6;
    uint32_t col_blue = ARGB(0xFF, 0, 0, 255);
    uint32_t col_white = ARGB(0xFF, 255, 255, 255);
    uint32_t col_gray = ARGB(0xFF, 128, 128, 128);
    uint32_t col_red = ARGB(0xFF, 255, 0, 0);
    uint32_t col_green = ARGB(0xFF, 0, 255, 0);

    // Row 1: Blue swatch -> "Preview background"
    olivec_rect(img, sw_x, sw_y, sw, sw, col_blue);
    olivec_text(img, "Preview background", sw_x + sw + 8, sw_y - 1, olivec_default_font, 1, ARGB(0xFF, 255, 255, 255));

    // Row 2: Green swatch -> "High value (green)"
    int r2y = sw_y + (sw + spacing);
    olivec_rect(img, sw_x, r2y, sw, sw, col_green);
    olivec_text(img, "High value (green)", sw_x + sw + 8, r2y - 1, olivec_default_font, 1, ARGB(0xFF, 255, 255, 255));

    // Row 3: White swatch -> "Neutral / bias ~ 0"
    int r3y = r2y + (sw + spacing);
    olivec_rect(img, sw_x, r3y, sw, sw, col_white);
    olivec_text(img, "Neutral / bias ~ 0 (white)", sw_x + sw + 8, r3y - 1, olivec_default_font, 1, ARGB(0xFF, 255, 255, 255));

    // Row 4: Gray swatch -> "Input neurons"
    int r4y = r3y + (sw + spacing);
    olivec_rect(img, sw_x, r4y, sw, sw, col_gray);
    olivec_text(img, "Input / inactive neurons (gray)", sw_x + sw + 8, r4y - 1, olivec_default_font, 1, ARGB(0xFF, 255, 255, 255));

    // Row 5: Red swatch -> "Low value (red)"
    int r5y = r4y + (sw + spacing);
    olivec_rect(img, sw_x, r5y, sw, sw, col_red);
    olivec_text(img, "Low value (red)", sw_x + sw + 8, r5y - 1, olivec_default_font, 1, ARGB(0xFF, 255, 255, 255));
}

// Render network visualization and preview
int nn_render(Olivec_Canvas img, nn net, int *arch, int arch_count,
              uint32_t *preview_sprite, int pw, int ph)
{
    // Fill background
    olivec_fill(img, ARGB(0xFF, 26, 26, 26)); // dark gray

    // Layout
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
            // draw lines to next layer
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
            // draw neuron circle (bias represented as grayscale)
            if (l > 0) {
                float b = MAT_AT(net.b[l - 1], 0, i);
                // map bias via logistic to 0..255
                uint8_t v = (uint8_t)(255.0f / (1.0f + expf(-b)));
                uint32_t col = ARGB(0xFF, v, v, v);
                olivec_circle(img, cx, cy, neuron_radius, col);
            } else {
                // input layer
                olivec_circle(img, cx, cy, neuron_radius, ARGB(0xFF, 128, 128, 128));
            }
        }
    }

    // Preview: build sprite (ARGB) and blit via olivec_sprite_copy for correct alignment/clipping
    if (preview_sprite && pw > 0 && ph > 0) {
        Olivec_Canvas sprite = olivec_canvas(preview_sprite, pw, ph, pw);
        // place it with margin from edges (bottom-right)
        int margin = 24;
        int px = img.width - pw - margin;
        int py = img.height - ph - margin;
        olivec_sprite_copy(img, px, py, pw, ph, sprite);
        // frame around preview
        olivec_frame(img, px, py, pw, ph, 2, ARGB(0xFF, 255, 255, 255));
    }

    // Draw legend (top-right)
    draw_legend(img);

    // Outer frame
    olivec_frame(img, 0, 0, img.width - 1, img.height - 1, 8, ARGB(0xFF, 255, 255, 255));
    return 0;
}

// Training function: multithreaded gradient accumulation (from upscaler_fast), with visualization frames.
// frame_count : number of frames to save during training (we will produce exactly that many frames)
float rate = 1.0f;
void train_nn_mt_vis(nn net, nn g, int epochs, mat tin, mat tout,
                     int arch[], int arch_count, int frame_count)
{
    int num_threads = omp_get_max_threads();
    printf("Using %d threads\n", num_threads);

    // prepare output dir
    system("mkdir -p vizns");

    // compute save schedule: we want exactly frame_count frames evenly spaced including epoch 0
    if (frame_count < 1) frame_count = 1;
    int save_every = epochs / frame_count;
    if (save_every <= 0) save_every = 1;

    int frame_index = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // zero global gradient accumulator
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

            nn_backprop(local_net, local_g, sub_tin, sub_tout);

#pragma omp critical
            {
                for (int l = 0; l < g.count; ++l) {
                    for (int r = 0; r < g.w[l].rows; ++r)
                        for (int c = 0; c < g.w[l].cols; ++c)
                            MAT_AT(g.w[l], r, c) += MAT_AT(local_g.w[l], r, c) * (float)sub_rows;
                    for (int r = 0; r < g.b[l].rows; ++r)
                        for (int c = 0; c < g.b[l].cols; ++c)
                            MAT_AT(g.b[l], r, c) += MAT_AT(local_g.b[l], r, c) * (float)sub_rows;
                }
            }
            // local_g/local_net intentionally not freed (nn.h hasn't provided free)
        } // end parallel

        // average gradients
        for (int l = 0; l < g.count; ++l) {
            for (int r = 0; r < g.w[l].rows; ++r)
                for (int c = 0; c < g.w[l].cols; ++c)
                    MAT_AT(g.w[l], r, c) /= (float)tin.rows;
            for (int r = 0; r < g.b[l].rows; ++r)
                for (int c = 0; c < g.b[l].cols; ++c)
                    MAT_AT(g.b[l], r, c) /= (float)tin.rows;
        }

        // apply learning
        nn_learn(net, g, rate);

        // periodic logging (10 steps)
        if (epochs > 0 && (epoch % ( (epochs/10)>0 ? (epochs/10) : 1) == 0)) {
            printf("[epoch %d/%d] cost = %f\n", epoch, epochs, nn_cost(net, tin, tout));
        }

        // Save a visualization frame if scheduled. We want evenly spaced frames; save at epoch=0 too.
        if ( (epoch % save_every == 0) && frame_index < frame_count ) {
            // build preview sprite (pw x ph) as ARGB pixels
            const int pw = 128, ph = 128;
            uint32_t *sprite_pixels = malloc(sizeof(uint32_t) * pw * ph);
            if (!sprite_pixels) {
                fprintf(stderr, "Preview sprite malloc failed\n");
            } else {
                for (int y = 0; y < ph; ++y) {
                    for (int x = 0; x < pw; ++x) {
                        MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (pw - 1);
                        MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (ph - 1);
                        nn_forward(net);
                        float v = MAT_AT(NN_OUTPUT_MAT(net), 0, 0);
                        if (v < 0.0f) v = 0.0f;
                        if (v > 1.0f) v = 1.0f;
                        sprite_pixels[y * pw + x] = t_to_rgcolor(v);
                    }
                }

                Olivec_Canvas canvas = olivec_canvas(img_pixels_canvas, IMG_X, IMG_Y, IMG_X);
                nn_render(canvas, net, arch, arch_count, sprite_pixels, pw, ph);

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
    } // end epochs

    printf("Final cost = %f\n", nn_cost(net, tin, tout));

    // If fewer frames saved than requested (due to rounding), ensure we output exactly frame_count images:
    // (this is unlikely with the scheduling above, but ensure consistency)
    // Note: frame files are named 0000.. so we already produced sequential frames.

    // Build GIF using ImageMagick convert (10 seconds total => 10 fps if 100 frames)
    // Delay for convert in 1/100ths of a second: for 10 fps use delay 10 (i.e., 100ths of a second)
    printf("Generating GIF vizns/training.gif (requires ImageMagick 'convert')...\n");
    // Use -delay 10 (10 hundredths = 0.10s per frame = 10fps) and loop 0
    system("convert -delay 10 -loop 0 vizns/upscaler-*.png vizns/training.gif");
}

// --- Main program ---
int main(void)
{
    double t_start = omp_get_wtime();

    char *img_path = "./MNIST/train/5/165.png";
    int img_width, img_height, img_comp;
    uint8_t *img_pixels_in = stbi_load(img_path, &img_width, &img_height, &img_comp, 0);
    if (!img_pixels_in) {
        fprintf(stderr, "Could not read image: %s\n", img_path);
        return 1;
    }
    if (img_comp != 1) {
        fprintf(stderr, "ERROR: %s has %d channels; only single-channel grayscale supported\n", img_path, img_comp);
        return 1;
    }
    printf("Loaded image: %s (%d x %d)\n", img_path, img_width, img_height);

    // Build training data matrix: [x_norm, y_norm, brightness]
    mat trd = mat_alloc(img_width * img_height, 3);
    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            int i = pix(x, y);
            MAT_AT(trd, i, 0) = (float)x / (img_width - 1);
            MAT_AT(trd, i, 1) = (float)y / (img_height - 1);
            MAT_AT(trd, i, 2) = img_pixels_in[pix(x, y)] / 255.0f;
        }
    }

    mat tin = {.rows = trd.rows, .cols = 2, .stride = trd.stride, .data = &MAT_AT(trd, 0, 0)};
    mat tout = {.rows = trd.rows, .cols = 1, .stride = trd.stride, .data = &MAT_AT(trd, 0, tin.cols)};

    srand((unsigned)time(NULL));
    int arch[] = {2, 28, 14, 7, 1};
    int arch_count = ARRAY_LEN(arch);
    nn net = nn_alloc(arch, arch_count);
    nn g = nn_alloc(arch, arch_count);
    nn_rand(net, -1, 1);

    printf("Initial cost = %f\n", nn_cost(net, tin, tout));

    // Train: 20000 epochs, save 100 frames
    const int epochs = 20000;
    const int frames = 100;
    train_nn_mt_vis(net, g, epochs, tin, tout, arch, arch_count, frames);

    // Produce final upscaled image (512x512 grayscale)
    const int out_w = 512, out_h = 512;
    uint8_t *out_pixels = malloc(out_w * out_h);
    if (!out_pixels) {
        fprintf(stderr, "Failed to allocate output image\n");
        return 1;
    }

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (out_w - 1);
            MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (out_h - 1);
            nn_forward(net);
            float v = MAT_AT(NN_OUTPUT_MAT(net), 0, 0);
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            out_pixels[y * out_w + x] = (uint8_t)(v * 255.0f);
        }
    }

    if (!stbi_write_png("./upscaled.png", out_w, out_h, 1, out_pixels, out_w)) {
        fprintf(stderr, "Could not save upscaled.png\n");
    } else {
        printf("Saved upscaled.png\n");
    }
    free(out_pixels);

    double elapsed = omp_get_wtime() - t_start;
    int hours = (int)(elapsed / 3600);
    int minutes = (int)((elapsed - hours * 3600) / 60);
    double seconds = elapsed - hours * 3600 - minutes * 60;
    if (hours > 0) printf("Total runtime: %d hr %d min %.3f sec\n", hours, minutes, seconds);
    else if (minutes > 0) printf("Total runtime: %d min %.3f sec\n", minutes, seconds);
    else printf("Total runtime: %.3f sec\n", seconds);

    return 0;
}
