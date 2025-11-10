// upscaler_combined.c
// Combined multithreaded training (from upscaler_fast.c) with visualization (from upscaler_test.c).
// Visualization uses red (low) -> green (high). Prints total runtime.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define OLIVEC_IMPLEMENTATION
#include "olive.c"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Canvas size used for visualization (same as upscaler_test)
#define IMG_X 1024
#define IMG_Y 768

uint32_t img_pixels_canvas[IMG_X * IMG_Y];

// Utility: line macro used previously
#define pix(x, y) ((y) * img_width + (x))

// Map weight/value (arbitrary float) to a red->green color.
// t in [0,1]: 0 => red (255,0,0) ; 1 => green (0,255,0)
// We'll derive t from w using tanh to spread values (stable for big/small weights).
static inline uint32_t red_green_color_from_float(float w)
{
    // map w to t in [0,1]
    // Use tanh(w) to squash to [-1,1], then scale to [0,1]
    float t = (tanhf(w) + 1.0f) * 0.5f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    uint8_t r = (uint8_t)((1.0f - t) * 255.0f);
    uint8_t g = (uint8_t)(t * 255.0f);
    // ARGB format expected by Olivec: 0xAARRGGBB (we set B=0)
    uint32_t col = 0xFF000000u | ((uint32_t)r << 16) | ((uint32_t)g << 8);
    return col;
}

// Render the network to the Olivec canvas similar to upscaler_test.nn_render
// But use red<->green color mapping for weights and preview pixels.
// preview is grayscale bytes[ pw * ph ]
int nn_render_rg(Olivec_Canvas img, nn net, int *arch, int arch_count, uint8_t *preview, int pw, int ph)
{
    uint32_t bg_col = 0xFF1A1A1A;
    olivec_fill(img, bg_col);

    int n_rad = 18;
    int layer_bvpad = 50;
    int layer_bhpad = 50;
    int net_width = img.width - layer_bhpad * 2;
    int net_height = img.height - 2 * layer_bvpad;
    int layer_hpad = net_width / arch_count;
    int net_x = img.width / 2 - net_width / 2;
    int net_y = img.height / 2 - net_height / 2;

    for (int l = 0; l < arch_count; l++)
    {
        int layer_vpad1 = net_height / arch[l];
        for (int i = 0; i < arch[l]; i++)
        {
            int cx1 = net_x + l * layer_hpad + layer_hpad / 2;
            int cy1 = net_y + i * layer_vpad1 + layer_vpad1 / 2;
            if (l + 1 < arch_count)
            {
                int layer_vpad2 = net_height / arch[l + 1];
                for (int j = 0; j < arch[l + 1]; j++)
                {
                    int cx2 = net_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    int cy2 = net_y + j * layer_vpad2 + layer_vpad2 / 2;
                    // sample weight w from net.w[l] at [j, i] (note matrices are stored as rows = input neurons, cols = output neurons)
                    float w = MAT_AT(net.w[l], j, i);
                    uint32_t col = red_green_color_from_float(w);
                    olivec_line(img, cx1, cy1, cx2, cy2, col);
                }
            }
            if (l > 0)
            {
                // draw neuron circle; bias affects intensity (use sigmoid of bias -> grayscale)
                float b = MAT_AT(net.b[l - 1], 0, i);
                uint8_t val = (uint8_t)(255.0f / (1.0f + expf(-b))); // 0..255
                uint32_t col = 0xFF000000 | (val << 16) | (val << 8) | val;
                olivec_circle(img, cx1, cy1, n_rad, col);
            }
            else
            {
                olivec_circle(img, cx1, cy1, n_rad, 0xFF808080);
            }
        }
    }

    // Overlay current image preview bottom-right corner (but colorized red->green)
    if (preview)
    {
        int px = img.width - pw - 20;
        int py = img.height - ph - 20;
        for (int y = 0; y < ph; y++)
        {
            for (int x = 0; x < pw; x++)
            {
                uint8_t v = preview[y * pw + x]; // 0..255
                // map v to color from red (low) to green (high)
                float t = (float)v / 255.0f;
                if (t < 0.0f) t = 0.0f;
                if (t > 1.0f) t = 1.0f;
                uint8_t r = (uint8_t)((1.0f - t) * 255.0f);
                uint8_t g = (uint8_t)(t * 255.0f);
                uint32_t c = 0xFF000000u | ((uint32_t)r << 16) | ((uint32_t)g << 8);
                olivec_blend_color(&OLIVEC_PIXEL(img, px + x, py + y), c);
            }
        }
        olivec_frame(img, px, py, px + pw, py + ph, 2, 0xFFFFFFFF);
    }

    olivec_frame(img, 0, 0, IMG_X - 1, IMG_Y - 1, 10, 0xFFFFFFFF);
    return 0;
}

// Multithreaded training: adapted from upscaler_fast.c but calling visualization periodically.
// net : shared network updated at epoch end using averaged gradients in g
// n : epochs
// tin/tout : training mats
// arch/arch_count for visualization layout
// preview generation uses pw/ph and writes frames to ./vizns/
float rate = 1.0f;
void train_nn_mt_vis(nn net, nn g, int n, mat tin, mat tout, int arch[], int arch_count, int save_every_epochs)
{
    int num_threads = omp_get_max_threads();
    printf("\nUsing %d threads\n", num_threads);

    // Make sure viz output directory exists (POSIX). If system() fails it's ok — best-effort.
    system("mkdir -p vizns");

    for (int epoch = 0; epoch < n; epoch++)
    {
        // Zero global gradient accumulator before each epoch
        nn_init(g, 0.0f);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int rows = tin.rows;
            int start = (tid * rows) / num_threads;
            int end = ((tid + 1) * rows) / num_threads;
            int sub_rows = end - start;

            // Thread-local gradient buffer and local net
            nn local_g = nn_alloc(arch, arch_count);
            nn_init(local_g, 0.0f);

            nn local_net = nn_alloc(arch, arch_count);
            // copy weights & biases from shared net
            for (int l = 0; l < net.count; l++) {
                mat_cpy(local_net.w[l], net.w[l]);
                mat_cpy(local_net.b[l], net.b[l]);
            }

            // Create mat views for this thread's chunk
            mat sub_tin = {
                .rows = sub_rows,
                .cols = tin.cols,
                .stride = tin.stride,
                .data = &MAT_AT(tin, start, 0),
            };
            mat sub_tout = {
                .rows = sub_rows,
                .cols = tout.cols,
                .stride = tout.stride,
                .data = &MAT_AT(tout, start, 0),
            };

            // Compute average gradients for this chunk using local_net and store into local_g
            nn_backprop(local_net, local_g, sub_tin, sub_tout);

            // Accumulate as sums into shared g (convert averages in local_g to sums by multiplying sub_rows)
            #pragma omp critical
            {
                for (int l = 0; l < g.count; l++)
                {
                    for (int r = 0; r < g.w[l].rows; r++)
                        for (int c = 0; c < g.w[l].cols; c++)
                            MAT_AT(g.w[l], r, c) += MAT_AT(local_g.w[l], r, c) * (float)sub_rows;

                    for (int r = 0; r < g.b[l].rows; r++)
                        for (int c = 0; c < g.b[l].cols; c++)
                            MAT_AT(g.b[l], r, c) += MAT_AT(local_g.b[l], r, c) * (float)sub_rows;
                }
            }
            // intentional: no frees (consistent with nn.h allocation style)
        } // end parallel

        // Now convert g sums to averages
        for (int l = 0; l < g.count; l++)
        {
            for (int r = 0; r < g.w[l].rows; r++)
                for (int c = 0; c < g.w[l].cols; c++)
                    MAT_AT(g.w[l], r, c) /= (float)tin.rows;

            for (int r = 0; r < g.b[l].rows; r++)
                for (int c = 0; c < g.b[l].cols; c++)
                    MAT_AT(g.b[l], r, c) /= (float)tin.rows;
        }

        // apply learning rate
        nn_learn(net, g, rate);

        // periodic status
        if (n > 0 && epoch % (n / 10 == 0 ? 1 : (n / 10)) == 0) {
            printf("\n[epoch %d/%d] cost = %f", epoch, n, nn_cost(net, tin, tout));
        }

        // Save visualization periodically
        if (save_every_epochs > 0 && (epoch % save_every_epochs == 0))
        {
            int pw = 128, ph = 128;
            uint8_t *preview = malloc(pw * ph);
            if (!preview) {
                fprintf(stderr, "\nPreview alloc failed\n");
            } else {
                for (int y = 0; y < ph; y++)
                {
                    for (int x = 0; x < pw; x++)
                    {
                        MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (pw - 1);
                        MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (ph - 1);
                        nn_forward(net);
                        float outv = MAT_AT(NN_OUTPUT_MAT(net), 0, 0);
                        // clamp to [0,1]
                        if (outv < 0.0f) outv = 0.0f;
                        if (outv > 1.0f) outv = 1.0f;
                        preview[y * pw + x] = (uint8_t)(outv * 255.0f);
                    }
                }

                Olivec_Canvas img = olivec_canvas(img_pixels_canvas, IMG_X, IMG_Y, IMG_X);
                nn_render_rg(img, net, arch, arch_count, preview, pw, ph);

                char img_op_path[256];
                static int frameIndex = 0;
                snprintf(img_op_path, sizeof(img_op_path), "./vizns/upscaler-%04d.png", frameIndex);
                if (!stbi_write_png(img_op_path, img.width, img.height, 4, img.pixels, img.stride * sizeof(uint32_t)))
                    printf("\nERROR while saving file: %s", img_op_path);
                else
                    printf("\nSaved visualization frame: %s", img_op_path);
                free(preview);
                frameIndex++;
            }
        }
    } // end epochs

    // final cost
    printf("\nFinal cost = %f\n", nn_cost(net, tin, tout));
}

int main()
{
    // Start timer
    double t_start = omp_get_wtime();

    // input image path (same default as the originals)
    char *img_path = "./MNIST/train/5/165.png";
    int img_width, img_height, img_comp;
    uint8_t *img_pixels_in = (uint8_t *)stbi_load(img_path, &img_width, &img_height, &img_comp, 0);
    if (!img_pixels_in)
    {
        printf("\nCould not read Image: %s\n", img_path);
        return 1;
    }
    if (img_comp != 1)
    {
        printf("\nERROR: The image %s had %d channels. Only 8-bit Grayscale images supported!\n", img_path, img_comp);
        return 1;
    }
    printf("\nImage: %s\nSize: %d x %d\n", img_path, img_width, img_height);

    // build training matrix: columns (x_norm, y_norm, brightness)
    mat trd = mat_alloc(img_width * img_height, 3);
    for (int y = 0; y < img_height; y++)
    {
        for (int x = 0; x < img_width; x++)
        {
            int i = pix(x, y);
            MAT_AT(trd, i, 0) = (float)x / (img_width - 1);
            MAT_AT(trd, i, 1) = (float)y / (img_height - 1);
            MAT_AT(trd, i, 2) = img_pixels_in[pix(x, y)] / 255.0f;
        }
    }

    mat tin = {.rows = trd.rows, .cols = 2, .stride = trd.stride, .data = &MAT_AT(trd, 0, 0)};
    mat tout = {.rows = trd.rows, .cols = 1, .stride = trd.stride, .data = &MAT_AT(trd, 0, tin.cols)};

    srand((unsigned)time(NULL));
    // architecture: you can adjust as desired; we use the network from upscaler_test for richer visualization
    int arch[] = {2, 28, 14, 7, 1};
    int arch_count = ARRAY_LEN(arch);
    nn net = nn_alloc(arch, arch_count);
    nn g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(net, -1, 1);

    printf("\nInitial cost = %f\n", nn_cost(net, tin, tout));

    int train_count = 20000; // epochs
    int save_every_epochs = (train_count / 10 > 0) ? (train_count / 10) : 1; // produce ~10 visualization frames
    train_nn_mt_vis(net, g, train_count, tin, tout, arch, arch_count, save_every_epochs);

    // After training, produce the final upscaled image
    int out_width = 512, out_height = 512;
    uint8_t *out_pixels = malloc(out_width * out_height);
    if (!out_pixels) {
        fprintf(stderr, "Failed to allocate out_pixels\n");
        return 1;
    }
    for (int y = 0; y < out_height; y++)
    {
        for (int x = 0; x < out_width; x++)
        {
            MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (out_width - 1);
            MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (out_height - 1);
            nn_forward(net);
            float val = MAT_AT(NN_OUTPUT_MAT(net), 0, 0);
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            out_pixels[y * out_width + x] = (uint8_t)(val * 255.0f);
        }
    }

    const char *out_path = "./upscaled.png";
    if (!stbi_write_png(out_path, out_width, out_height, 1, out_pixels, out_width))
    {
        fprintf(stderr, "\nCould not save image %s", out_path);
        return 1;
    }

    printf("\nUpscaled image saved as %s\n", out_path);

    // Stop timer and print total time
    double t_end = omp_get_wtime();
    double elapsed_sec = t_end - t_start;
    // Print in human-friendly format
    int hours = (int)(elapsed_sec / 3600.0);
    int minutes = (int)((elapsed_sec - hours * 3600) / 60);
    double seconds = elapsed_sec - hours * 3600 - minutes * 60;
    if (hours > 0)
        printf("\nTotal runtime: %d hr %d min %.3f sec\n", hours, minutes, seconds);
    else if (minutes > 0)
        printf("\nTotal runtime: %d min %.3f sec\n", minutes, seconds);
    else
        printf("\nTotal runtime: %.3f sec\n", seconds);

    return 0;
}
