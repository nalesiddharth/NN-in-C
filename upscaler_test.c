#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

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

uint32_t img_pixels[IMG_X * IMG_Y];

// Sign-based color: positive weights green, negative red, intensity by magnitude.
static inline uint32_t weight_color(float w)
{
    float sig = 1.0f / (1.0f + expf(-fabsf(w)));
    uint8_t intensity = (uint8_t)(sig * 255.0f);
    if (w >= 0)
        return 0xFF000000 | (intensity << 8); // green
    else
        return 0xFF000000 | (intensity << 16); // red
}

int nn_render(Olivec_Canvas img, nn net, int *arch, int arch_count, uint8_t *preview, int pw, int ph)
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
                    float w = MAT_AT(net.w[l], j, i);
                    uint32_t col = weight_color(w);
                    olivec_line(img, cx1, cy1, cx2, cy2, col);
                }
            }
            if (l > 0)
            {
                float b = MAT_AT(net.b[l - 1], 0, i);
                uint8_t val = (uint8_t)(255.0f / (1.0f + expf(-b)));
                uint32_t col = 0xFF000000 | (val << 8) | (val << 16);
                olivec_circle(img, cx1, cy1, n_rad, col);
            }
            else
                olivec_circle(img, cx1, cy1, n_rad, 0xFF808080);
        }
    }

    // Overlay current image preview bottom-right corner
    if (preview)
    {
        int px = img.width - pw - 20;
        int py = img.height - ph - 20;
        for (int y = 0; y < ph; y++)
        {
            for (int x = 0; x < pw; x++)
            {
                uint8_t v = preview[y * pw + x];
                uint32_t c = 0xFF000000 | (v << 16) | (v << 8) | v;
                olivec_blend_color(&OLIVEC_PIXEL(img, px + x, py + y), c);
            }
        }
        olivec_frame(img, px, py, px + pw, py + ph, 2, 0xFFFFFFFF);
    }

    olivec_frame(img, 0, 0, IMG_X - 1, IMG_Y - 1, 10, 0xFFFFFFFF);
    return 0;
}

#define pix(x, y) ((y) * img_width + (x))

float rate = 1.0f;
void train_nn(nn net, nn g, int n, mat tin, mat tout, int arch[], int arch_count, uint8_t *input_img, int iw, int ih)
{
    int save_every = n / 10;
    for (int i = 0; i < n; i++)
    {
        nn_backprop(net, g, tin, tout);
        nn_learn(net, g, rate);
        if (i % (n / 10) == 0)
            printf("\ncost = %f", nn_cost(net, tin, tout));
        if (i % save_every == 0)
        {
            // Generate quick upscaled preview
            int pw = 128, ph = 128;
            uint8_t *preview = malloc(pw * ph);
            for (int y = 0; y < ph; y++)
            {
                for (int x = 0; x < pw; x++)
                {
                    MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (pw - 1);
                    MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (ph - 1);
                    nn_forward(net);
                    preview[y * pw + x] = (uint8_t)(fminf(1.0f, fmaxf(0.0f, MAT_AT(NN_OUTPUT_MAT(net), 0, 0))) * 255.0f);
                }
            }

            Olivec_Canvas img = olivec_canvas(img_pixels, IMG_X, IMG_Y, IMG_X);
            nn_render(img, net, arch, arch_count, preview, pw, ph);
            char img_op_path[256];
            static int frameIndex = 0;
            snprintf(img_op_path, sizeof(img_op_path), "./vizns/upscaler-%03d.png", frameIndex);
            if (!stbi_write_png(img_op_path, img.width, img.height, 4, img.pixels, img.stride * sizeof(uint32_t)))
                printf("\nERROR while saving file: %s", img_op_path);
            else
                printf("\nSaved visualization frame: %s", img_op_path);
            free(preview);
            frameIndex++;
        }
    }
}

int main()
{
    char *img_path = "./MNIST/train/5/165.png";
    int img_width, img_height, img_comp;
    uint8_t *img_pixels_in = (uint8_t *)stbi_load(img_path, &img_width, &img_height, &img_comp, 0);
    if (!img_pixels_in)
    {
        printf("\nCould not read Image!");
        return 1;
    }
    if (img_comp != 1)
    {
        printf("\nERROR: The image %s had %d channels. Only 8-bit Grayscale images supported!", img_path, img_comp);
        return 1;
    }
    printf("\nImage: %s\nSize: %d*%d\n", img_path, img_width, img_height);

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
    int arch[] = {2, 28, 14, 7, 1};
    int arch_count = ARRAY_LEN(arch);
    nn net = nn_alloc(arch, arch_count);
    nn g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(net, -1, 1);

    printf("\nInitial cost = %f", nn_cost(net, tin, tout));

    int train_count = 20000;
    train_nn(net, g, train_count, tin, tout, arch, arch_count, img_pixels_in, img_width, img_height);

    int out_width = 512, out_height = 512;
    uint8_t *out_pixels = malloc(out_width * out_height);
    for (int y = 0; y < out_height; y++)
    {
        for (int x = 0; x < out_width; x++)
        {
            MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (out_width - 1);
            MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (out_height - 1);
            nn_forward(net);
            out_pixels[y * out_width + x] = (uint8_t)(fminf(1.0f, fmaxf(0.0f, MAT_AT(NN_OUTPUT_MAT(net), 0, 0))) * 255.0f);
        }
    }

    const char *out_path = "./upscaled.png";
    if (!stbi_write_png(out_path, out_width, out_height, 1, out_pixels, out_width))
    {
        fprintf(stderr, "\nCould not save image %s", out_path);
        return 1;
    }

    printf("\nUpscaled image saved as %s", out_path);
    return 0;
}
