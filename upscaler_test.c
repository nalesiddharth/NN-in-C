#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn_test.h"

#define pix(x, y) ((y) * img_width + (x))

/* Helper: shuffle an int array in-place */
static void shuffle_indices(int *arr, int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
}

/* New training: mini-batch SGD */
void train_nn(nn net, nn g, mat tin, mat tout,
              int epochs, int batch_size, float rate, int display_every)
{
    int sample_count = tin.rows;
    int *indices = malloc(sizeof(*indices) * sample_count);
    for (int i = 0; i < sample_count; i++) indices[i] = i;

    nn gtemp = nn_alloc((int[]){1}, 1); // dummy to satisfy allocation macro - not used; we'll allocate proper grads
    // Proper gradient net
    int arch_count = net.count + 1;
    int *arch = malloc(sizeof(*arch) * arch_count);
    // Build architecture array from net.a sizes (for allocation ease)
    for (int i = 0; i < arch_count; i++)
        arch[i] = net.a[i].cols;
    nn grads = nn_alloc(arch, arch_count);

    int steps_per_epoch = (sample_count + batch_size - 1) / batch_size;
    for (int e = 0; e < epochs; e++)
    {
        shuffle_indices(indices, sample_count);
        for (int s = 0; s < steps_per_epoch; s++)
        {
            int start = s * batch_size;
            int end = start + batch_size;
            if (end > sample_count) end = sample_count;
            int cur_batch = end - start;
            // prepare batch index list
            int *batch_idx = &indices[start];
            // compute gradients over this mini-batch
            nn_backprop_batch(net, grads, tin, tout, batch_idx, cur_batch);
            // update
            nn_learn(net, grads, rate);
        }

        if ((e % display_every) == 0)
        {
            float c = nn_cost(net, tin, tout);
            printf("Epoch %4d / %d   cost = %f\n", e, epochs, c);
        }
    }

    free(indices);
    free(arch);
}

/* main: uses a smaller/faster architecture and mini-batch SGD by default */
int main()
{
    char *img_path = "./MNIST/train/8/260.png";
    printf("%s\n", img_path);
    int img_width, img_height, img_comp;
    uint8_t *img_pixels = (uint8_t *) stbi_load(img_path, &img_width, &img_height, &img_comp, 0);
    if(img_pixels == NULL)
    {
        printf("\nCould not read Image!");
        return 1;
    }
    if(img_comp != 1)
    {
        printf("\nERROR: The image %s had %d channels. Only 8-bit Grayscale supported!", img_path, img_comp);
        return 1;
    }
    printf("\nImage: %s\nSize: %d*%d, %d channels\n", img_path, img_width, img_height, img_comp);

    mat trd = mat_alloc(img_width*img_height, 3); // training data rows = pixels, cols = [x_norm, y_norm, brightness]

    for(int y = 0; y<img_height; y++)
    {
        for(int x = 0; x<img_width; x++)
        {
            int i = pix(x, y);
            uint8_t px = img_pixels[i];
            MAT_AT(trd, i, 0) = (float) x / (img_width - 1);
            MAT_AT(trd, i, 1) = (float) y / (img_height - 1);
            MAT_AT(trd, i, 2) = px / 255.f;
        }
    }

    mat tin = {
        .rows = trd.rows,
        .cols = 2,
        .stride = trd.stride,
        .data = &MAT_AT(trd, 0, 0),
    };

    mat tout = {
        .rows = trd.rows,
        .cols = 1,
        .stride = trd.stride,
        .data = &MAT_AT(trd, 0, tin.cols),
    };

    srand(68);

    /* Recommended architecture: 2 -> 32 -> 1 (fast and effective for mapping coords -> brightness)
       You can change this array to experiment (eg {2,64,64,1} but that will be slower) */
    int arch[] = {2, 7, 4, 1};
    int arch_count = ARRAY_LEN(arch);
    nn net = nn_alloc(arch, arch_count);
    nn grads = nn_alloc(arch, arch_count);
    nn_rand(net, -1, 1);

    printf("\nInitial cost = %f\n", nn_cost(net, tin, tout));

    /* Training hyperparameters - tune these */
    int epochs = 10000;         // number of epochs
    int batch_size = 32;      // mini-batch size
    float rate = 0.1f;       // learning rate
    int display_every = 1000;   // print cost every N epochs

    train_nn(net, grads, tin, tout, epochs, batch_size, rate, display_every);

    printf("\nTraining complete. Final cost = %f\n", nn_cost(net, tin, tout));

    /* Create upscaled image */
    int out_width = 512;
    int out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels) * out_height * out_width);
    for(int y = 0; y < out_height; y++)
    {
        for(int x = 0; x < out_width; x++)
        {
            MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float) x / (out_width - 1);
            MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float) y / (out_height - 1);
            nn_forward(net);
            float v = MAT_AT(NN_OUTPUT_MAT(net), 0, 0);
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            uint8_t px = (uint8_t)(v * 255.0f + 0.5f);
            out_pixels[y * out_width + x] = px;
        }
    }
    const char *out_path = "./upscaled_test.png";
    if (!stbi_write_png(out_path, out_width, out_height, 1, out_pixels, out_width * sizeof(*out_pixels)))
    {
        fprintf(stderr, "Could not save image %s", out_path);
        return 1;
    }

    printf("\nUpscaled image saved as %s\n", out_path);

    return 0;
}
