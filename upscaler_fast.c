#include <stdio.h>
#include <time.h>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define pix(x, y) y*img_width + x

float rate = 1;
void train_nn(nn net, nn g, int n, mat tin, mat tout)
{
    int arch[] = {2, 14, 7, 1};
    int arch_count = 4;

    int num_threads = omp_get_max_threads();
    printf("\nUsing %d threads\n", num_threads);

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

            // Thread-local gradient buffer
            nn local_g = nn_alloc(arch, arch_count);
            nn_init(local_g, 0.0f);

            // Thread-local copy of the network for safe forward/backprop
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

            // Important: call nn_backprop once for the chunk.
            // nn_backprop will zero local_g and then compute average gradients over sub_rows.
            nn_backprop(local_net, local_g, sub_tin, sub_tout);

            // Accumulate (as sums) into shared g
            // local_g currently contains *average* gradients for this thread's chunk,
            // so multiply by sub_rows to convert to sums before adding to g.
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
            // note: we intentionally don't free local_g/local_net (nn.h has no free)
        } // end parallel

        // Now g contains sums over all samples; convert to average by dividing by total rows
        for (int l = 0; l < g.count; l++)
        {
            for (int r = 0; r < g.w[l].rows; r++)
                for (int c = 0; c < g.w[l].cols; c++)
                    MAT_AT(g.w[l], r, c) /= (float)tin.rows;

            for (int r = 0; r < g.b[l].rows; r++)
                for (int c = 0; c < g.b[l].cols; c++)
                    MAT_AT(g.b[l], r, c) /= (float)tin.rows;
        }

        // apply a safe learning rate (tune as needed)
        nn_learn(net, g, 1.0f);

        if (epoch % (n / 10) == 0)
            printf("\ncost = %f", nn_cost(net, tin, tout));
    }
}



int main()
{
    char *img_path = "./MNIST/train/5/165.png";
    printf("%s", img_path);
    int img_width, img_height, img_comp;
    uint8_t *img_pixels = (uint8_t *) stbi_load(img_path, &img_width, &img_height, &img_comp, 0);
    if(img_pixels == NULL)
    {
        printf("\nCould not read Image!");
        return 1;
    }
    if(img_comp != 1)
    {
        printf("\nERROR: The image %s had %d bits. Only 8-bit Grayscale images supported!", img_path, img_comp*8);
        return 1;
    }
    printf("\nImage: %s\nSize: %d*%d, %d bits\n", img_path, img_width, img_height, img_comp*8);

    mat trd = mat_alloc(img_width*img_height, 3); //training data

    for(int y = 0; y<img_height; y++)
    {
        for(int x = 0; x<img_width; x++)
        {
            //normalizing coordinate(x,y) and brightness values to 0>x>1, which makes it possible to "upscale" the image
            int i = pix(x, y);
            uint8_t px = img_pixels[pix(x,y)];
            if(px) printf("%3u ", px); else printf("    ");

            MAT_AT(trd, i, 0) = (float) x/(img_width-1);
            MAT_AT(trd, i, 1) = (float) y/(img_height-1);
            MAT_AT(trd, i, 2) = img_pixels[pix(x, y)]/255.f;
            // MAT_AT(tin, i, 0) = (float) x/(img_width-1);
            // MAT_AT(tin, i, 1) = (float) y/(img_height-1);
            // MAT_AT(tout, i, 0) = img_pixels[pix(x, y)]/255.f;
        }
        printf("\n");
    }
    //MAT_PRINT(trd);
    

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


    srand(0);
    int arch[] = {2, 14, 7, 1};
    int arch_count = ARRAY_LEN(arch);
    nn net = nn_alloc(arch, arch_count);
    nn g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(net, -1, 1);

    printf("\ncost = %f", nn_cost(net, tin, tout));

    int train_count = 100000;
    train_nn(net, g, train_count, tin, tout);
    printf("\n");

    int out_width = 2048;
    int out_height = 2048;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_height*out_width);
    for(int y = 0; y<out_height; y++)
    {
        for(int x = 0; x<out_width; x++)
        {
            MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float) x/(out_width - 1);
            MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float) y/(out_height - 1);
            nn_forward(net);
            uint8_t px = MAT_AT(NN_OUTPUT_MAT(net), 0, 0)*255.f;
            out_pixels[y * out_width + x] = px;
        }
    }
    const char *out_path = "./upscaled.png";
    if (!stbi_write_png(out_path, out_width, out_height, 1, out_pixels, out_width*sizeof(*out_pixels)))
    {    
        fprintf(stderr, "Could not save image %s", out_path);
        return 1;
    }

    printf("\nUpscaled image saved as %s", out_path);
    return 0;
}