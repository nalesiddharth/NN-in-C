#include <stdio.h>
#include <time.h>
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
    for (int i = 0; i < n; i++)
    {
        nn_backprop(net, g, tin, tout);
        //nn_finite_diff(net, g, .01f, tin, tout);
        nn_learn(net, g, rate);
        if (i % (n / 10) == 0)
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