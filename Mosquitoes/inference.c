#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define MODEL_FILE "D:/DevEnv/NN-in-C/Mosquitoes/model.dat"
#define INPUT_FILE "D:/DevEnv/NN-in-C/Mosquitoes/Downscaled/aegypti1b.png"
#define OUTPUT_FILE "D:/DevEnv/NN-in-C/Mosquitoes/inference_upscaled.png"

#define OUT_W 512
#define OUT_H 385

int main(void) {
    int arch[] = {2, 64, 32, 16, 1};
    nn net = nn_alloc(arch, ARRAY_LEN(arch));

    // Load weights and biases
    FILE *fp = fopen(MODEL_FILE, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open model file.\n");
        return 1;
    }
    for (int i = 0; i < net.count; i++) {
        fread(net.w[i].data, sizeof(float), net.w[i].rows * net.w[i].cols, fp);
        fread(net.b[i].data, sizeof(float), net.b[i].rows * net.b[i].cols, fp);
    }
    fclose(fp);

    // Read input image
    int iw, ih, ic;
    uint8_t *img = stbi_load(INPUT_FILE, &iw, &ih, &ic, 1);
    if (!img) {
        fprintf(stderr, "Could not read %s\n", INPUT_FILE);
        return 1;
    }

    uint8_t *out = malloc(OUT_W * OUT_H);

    // Generate upscaled pixels
    for (int y = 0; y < OUT_H; y++) {
        for (int x = 0; x < OUT_W; x++) {
            MAT_AT(NN_INPUT_MAT(net), 0, 0) = (float)x / (OUT_W - 1);
            MAT_AT(NN_INPUT_MAT(net), 0, 1) = (float)y / (OUT_H - 1);
            nn_forward(net);
            float v = MAT_AT(NN_OUTPUT_MAT(net), 0, 0);
            out[y * OUT_W + x] = (uint8_t)(v * 255);
        }
    }

    stbi_write_png(OUTPUT_FILE, OUT_W, OUT_H, 1, out, OUT_W);
    printf("Saved upscaled image to %s\n", OUTPUT_FILE);

    free(out);
    stbi_image_free(img);
    return 0;
}
