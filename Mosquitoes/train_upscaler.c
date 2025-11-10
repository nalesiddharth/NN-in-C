#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dirent.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define INPUT_DIR  "./Downscaled"
#define TARGET_DIR "./Upscaled"
#define MODEL_FILE "./model.dat"

#define INPUT_W  56
#define INPUT_H  42
#define TARGET_W 512
#define TARGET_H 385

#define TRAIN_STEPS  50090
#define LEARN_RATE   0.5f
#define TARGET_COST  0.0015f   // stop training when cost < 0.003

// Utility: get pixel from grayscale image
#define PIX(img, x, y, w) ((img)[(y)*(w)+(x)])

// Train the model on all images
int main(void) {
    srand(time(NULL));

    // Network architecture
    int arch[] = {2, 128, 64, 32, 1};
    nn net = nn_alloc(arch, ARRAY_LEN(arch));
    nn g   = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(net, -1, 1);

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(INPUT_DIR)) == NULL) {
        fprintf(stderr, "Could not open %s\n", INPUT_DIR);
        return 1;
    }

    int img_count = 0;
    float total_cost = 0;

    // Go through each image pair
    while ((ent = readdir(dir)) != NULL) {
        if (strstr(ent->d_name, ".png") == NULL) continue;

        // Build full paths
        char input_path[512], target_path[512];
        snprintf(input_path, sizeof(input_path), "%s/%s", INPUT_DIR, ent->d_name);
        snprintf(target_path, sizeof(target_path), "%s/%s", TARGET_DIR, ent->d_name);

        int iw, ih, ic, tw, th, tc;
        uint8_t *in_img = stbi_load(input_path, &iw, &ih, &ic, 1);
        uint8_t *tg_img = stbi_load(target_path, &tw, &th, &tc, 1);
        if (!in_img || !tg_img) {
            fprintf(stderr, "Skipping %s (load error)\n", ent->d_name);
            continue;
        }

        if (iw != INPUT_W || ih != INPUT_H || tw != TARGET_W || th != TARGET_H) {
            fprintf(stderr, "Skipping %s (wrong size)\n", ent->d_name);
            stbi_image_free(in_img);
            stbi_image_free(tg_img);
            continue;
        }

        int samples = INPUT_W * INPUT_H;
        mat tin  = mat_alloc(samples, 2);
        mat tout = mat_alloc(samples, 1);

        for (int y = 0; y < ih; y++) {
            for (int x = 0; x < iw; x++) {
                int i = y * iw + x;
                MAT_AT(tin, i, 0) = (float)x / (iw - 1);
                MAT_AT(tin, i, 1) = (float)y / (ih - 1);
                MAT_AT(tout, i, 0) = PIX(in_img, x, y, iw) / 255.0f;
            }
        }

        printf("\nTraining on %s...\n", ent->d_name);
        float cost = 0.0f;
        cost = nn_cost(net, tin, tout);
        // Train until either TRAIN_STEPS reached or cost < TARGET_COST
        for (int step = 1; step <= TRAIN_STEPS; step++) {
            nn_backprop(net, g, tin, tout);
            nn_learn(net, g, LEARN_RATE);

            // Check cost occasionally
            if (step % 500 == 0 || step == TRAIN_STEPS) {
                cost = nn_cost(net, tin, tout);
                printf("  Step %d | Cost = %.6f\n", step, cost);
            }

            // if (cost < TARGET_COST) {
            //     printf("  Early stopping at step %d (cost=%.6f)\n", step, cost);
            //     break;
            // }
        }
        total_cost += cost;
        img_count++;
        printf("Finished %s | Final cost = %.6f\n", ent->d_name, cost);

        stbi_image_free(in_img);
        stbi_image_free(tg_img);
        free(tin.data);
        free(tout.data);
    }

    closedir(dir);
    printf("\nTrained on %d images, avg cost=%.6f\n", img_count, total_cost/img_count);

    // Save model parameters
    FILE *fp = fopen(MODEL_FILE, "wb");
    if (!fp) {
        fprintf(stderr, "Could not save model!\n");
        return 1;
    }
    for (int i = 0; i < net.count; i++) {
        fwrite(net.w[i].data, sizeof(float), net.w[i].rows * net.w[i].cols, fp);
        fwrite(net.b[i].data, sizeof(float), net.b[i].rows * net.b[i].cols, fp);
    }
    fclose(fp);
    printf("Model saved to %s\n", MODEL_FILE);

    return 0;
}
