#include <stdio.h>
#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define IMG_X 800
#define IMG_Y 600

uint32_t img_pixels[IMG_X*IMG_Y];

int main(void)
{
    srand(time(0));
    int arch[] = {2, 4, 3, 1};
    int arch_count = ARRAY_LEN(arch);
    nn net = nn_alloc(arch, arch_count);
    nn_rand(net, -1, 1);
    NN_PRINT(net);

    //ABGR color format
    uint32_t bg_col = 0xFF472D0F;
    uint32_t conn_col = 0xFF00FF00;
    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_X, IMG_Y, IMG_X);
    olivec_fill(img, bg_col);
    
    //calculating neuron draw positions
    int n_rad = 25;
    int layer_bvpad = 50;
    int layer_bhpad = 50;
    int net_width = img.width - layer_bhpad*2;
    int net_height = img.height - 2*layer_bvpad;
    int layer_hpad =  net_width/arch_count;
    int net_x = img.width/2 - net_width/2;
    int net_y = img.height/2 - net_height/2;
    for(int l = 0; l<arch_count; l++) //iterating through the layers
    {
        int layer_vpad1 = net_height/arch[l];
        for(int i = 0; i<arch[l]; i++) //iterating through each neuron
        {
            int cx1 = net_x + l*layer_hpad + layer_hpad/2;
            int cy1 = net_y + i*layer_vpad1 + layer_vpad1/2;
            if(l+1<arch_count)
            {
                int layer_vpad2 = net_height/arch[l+1];
                for(int j = 0; j< arch[l+1]; j++)
                {
                    int cx2 = net_x + (l+1)*layer_hpad + layer_hpad/2;
                    int cy2 = net_y + j*layer_vpad2 + layer_vpad2/2;
                    olivec_line(img, cx1, cy1, cx2, cy2, conn_col);
                }
            }
            if(l>0)
            {
                uint32_t s = floorf(255.f*sigmoidf(MAT_AT(net.b[l-1], 0, 1)));
                uint32_t n_col = 0xFF0000FF;
                olivec_blend_color(&n_col, (s<<(8*3))|0x0000FF00);
                olivec_circle(img, cx1, cy1, n_rad, n_col);
            }
            else
                olivec_circle(img, cx1, cy1, n_rad, 0xFF808080);

        }
    } 

    uint32_t frame_col = 0xFF642D11;
    uint32_t frame_thick = 10;
    olivec_frame(img, 0, 0, IMG_X-1, IMG_Y-1, frame_thick, frame_col);
    
    const char *img_path = "./vizns/nn.png";
    if(!stbi_write_png(img_path, img.width, img.height, 4, img.pixels, img.stride*sizeof(uint32_t)))
    {
        printf("ERROR while saving file: %s", img_path);
        return 1;
    }
    else
    {
        printf("Image save successfull at %s", img_path);
    }
    return 0;
}