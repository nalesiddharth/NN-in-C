#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef float dataset[3];

dataset or_data[] = {
    {0, 0, 0}, 
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1}
};

dataset and_data[] = {
    {0, 0, 0}, 
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1}
};

dataset nand_data[] = {
    {0, 0, 1}, 
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
};

dataset *training_data = nand_data;

#define train_len 4

float rand_float(void)
{
    float f = (float) rand()/ (float) RAND_MAX;
    return f;
}

float sigmoidf(float x)
{
    float s = 1.0f/(1.0f + expf(-x));
    return s;
}

float cost(float w1, float w2, float b)
{
    float res = 0.0f;
    for(int i = 0; i<train_len; i++)
    {
        float x1 = training_data[i][0];
        float x2 = training_data[i][1];
        float y = sigmoidf(x1*w1 + x2*w2+b); 
        float d = y-training_data[i][2];
        res += d*d;
    }
    res /= train_len;
    return res;
}

float inference(float w1, float w2)
{
    printf("\n\nResult: (Predicted, Actual)");
    for(int i = 0; i<train_len; i++)
        printf("\n(%f, %f)", training_data[i][0]*w1, training_data[i][1]);
}

void main()
{
    srand(time(0));
    //y = x*w
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();
    float eps = 1e-3;
    float rate = 1e-3;
    
    int train_count = 1000000;
    for(int i = 0; i<train_count; i++)
    {
        float c = cost(w1, w2, b);
        float dw1 = (cost(w1+eps, w2, b) - c)/eps;
        float dw2 = (cost(w1, w2+eps, b) - c)/eps;
        float db = (cost(w1, w2, b+eps) - c)/eps;
        
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*db;

        if(i%(train_count/10) == 0)
            printf("\nCost = %f, w1= %f, w2 = %f, b = %f", cost(w1, w2, b), w1, w2, b);
    }

    printf("\nInference: \n");
    for(int i = 0; i<2; i++)
        for(int j = 0; j<2; j++)
            printf("%d | %d = %f\n", i, j, sigmoidf(i*w1 + j*w2 + b));
}


// void main()
// {
//     float x[] = {1.0f, 2.0f, 3.0f};
//     float *px[] = &x;
//     printf("%f", *px);
// }