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

dataset *training_data = or_data;

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

void gdcost(float w1, float w2, float b, float *dw1, float *dw2, float *db)
{
    *dw1 = 0.0f;
    *dw2 = 0.0f;
    *db = 0.0f;
    for(int i = 0; i<train_len; i++)
    {
        float x1 = training_data[i][0];
        float x2 = training_data[i][1];
        float y = training_data[i][2];
        float a = sigmoidf(x1*w1+x2*w2+b);
        float t = 2*(a-y)*a*(1-a);
        *dw1 += t*x1;
        *dw2 += t*x2;
        *db += t;
    }

    *dw1 /= train_len;
    *dw2 /= train_len;
    *db /= train_len;
}

float inference(float w1, float w2)
{
    printf("\n\nResult: (Predicted, Actual)");
    for(int i = 0; i<train_len; i++)
        printf("\n(%f, %f)", training_data[i][0]*w1, training_data[i][1]);
}

void main()
{
    //srand(time(0));
    srand(10);
    //y = x*w
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();
    float eps = 1e-3;
    float rate;
    
    int train_count = 10000;
    for(int i = 0; i<train_count; i++)
    {
        float dw1, dw2, db;
        #if 0
        float c = cost(w1, w2, b);
        dw1 = (cost(w1+eps, w2, b) - c)/eps;
        dw2 = (cost(w1, w2+eps, b) - c)/eps;
        db = (cost(w1, w2, b+eps) - c)/eps;
        rate = 1e-2;
        #else
        gdcost(w1, w2, b, &dw1, &dw2, &db);
        rate = 1e-1;
        #endif
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