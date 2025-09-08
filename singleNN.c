#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

float training_data[][2] = { 
    {0, 0}, 
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

#define train_len (sizeof(training_data) / sizeof(training_data[0]))

float rand_float(void)
{
    float f = (float) rand()/ (float) RAND_MAX;
    return f;
}

float cost(float w, float b)
{
    float res = 0.0f;
    for(int i = 0; i<train_len; i++)
    {
        float x = training_data[i][0];
        float y = x*w + b; 
        float d = y-training_data[i][1];
        res += d*d;
    }
    res /= train_len;
    return res;
}

float inference(float w, float b)
{
    printf("\n\nResult: (Predicted, Actual)");
    for(int i = 0; i<train_len; i++)
        printf("\n(%f, %f)", training_data[i][0]*w+b, training_data[i][1]);
}

void main()
{
    srand(time(0));
    //y = x*w
    float w = rand_float()*10.0f;
    float b = rand_float()*5.0f;
    // float w = 1.0f;
    float eps = 1e-3;
    float rate = 1e-3;
    float c;

    int train_count = 50000;
    for(int i = 0; i<train_count; i++)
    {
        c = cost(w,b);
        float dw = (cost(w+eps, b) - c)/eps;
        float db = (cost(w, b+eps) - c)/eps;
        w -= rate*dw;
        b -= rate*db;
        if(i%(train_count/10) == 0)
            printf("\nCost = %f, w = %f, b = %f", cost(w,b), w, b);
    }
    inference(w, b);

    printf("\n\nW: %f", w);
}