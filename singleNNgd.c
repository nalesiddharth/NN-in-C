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

float gdcost(float w)
{
    float res = 0.0f;
    for(int i = 0; i<train_len; i++)
    {
        float x = training_data[i][0];
        float y = training_data[i][1];
        float t = 2*(x*w - y)*x;
        res += t;
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
    //srand(time(0));
    srand(69);
    //y = x*w
    float w = 200.0f;
    float b = rand_float()*5.0f;
    // float w = 1.0f;
    float eps = 1e-3;
    float rate = 1e-1;
    float c;

    int train_count = 10;
    for(int i = 0; i<train_count; i++)
    {
        c = cost(w,b);
        #if 0
        float dw = (cost(w+eps, b) - c)/eps;
        #else
        float dw = gdcost(w);
        #endif
        w -= rate*dw;
        //float db = (cost(w, b+eps) - c)/eps;
        //b -= rate*db;
        if(i%(train_count/10) == 0)
            printf("\nCost = %f, w = %f", gdcost(w), w);
    }
    //inference(w, b);

    printf("\n\nW: %f", w);
}