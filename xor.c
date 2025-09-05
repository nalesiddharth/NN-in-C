#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define train_len 4

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

dataset xor_data[] = {
    {0, 0, 0}, 
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}  
};

dataset *training_data = xor_data;


struct Xor
{
    float or_w1;
    float or_w2;
    float or_b;
    
    float nand_w1;
    float nand_w2;
    float nand_b;

    float and_w1;
    float and_w2;
    float and_b;
};

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

float forward(struct Xor m, float x1, float x2)
{
    float a = sigmoidf(x1*m.or_w1 + x2*m.or_w2 + m.or_b);
    float b = sigmoidf(x1*m.nand_w1 + x2*m.nand_w2 + m.nand_b);
    float y = sigmoidf(a*m.and_w1 + b*m.and_w2 + m.and_b);
    return y;
}

struct Xor rand_model()
{
    struct Xor x;
    x.or_w1 = (float) rand()/ (float) RAND_MAX;
    x.or_w2 = (float) rand()/ (float) RAND_MAX;
    x.or_b = (float) rand()/ (float) RAND_MAX;
    x.and_w1 = (float) rand()/ (float) RAND_MAX;
    x.and_w2 = (float) rand()/ (float) RAND_MAX;
    x.and_b = (float) rand()/ (float) RAND_MAX;
    x.nand_w1 = (float) rand()/ (float) RAND_MAX;
    x.nand_w2 = (float) rand()/ (float) RAND_MAX;
    x.nand_b = (float) rand()/ (float) RAND_MAX;
    return x;
}

void print_model(struct Xor m)
{
    printf("or_w1 = %f\n", m.or_w1);
    printf("or_w2 = %f\n", m.or_w2);
    printf("or_b = %f\n", m.or_b);

    printf("nand_w1 = %f\n", m.nand_w1);
    printf("nand_w2 = %f\n", m.nand_w2);
    printf("nand_b = %f\n", m.nand_b);

    printf("and_w1 = %f\n", m.and_w1);
    printf("and_w2 = %f\n", m.and_w2);
    printf("and_b = %f\n", m.and_b);
}

float cost(struct Xor m)
{
    float res = 0.0f;
    for(int i = 0; i<train_len; i++)
    {
        float x1 = training_data[i][0];
        float x2 = training_data[i][1];
        float y = forward(m, x1, x2); 
        float d = y-training_data[i][2];
        res += d*d;
    }
    res /= train_len;
    return res;
}

struct Xor finite_diff(struct Xor m, float eps)
{
    struct Xor g;
    float temp;
    float c = cost(m);

    temp = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c)/eps;
    m.or_w1 = temp;

    temp = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c)/eps;
    m.or_w2 = temp;

    temp = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c)/eps;
    m.or_b = temp;

    temp = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c)/eps;
    m.and_w1 = temp;

    temp = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c)/eps;
    m.and_w2 = temp;

    temp = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c)/eps;
    m.and_b = temp;

    temp = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c)/eps;
    m.nand_w1 = temp;

    temp = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c)/eps;
    m.nand_w2 = temp;

    temp = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c)/eps;
    m.nand_b = temp;

    return g;
}

struct Xor learn(struct Xor m, float rate, float eps)
{
    struct Xor g = finite_diff(m, eps);

    m.or_w1 -= g.or_w1*rate;
    m.or_w2 -= g.or_w2*rate;
    m.or_b -= g.or_b*rate;

    m.and_w1 -= g.and_w1*rate;
    m.and_w2 -= g.and_w2*rate;
    m.and_b -= g.and_b*rate;

    m.nand_w1 -= g.nand_w1*rate;
    m.nand_w2 -= g.nand_w2*rate;
    m.nand_b -= g.nand_b*rate;

    return m;
}

void main()
{
    srand(time(0));
    struct Xor m = rand_model();

    float rate = 1e-1; 
    float eps = 1e-1;

    //print_model(m);
    printf("\ncost = %f", cost(m));
    int train_count = 100000;
    for(int i = 0; i<train_count; i++)
    {
        m = learn(m, rate, eps);
        if(i%(train_count/10) == 0)
            printf("\ncost = %f", cost(m));
    }

    printf("\nInference: \n");
    for(int i = 0; i<2; i++)
        for(int j = 0; j<2; j++)
            printf("%d | %d = %f\n", i, j, forward(m, i, j));

    printf("\n\n'OR' Neuron:\n");
    for(int i = 0; i<2; i++)
        for(int j = 0; j<2; j++)
            printf("%d | %d = %f\n", i, j, sigmoidf(m.or_w1*i + m.or_w2*j +m.or_b));

    printf("\n\n'AND' Neuron:\n");
    for(int i = 0; i<2; i++)
        for(int j = 0; j<2; j++)
            printf("%d | %d = %f\n", i, j, sigmoidf(m.and_w1*i + m.and_w2*j +m.and_b));
    
    printf("\n\n'NAND' Neuron:\n");
    for(int i = 0; i<2; i++)
        for(int j = 0; j<2; j++)
            printf("%d | %d = %f\n", i, j, sigmoidf(m.nand_w1*i + m.nand_w2*j +m.nand_b));


}