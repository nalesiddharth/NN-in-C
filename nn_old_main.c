#define NN_IMPLEMENTATION
#include <stdio.h>
#include <time.h>
#include "nn_old.h"

float or_data[] = {
    0.0f, 0.0f, 0.0f, 
    0.0f, 1.0f, 1.0f, 
    1.0f, 0.0f, 1.0f, 
    1.0f, 1.0f, 1.0f};

float and_data[] = {
    0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f};

float nand_data[] = {
    0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 1.0f,
    1.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f};

float xor_data[] = {
    0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 1.0f,
    1.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f};

typedef struct{
    mat a0;
    mat w1, b1, a1;
    mat w2, b2, a2;
} xor;

const int stride = 3;
const int n = sizeof(xor_data) / sizeof(xor_data[0]);
const int row_count = n / stride;

mat tin = {
    .rows = row_count,
    .cols = 2,
    .stride = stride,
    .data = and_data};

mat tout = {
    .rows = row_count,
    .cols = 1,
    .stride = stride,
    .data = and_data + 2};

xor xor_alloc()
{
    xor m;
    m.a0 = mat_alloc(1, 2);
    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);
    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);
    return m;
}

float forward_xor(xor m)
{
    mat_mult(m.a1, m.a0, m.w1);
    mat_add(m.a1, m.b1);
    mat_sigmoidf(m.a1);

    mat_mult(m.a2, m.a1, m.w2);
    mat_add(m.a2, m.b2);
    mat_sigmoidf(m.a2);

    float y = MAT_AT(m.a2, 0, 0);

    return y;
}

float cost(xor m)
{
    NN_ASSERT(tin.rows == tout.rows);
    NN_ASSERT(tout.cols == m.a2.cols);
    float c = 0.0f;
    for (int i = 0; i < tin.rows; i++)
    {
        mat x = mat_getRow(tin, i);
        mat y = mat_getRow(tout, i);
        mat_cpy(m.a0, x);
        forward_xor(m);

        for (int j = 0; j < tout.cols; j++)
        {
            float d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
            c += d * d;
        }
    }
    c /= tin.rows;
    return c;
}

void finite_diff(xor m, xor g, float eps)
{
    float temp;
    float c = cost(m);
    for (int i = 0; i < m.w1.rows; i++)
    {
        for (int j = 0; j < m.w1.cols; j++)
        {
            temp = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost(m) - c) / eps;
            MAT_AT(m.w1, i, j) = temp;
        }
    }

    for (int i = 0; i < m.b1.rows; i++)
    {
        for (int j = 0; j < m.b1.cols; j++)
        {
            temp = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (cost(m) - c) / eps;
            MAT_AT(m.b1, i, j) = temp;
        }
    }

    for (int i = 0; i < m.w2.rows; i++)
    {
        for (int j = 0; j < m.w2.cols; j++)
        {
            temp = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (cost(m) - c) / eps;
            MAT_AT(m.w2, i, j) = temp;
        }
    }

    for (int i = 0; i < m.b2.rows; i++)
    {
        for (int j = 0; j < m.b2.cols; j++)
        {
            temp = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (cost(m) - c) / eps;
            MAT_AT(m.b2, i, j) = temp;
        }
    }
}

void xor_learn(xor m, xor g, float rate)
{
    for (int i = 0; i < m.w1.rows; i++)
        for (int j = 0; j<m.w1.cols; j++)
            MAT_AT(m.w1, i, j) -= rate*MAT_AT(g.w1, i, j);

    for (int i = 0; i < m.b1.rows; i++)
        for (int j = 0; j < m.b1.cols; j++)
            MAT_AT(m.b1, i, j) -= rate*MAT_AT(g.b1, i, j);

    for (int i = 0; i < m.w2.rows; i++)
        for (int j = 0; j < m.w2.cols; j++)
            MAT_AT(m.w2, i, j) -= rate*MAT_AT(g.w2, i, j);

    for (int i = 0; i < m.b2.rows; i++)
        for (int j = 0; j < m.b2.cols; j++)
            MAT_AT(m.b2, i, j) -= rate*MAT_AT(g.b2, i, j);
}

int main()
{

    srand(time(0));

    MAT_PRINT(tin);
    MAT_PRINT(tout);
    
    float eps = 1e-1;
    float rate = 1e-1;

    xor m = xor_alloc();
    xor g = xor_alloc();

    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1);
    mat_rand(m.b2, 0, 1);

    // MAT_PRINT(m.w1);
    // MAT_PRINT(m.b1);
    // MAT_PRINT(m.w2);
    // MAT_PRINT(m.b2);

    
    printf("\ncost = %f", cost(m));
    int train_count = 30000;
    for (int i = 0; i < train_count; i++)
    {
        finite_diff(m, g, eps);
        xor_learn(m, g, rate);
        if (i % (train_count / 10) == 0)
            printf("\ncost = %f", cost(m));
    }

    printf("\n\nInference: \n");
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            printf("%d ^ %d = %f", i, j, forward_xor(m));
            printf("\n");
        }
    }

    printf("\nCost = %f", cost(m));

    return 0;
}