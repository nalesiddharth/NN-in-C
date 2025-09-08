#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stddef.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

typedef struct {
    int rows;
    int cols;
    int stride;
    float *data;
} mat;

#define MAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m)

float rand_float(void);
float sigmoidf(float x);
 
mat mat_alloc(int rows, int cols);
void mat_init(mat m, float n);
void mat_rand(mat m, int lo, int hi);
void mat_mult(mat res, mat a, mat b);
void mat_add(mat res, mat a);
void mat_print(mat m, const char *name);
void mat_sigmoidf(mat m);
mat mat_getRow(mat m, int row);
void mat_cpy(mat dest, mat src);

#endif // NN_H





#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    float s = 1.0f / (1.0f + expf(-x));
    return s;
}

mat mat_alloc(int rows, int cols)
{
    mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = NN_MALLOC(sizeof(*m.data) * rows * cols);
    NN_ASSERT(m.data != NULL);
    return m;
}

void mat_init(mat m, float n)
{
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = n;
}

void mat_rand(mat m, int lo, int hi)
{
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = rand_float() * (hi - lo) + lo;
}

void mat_mult(mat res, mat a, mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(res.rows == a.rows);
    NN_ASSERT(res.cols == b.cols);
    float tempSum = 0.0f;
    float temp = 0.0f;
    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.cols; j++)
        {
            MAT_AT(res, i, j) = 0.0f;
            for (int k = 0; k < b.rows; k++)
                MAT_AT(res, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
        }
}

void mat_add(mat res, mat a)
{
    NN_ASSERT(res.rows == a.rows);
    NN_ASSERT(res.cols == a.cols);
    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.cols; j++)
            MAT_AT(res, i, j) += MAT_AT(a, i, j);
}

void mat_print(mat m, const char *name)
{
    printf("%s = [\n", name);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
            printf("  %f ", MAT_AT(m, i, j));
        printf("\n");
    }
    printf("]\n\n");
}

void mat_sigmoidf(mat m)
{
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
}

mat mat_getRow(mat m, int row)
{
    mat x = {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .data = &MAT_AT(m, row, 0),
    };

    return x;
}

void mat_cpy(mat dest, mat src)
{
    NN_ASSERT(dest.rows = src.rows);
    NN_ASSERT(dest.cols = src.cols);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
}


#endif // NN_IMPLEMENTATION