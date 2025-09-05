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
    float *data;
} mat;

#define MAT_AT(m, i, j) (m).data[(i) * (m).cols + (j)]

float rand_float(void);
mat mat_alloc(int rows, int cols);
void mat_init(mat m, float n);
void mat_rand(mat m, int lo, int hi);
void mat_mult(mat res, mat a, mat b);
void mat_add(mat res, mat a);
void mat_print(mat m);

#endif // NN_H





#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

mat mat_alloc(int rows, int cols)
{
    mat m;
    m.rows = rows;
    m.cols = cols;
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

    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.cols; j++)
        {
            
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

void mat_print(mat m)
{
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
            printf("%f ", MAT_AT(m, i, j));
        printf("\n");
    }
}




#endif // NN_IMPLEMENTATION