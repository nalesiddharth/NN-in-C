#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

typedef struct
{
    int rows;
    int cols;
    int stride;
    float *data;
} mat;

typedef struct
{

    int count;
    mat *w; // weights
    mat *b; // biases
    mat *a; // activations
} nn;

#define MAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m)

#define ARRAY_LEN(a) sizeof((a)) / sizeof((a)[0])
#define NN_PRINT(net) nn_print(net, #net)
#define NN_INPUT_MAT(nn) (nn).a[0]
#define NN_OUTPUT_MAT(nn) (nn).a[(nn).count]

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

nn nn_alloc(int *arch, int arch_count);
void nn_init(nn net, float n);
nn nn_print(nn net, const char *name);
void nn_rand(nn net, int lo, int hi);
void nn_forward(nn net);
float nn_cost(nn net, mat tin, mat tout);
void nn_finite_diff(nn net, nn gradients, float eps, mat tin, mat tout);
void nn_learn(nn net, nn gradients, float rate);

void nn_backprop(nn net, nn gradients, mat tin, mat tout);

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
    printf("    %s = [\n", name);
    for (int i = 0; i < m.rows; i++)
    {
        printf("    ");
        for (int j = 0; j < m.cols; j++)
            printf("  %f ", MAT_AT(m, i, j));
        printf("\n");
    }
    printf("    ]\n\n");
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
    NN_ASSERT(dest.rows == src.rows);
    NN_ASSERT(dest.cols == src.cols);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
}

nn nn_alloc(int *arch, int arch_count)
{
    NN_ASSERT(arch_count > 0);
    nn net;

    net.count = arch_count - 1;
    net.w = malloc(sizeof(*net.w) * net.count);
    NN_ASSERT(net.w != NULL);
    net.b = malloc(sizeof(*net.b) * net.count);
    NN_ASSERT(net.b != NULL);
    net.a = malloc(sizeof(*net.a) * arch_count);
    NN_ASSERT(net.a != NULL);

    net.a[0] = mat_alloc(1, arch[0]);
    for (int i = 1; i < arch_count; i++)
    {
        net.w[i - 1] = mat_alloc(net.a[i - 1].cols, arch[i]);
        net.b[i - 1] = mat_alloc(1, arch[i]);
        net.a[i] = mat_alloc(1, arch[i]);
    }

    return net;
}

void nn_init(nn net, float n)
{
    for(int i = 0; i<net.count; i++)
    {
        mat_init(net.w[i], n);
        mat_init(net.b[i], n);
        mat_init(net.a[i], n);
    }
    mat_init(net.a[net.count], n);
}

nn nn_print(nn net, const char *name)
{
    char buf[256];
    printf("\n\n%s:", name);
    printf("\nWeights: \n");
    for (int i = 0; i < net.count; i++)
    {
        snprintf(buf, sizeof(buf), "w[%d]", i);
        mat_print(net.w[i], buf);
    }
    printf("\nBiases: \n");
    for (int i = 0; i < net.count; i++)
    {
        snprintf(buf, sizeof(buf), "b[%d]", i);
        mat_print(net.b[i], buf);
    }
    printf("\nActivations: \n");
    mat_print(net.a[0], "a[0]");
    for (int i = 1; i <= net.count; i++)
    {
        snprintf(buf, sizeof(buf), "a[%d]", i);
        mat_print(net.a[i], buf);
    }
}

void nn_rand(nn net, int lo, int hi)
{
    for (int i = 0; i < net.count; i++)
    {
        mat_rand(net.w[i], lo, hi);
        mat_rand(net.b[i], lo, hi);
        // mat_rand(net.a[i+1], lo, hi); Activation matrices don't need initialization. I think.
    }
}

void nn_forward(nn net)
{
    for (int i = 0; i < net.count; i++)
    {
        mat_mult(net.a[i + 1], net.a[i], net.w[i]);
        mat_add(net.a[i + 1], net.b[i]);
        mat_sigmoidf(net.a[i + 1]);
    }
}

float nn_cost(nn net, mat tin, mat tout)
{
    NN_ASSERT(tin.rows == tout.rows);
    NN_ASSERT(tout.cols == NN_OUTPUT_MAT(net).cols);
    float c = 0.0f;
    float d;
    for (int i = 0; i < tin.rows; i++)
    {
        mat x = mat_getRow(tin, i);  // expected input
        mat y = mat_getRow(tout, i); // expected output
        mat_cpy(NN_INPUT_MAT(net), x);
        nn_forward(net);
        for (int j = 0; j < tout.cols; j++) // loop only runs once in case of arch=(2, 2, 1), but in the future, for multidimensional outputs (ie. outputs with multiple cols) the loop is necessary
        {
            d = MAT_AT(NN_OUTPUT_MAT(net), 0, j) - MAT_AT(y, 0, j);
            //printf("d in cost: %d\n");
            c += d * d;
        }
    }
    return c / tin.rows;
}

void nn_finite_diff(nn net, nn gradients, float eps, mat tin, mat tout)
{

    float temp;
    float cost = nn_cost(net, tin, tout);
    for (int i = 0; i < net.count; i++)
    {
        for (int j = 0; j < net.w[i].rows; j++)
        {
            for (int k = 0; k < net.w[i].cols; k++)
            {
                temp = MAT_AT(net.w[i], j, k);
                MAT_AT(net.w[i], j, k) += eps;
                MAT_AT(gradients.w[i], j, k) = (nn_cost(net, tin, tout) - cost) / eps;
                MAT_AT(net.w[i], j, k) = temp;
            }
        }

        for (int j = 0; j < net.b[i].rows; j++)
        {
            for (int k = 0; k < net.b[i].cols; k++)
            {
                temp = MAT_AT(net.b[i], j, k);
                MAT_AT(net.b[i], j, k) += eps;
                MAT_AT(gradients.b[i], j, k) = (nn_cost(net, tin, tout) - cost) / eps;
                MAT_AT(net.b[i], j, k) = temp;
            }
        }
    }
}

void nn_backprop(nn net, nn gradients, mat tin, mat tout)
{
    NN_ASSERT(tin.rows == tout.rows);
    NN_ASSERT(NN_OUTPUT_MAT(net).cols == tout.cols);
    int n = tin.rows; 
    nn_init(gradients, 0.0f);

    for(int i = 0; i<n; i++) //iterating through the samples
    {
        mat_cpy(NN_INPUT_MAT(net), mat_getRow(tin, i));
        nn_forward(net);

        for(int j = 0; j<gradients.count; j++) //resetting the gradient activations to 0 to avoid unnecessary accumulation 
            mat_init(gradients.a[j], 0.0f);
        
        for(int j = 0; j<tin.cols; j++)
        {   //The activation layer of the gradient NN can be used as an intermediate storage since it is unused (and wasted) in the finite difference implementation.
            //Here we initialize the backprop by setting the last layer's actual vs expected difference. The rest will be calculated further inside the loops.
            MAT_AT(NN_OUTPUT_MAT(gradients), 0, j) = MAT_AT(NN_OUTPUT_MAT(net), 0, j) - MAT_AT(tout, i, j);
        }

        for(int l = net.count-1; l>=0; l--) //iterating through the layers. A little dicey to understand because of count(activations) == count(weight)+1, but that's how the indexing system of the architecture inherently works.
        {
            for(int j = 0; j<net.a[l+1].cols; j++) //iterating through each neuron of the layer.
            {
                float a = MAT_AT(net.a[l+1], 0, j);
                float da = MAT_AT(gradients.a[l+1], 0, j);
                float di = 2*da*a*(1-a);
                MAT_AT(gradients.b[l], 0, j) += di;
                for(int k = 0; k<net.a[l].cols; k++) // iterating/accessing the parameters of each neuron from the last layer
                {
                    MAT_AT(gradients.w[l], k, j) += (di * MAT_AT(net.a[l], 0, k));
                    MAT_AT(gradients.a[l], 0, k) += (di * MAT_AT(net.w[l], k, j));
                }
            }
        }
    }

    //calculating average gradients. Excuse the confusion, I wanted to use as few nested loops as possible
    for(int i = 0; i<gradients.count; i++)
    {
        NN_ASSERT(gradients.w[i].cols == gradients.b[i].cols);
        for(int j=0; j<gradients.w[i].cols; j++)
        {
            MAT_AT(gradients.b[i], 0, j) /= n; //calculated here since bias matrix only ever has 1 row
            for(int k = 0; k<gradients.w[i].rows; k++)
                MAT_AT(gradients.w[i], k, j) /= n;
        }
    }
}

void nn_learn(nn net, nn gradients, float rate)
{
    for (int i = 0; i < net.count; i++)
    {
        for (int j = 0; j < net.w[i].rows; j++)
            for (int k = 0; k < net.w[i].cols; k++)
                MAT_AT(net.w[i], j, k) -= rate * MAT_AT(gradients.w[i], j, k);

        for (int j = 0; j < net.b[i].rows; j++)
            for (int k = 0; k < net.b[i].cols; k++)
                MAT_AT(net.b[i], j, k) -= rate * MAT_AT(gradients.b[i], j, k);
    }
}




#endif // NN_IMPLEMENTATION