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

#define ARRAY_LEN(a) (sizeof((a)) / sizeof((a)[0]))
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

/* New: backprop over a batch of sample indices (indices array, idx_count)
   indices are 0..tin.rows-1 referring to rows of tin/tout */
void nn_backprop_batch(nn net, nn gradients, mat tin, mat tout, int *indices, int idx_count);

void nn_learn(nn net, nn gradients, float rate);

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
    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.cols; j++)
        {
            float s = 0.0f;
            for (int k = 0; k < b.rows; k++)
                s += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            MAT_AT(res, i, j) = s;
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
    }
}

/* Activation strategy:
   - hidden layers: ReLU
   - output layer: Sigmoid (keeps brightness 0..1)
*/
static inline float act_derivative(int layer_is_output, float activated_value)
{
    if (layer_is_output)
    {
        // sigmoid derivative: a * (1 - a)
        return activated_value * (1.0f - activated_value);
    }
    else
    {
        // ReLU derivative from activated value: if activated_value > 0 -> 1 else 0
        return (activated_value > 0.0f) ? 1.0f : 0.0f;
    }
}

void nn_forward(nn net)
{
    for (int i = 0; i < net.count; i++)
    {
        mat_mult(net.a[i + 1], net.a[i], net.w[i]);
        mat_add(net.a[i + 1], net.b[i]);
        // activation
        if (i == net.count - 1)
            mat_sigmoidf(net.a[i + 1]); // output layer
        else
        {
            // ReLU for hidden
            for (int r = 0; r < net.a[i + 1].rows; r++)
                for (int c = 0; c < net.a[i + 1].cols; c++)
                {
                    float v = MAT_AT(net.a[i + 1], r, c);
                    MAT_AT(net.a[i + 1], r, c) = (v > 0.0f) ? v : 0.0f;
                }
        }
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
        for (int j = 0; j < tout.cols; j++)
        {
            d = MAT_AT(NN_OUTPUT_MAT(net), 0, j) - MAT_AT(y, 0, j);
            c += d * d;
        }
    }
    return c / tin.rows;
}

/* Backprop over a list of sample indices (mini-batch).
   The grads structure should have same architecture as net and will be filled with averaged gradients across the batch.
*/
void nn_backprop_batch(nn net, nn gradients, mat tin, mat tout, int *indices, int idx_count)
{
    NN_ASSERT(tin.rows == tout.rows);
    NN_ASSERT(NN_OUTPUT_MAT(net).cols == tout.cols);
    int n = idx_count;
    nn_init(gradients, 0.0f);

    for (int sample_idx = 0; sample_idx < n; sample_idx++)
    {
        int i = indices[sample_idx];
        mat_cpy(NN_INPUT_MAT(net), mat_getRow(tin, i));
        nn_forward(net);

        // zero intermediate gradient activations
        for (int j = 0; j < gradients.count; j++)
            mat_init(gradients.a[j], 0.0f);

        // set output-layer delta = (a - y)
        for (int j = 0; j < tout.cols; j++)
            MAT_AT(NN_OUTPUT_MAT(gradients), 0, j) = MAT_AT(NN_OUTPUT_MAT(net), 0, j) - MAT_AT(tout, i, j);

        // backpropagate layer by layer
        for (int l = net.count - 1; l >= 0; l--)
        {
            int is_output_layer = (l == net.count - 1);
            for (int j = 0; j < net.a[l + 1].cols; j++)
            {
                float a = MAT_AT(net.a[l + 1], 0, j);
                float da = MAT_AT(gradients.a[l + 1], 0, j);
                float deriv = act_derivative(is_output_layer, a);
                float di = 2.0f * da * deriv; // 2*(a-y)*activation'
                MAT_AT(gradients.b[l], 0, j) += di;
                for (int k = 0; k < net.a[l].cols; k++)
                {
                    MAT_AT(gradients.w[l], k, j) += (di * MAT_AT(net.a[l], 0, k));
                    MAT_AT(gradients.a[l], 0, k) += (di * MAT_AT(net.w[l], k, j));
                }
            }
        }
    }

    // average gradients across batch
    if (n > 0)
    {
        for (int i = 0; i < gradients.count; i++)
        {
            NN_ASSERT(gradients.w[i].cols == gradients.b[i].cols);
            for (int j = 0; j < gradients.w[i].cols; j++)
            {
                MAT_AT(gradients.b[i], 0, j) /= n;
                for (int k = 0; k < gradients.w[i].rows; k++)
                    MAT_AT(gradients.w[i], k, j) /= n;
            }
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
