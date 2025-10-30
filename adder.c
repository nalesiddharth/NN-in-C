#define NN_IMPLEMENTATION
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "nn.h"

#define NBITS 3

const int ip_stride = 2 * NBITS;
const int op_stride = NBITS + 1;
const int row_count = (1<<NBITS)*(1<<NBITS);


void main()
{
    float training_input[row_count * ip_stride];
    float training_output[row_count * op_stride];
    printf("BITS = %d, ip_stride = %d, row_count = %d, (1<<NBITS) = %d\n", NBITS, ip_stride, row_count, (1<<NBITS));

    //generating input training data
    float bit;
    for(int j = 1; j<=ip_stride; j++)
    {
        bit = 0.0f;
        for(int i = 1; i<=row_count; i++)
        {   
            int p = (int) pow(2, j-1);
            training_input[(i-1)*ip_stride + (ip_stride-j)] = bit;
            if(i%p == 0)
                bit = (bit == 0.0f)? 1.0f : 0.0f;
        }
    }

    //generating output training data
    for (int i = 0; i < row_count; i++)
    {
        int A = 0;
        int B = 0;

        for (int b = 0; b < NBITS; b++)
        {
            A = (A << 1) | (int)training_input[i * ip_stride + b];
            B = (B << 1) | (int)training_input[i * ip_stride + NBITS + b];
        }
        int sum = A + B;

        for (int b = 0; b < op_stride; b++)
        {
            int bitval = (sum >> (op_stride - 1 - b)) & 1;
            training_output[i * op_stride + b] = (float)bitval;
        }
    }
    mat tin = {
        .rows = row_count,
        .cols = ip_stride,
        .stride = ip_stride,
        .data = training_input};
        
    mat tout = {
        .rows = row_count,
        .cols = op_stride,
        .stride = op_stride,
        .data = training_output};

    // MAT_PRINT(tin);
    // MAT_PRINT(tout);

    // srand(time(0));
    srand(70);
    int arch[] = {ip_stride, ip_stride+1, op_stride};
    nn addnet = nn_alloc(arch, ARRAY_LEN(arch));
    nn g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(addnet, 0, 1);
    printf("\ncost = %f", nn_cost(addnet, tin, tout));
    float rate = 1;
    float eps = 1e-1;

    int train_count = 500000;
    for (int i = 0; i < train_count; i++)
    {
        #if 0
        nn_finite_diff(addnet, g, eps, tin, tout);
        #else
        nn_backprop(addnet, g, tin, tout);
        #endif
        //NN_PRINT(xor_g);
        nn_learn(addnet, g, rate);
        if (i % (train_count / 100) == 0)
            printf("\ncost = %f", nn_cost(addnet, tin, tout));
    }
    printf("\n\n");
    //MAT_PRINT(tin);

    //inference
    int t;
    for(int i = 0; i<row_count; i++)
    {
        NN_INPUT_MAT(addnet) = mat_getRow(tin, i);
        //MAT_PRINT(NN_INPUT_MAT(addnet));
        nn_forward(addnet);
        for(int j = 0; j<ip_stride/2; j++)
        {
            t = (MAT_AT(mat_getRow(tin, i), 0, j)) ? 1 : 0;
            printf("%d ", t);
        }
        printf("+ ");
        for(int j = ip_stride/2; j<ip_stride; j++)
        {
            t = (MAT_AT(mat_getRow(tin, i), 0, j)) ? 1 : 0;
            printf("%d ", t);
        }
        printf(" =  ");
        for(int j = 0; j<op_stride; j++)
        {
            //MAT_PRINT(NN_OUTPUT_MAT(addnet));
            t = (MAT_AT(NN_OUTPUT_MAT(addnet), 0, j) > 0.9f)? 1 : 0;
            printf("%d ", t);
        }
        printf("\n");
    }
    //NN_PRINT(addnet);
}