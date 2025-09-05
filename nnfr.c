#define NN_IMPLEMENTATION
#include <stdio.h>
#include <time.h>
#include "nn.h"


int main()
{
    srand(time(0));
    mat m = mat_alloc(2, 2);
    mat_rand(m, 0, 10);
    mat_print(m);

    return 0;
}