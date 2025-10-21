#include <stdio.h>
void main()
{
    float a = 3.0f;
    float da = 2.0f;
    float di = 2*da*a*(1-a);
    float c = 2.0f;
    c += di;
    float d = 2.0f;
    d += 2*da*a*(1-a);
    printf("\n%f", c);
    printf("\n%f", d);
}