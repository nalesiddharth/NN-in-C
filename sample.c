#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
void main()
{
    int f1 = time(0);
    sleep(10);
    int f2 = time(0);
    int f3 = f2-f1;
    printf("TimeDiff = %d\n%d\n= %d", f1, f2, f3);
}