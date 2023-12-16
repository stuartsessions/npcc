#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
__managed__ uintptr_t buffer[1000];

__global__ void add()
{
    for(int i=0;i<1000;i++)
    {
        buffer[i]=(uintptr_t)i;
    }
}

int main(int argc, char** argv)
{
    add<<<1,1>>>();
    for(int i=0;i<1000;i++)
    {
        printf("%lu",buffer[i]);
    }
    return 0;
}
