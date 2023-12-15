#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

__global__ void add(uintptr_t* clock)
{
    ++clock;
    if(clock==10)
        printf("success");

}

int main(int argc, char** argv)
{
    uintptr_t iters;
    uintptr_t* d_iters;
    iters = 0;
    cudaMalloc(&d_iters,sizeof(uintptr_t));
    cudaMemcpy(d_iters,&iters,sizeof(uintptr_t),cudaMemcpyHostToDevice);
    for(int i=0;i<10;i++)
    {
        
        add<<<1,1>>>(*d_iters);
        cudaMemcpy(iters,d_iters,sizeof(uintptr_t)
    }
    
    return 0;
}
