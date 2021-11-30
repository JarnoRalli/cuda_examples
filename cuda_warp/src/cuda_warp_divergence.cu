#include <iostream>
#include <cmath>
#include <stdio.h>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

// Even if this kernel has if-else flow control, no warp divergence happens
// as the boundary of the condition happens on warp boundaries
__global__
void no_warp_divergence()
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gid / 32;
    unsigned int a = 0, b = 0;

    if(warp_id % 2 == 0)
    {
        a++;
    }else{
        b++;
    }
}

// If this kernel is built with -G debug option, which disables optimizations, then
// due to the if-else condition this kernel suffers from warp divergence
__global__
void warp_divergence()
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a = 0, b = 0;

    // Even threads execute the if-part and odd threads execute the else-part
    if(gid % 2 == 0)
    {
        a++;
    }else{
        b++;
    }
}

int main(int argc, char** argv)
{
    cudaProfilerStart();
    int size = 1 << 22;
    
    dim3 block_size(128);
    dim3 grid_size((size + block_size.x -1)/ block_size.x);

    no_warp_divergence<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    warp_divergence<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    cudaProfilerStop();
    cudaDeviceReset();

    return 0;
}
