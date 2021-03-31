#include <iostream>
#include <cmath>

#include <stdio.h>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

__global__
void hello_cuda()
{
    printf("Hello world from CUDA\n");
}

int main(int argc, char** argv)
{
    int nx, ny;
    nx = 16; // Total number of threads in x-direction
    ny = 4;  // Total number of threads in y-direction

    dim3 block(8, 2, 1);
    dim3 grid(nx / block.x, ny / block.y, 1);

    // Will print "Hello world..." 16 * 4 times
    hello_cuda<<<grid, block>>>();

    // Wait for all the kernels to finnish execution
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
