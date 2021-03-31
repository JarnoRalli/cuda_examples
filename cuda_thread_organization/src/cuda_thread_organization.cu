#include <iostream>
#include <cmath>

#include <stdio.h>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

__global__
void print_threadIds()
{
    printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n",
           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char** argv)
{
    int nx, ny, nz;
    nx = 4; // Total number of threads in x-direction
    ny = 4; // Total number of threads in y-direction
    nz = 4; // Totan number of threads in z-direction

    dim3 block(2, 2, 2);
    dim3 grid(nx / block.x, ny / block.y, nz / block.z);

    // Will print blockId and threadId for each grid and block
    print_threadIds<<<grid, block>>>();

    // Wait for all the kernels to finnish execution
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
