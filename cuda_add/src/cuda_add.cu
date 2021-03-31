#include <iostream>
#include <cmath>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

__global__
void addKernel(int n, float *x, float *y)
{
    // This is a so called grid-stride-loop, which ensures
    // that add addressing within warps is unit-stride, and thus
    // achieves maximum memory coalescing.
    // For more information: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    // and https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char** argv)
{
    int N = 1 << 20; //1M elements

    float *x, *y;

    // Allocate memory, uses unified memory
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // Initialize the memory on the host device
    for( int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run the kernel on the elements
    int blockSize = 256; // For efficiency, this needs to be a multiple of 32
    int numBlocks = (N + blockSize -1) / blockSize;
    
    std::cout << "Number of elements: " << N << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "Number of blocks: " << numBlocks << std::endl;
    
    addKernel<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for all the kernels to finnish execution
    cudaDeviceSynchronize();

    //Check for errors (all values should be 3.0f)
    float maxError = 0.0f;

    for( int i = 0; i < N; ++i )
    {
        maxError = std::max(maxError, std::abs(y[i] - 3.0f));
    }

    cudaFree(x);
    cudaFree(y);

    std::cout << "Max error: " << maxError << std::endl;

    cudaDeviceReset();
    return 0;
}
