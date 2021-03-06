#include <iostream>
#include <cmath>

#include <stdio.h>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "tools/timing.hpp"
#include "tools/error_handling.hpp"

// Vector element type
using vector_t = float;

__global__
void addKernel(int n, float *result, float *x, float *y, float *z)
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
        result[i] = x[i] + y[i] + z[i];
    }
}

void add(int n, float *result, float *x, float *y, float *z)
{
    for(unsigned int idx = 0; idx != n; ++idx)
    {
        result[idx] = x[idx] + y[idx] + z[idx];
    }
}

int main(int argc, char** argv)
{
    int N = 1 << 22;

    vector_t *h_x, *h_y, *h_z, *h_result_cpu, *h_result_gpu;

    // Allocate memory on the host
    if( (h_x = new(std::nothrow) vector_t[N]) == nullptr )
    {
        std::cerr << "Failed to reserve memory for 'h_x'" << std::endl;
        exit(1);
    }

    if( (h_y = new(std::nothrow) vector_t[N]) == nullptr )
    {
        std::cerr << "Failed to reserve memory for 'h_y'" << std::endl;
        exit(1);
    }

    if( (h_z = new(std::nothrow) vector_t[N]) == nullptr )
    {
        std::cerr << "Failed to reserve memory for 'h_z'" << std::endl;
        exit(1);
    }

    if( (h_result_cpu = new(std::nothrow) vector_t[N]) == nullptr )
    {
        std::cerr << "Failed to reserve memory for 'h_result'" << std::endl;
        exit(1);
    }
    
    if( (h_result_gpu = new(std::nothrow) vector_t[N]) == nullptr )
    {
        std::cerr << "Failed to reserve memory for 'h_result_gpu'" << std::endl;
        exit(1);
    }

    // Initialize the memory on the host device
    for(int i = 0; i < N; ++i)
    {
        h_x[i] = vector_t(1.0);
        h_y[i] = vector_t(2.0);
        h_z[i] = vector_t(3.0);
    }

    auto start_time = std::chrono::steady_clock::now();
    add(N, h_result_cpu, h_x, h_y, h_z);
    auto cpu_execution_time = since(start_time);

    // Memory on the device
    vector_t *d_x, *d_y, *d_z, *d_result_gpu;

    // Allocate space in the device
    gpuErrchk(cudaMalloc((vector_t**)&d_x, N*sizeof(vector_t)));
    gpuErrchk(cudaMalloc((vector_t**)&d_y, N*sizeof(vector_t)));
    gpuErrchk(cudaMalloc((vector_t**)&d_z, N*sizeof(vector_t)));
    gpuErrchk(cudaMalloc((vector_t**)&d_result_gpu, N*sizeof(vector_t)));

    // Transfer the data to the device
    start_time = std::chrono::steady_clock::now();
    gpuErrchk(cudaMemcpy(d_x, h_x, N*sizeof(vector_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, h_y, N*sizeof(vector_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_z, h_z, N*sizeof(vector_t), cudaMemcpyHostToDevice));
    auto cpy_host2device_time = since(start_time);

    // Run the kernel on the elements
    int blockSize = 256; // For efficiency, this needs to be a multiple of 32
    int gridSize = (N + blockSize - 1) / blockSize;

    start_time = std::chrono::steady_clock::now();
    addKernel<<<gridSize, blockSize>>>(N, d_result_gpu, d_x, d_y, d_z);

    // Wait for all the kernels to finnish execution
    gpuErrchk(cudaDeviceSynchronize());
    auto gpu_execution_time = since(start_time);

    // Transfer the results back to the host
    start_time = std::chrono::steady_clock::now();
    gpuErrchk(cudaMemcpy(h_result_gpu, d_result_gpu, N*sizeof(vector_t), cudaMemcpyDeviceToHost));
    auto cpy_device2host_time = since(start_time);

    //Check for errors (all values should be 6.0)
    vector_t maxError_cpu = vector_t(0.0);

    for(int i = 0; i < N; ++i)
    {
        maxError_cpu = std::max(maxError_cpu, std::abs(h_result_cpu[i] - vector_t(6.0)));
    }

    //Check for errors (all values should be 6.0)
    vector_t maxError_gpu = vector_t(0.0);

    for(int i = 0; i < N; ++i)
    {
        maxError_gpu = std::max(maxError_gpu, std::abs(h_result_gpu[i] - vector_t(6.0)));
    }

    std::cout << "CPU execution time (ns): " << cpu_execution_time.count() << std::endl;
    std::cout << "GPU execution time (ns): " << gpu_execution_time.count() << std::endl;
    std::cout << "Host to device transfer time (ns): " << cpy_host2device_time.count() << std::endl;
    std::cout << "Device to hose transfer time (ns): " << cpy_device2host_time.count() << std::endl;
    std::cout << "CPU add maximum error: " << maxError_cpu;
    std::cout << "GPU add maximum error: " << maxError_gpu;

    // Free host memory
    delete[] h_x;
    delete[] h_y;
    delete[] h_z;
    delete[] h_result_cpu;

    // Free device memory
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_y));
    gpuErrchk(cudaFree(d_z));
    gpuErrchk(cudaFree(d_result_gpu));

    cudaDeviceReset();
    return 0;
}
