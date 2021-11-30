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
void print_vector_values(vector_t* vector)
{
    // This is a so called monolithic kernel, which assumes
    // that the grid of threads processes the array in a single pass.

     // Row stride
     int row_stride = blockDim.x * gridDim.x;

    // Block offset is the offset to the beginning of the block in question
    int block_offset = blockDim.x * blockIdx.x  + row_stride * blockDim.y * blockIdx.y;

    //Global index
    int gidx =  block_offset + row_stride * threadIdx.y + threadIdx.x;

    printf("blockIdx: (%d, %d), threadIdx: (%d, %d), value: %f\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, vector[gidx]);
}

int main(int argc, char** argv)
{
    int nx, ny;
    nx = 8; // Total number of threads in x-direction
    ny = 4; // Total number of threads in y-direction

    dim3 block(4, 2); // Number of threads per block
    dim3 grid(nx / block.x, ny / block.y);

    // Create a vector in the host. This vector represents
    // the following 2D data:
    // 11, 12, 13, 14, 15, 16, 17, 18
    // 21, 22, 23, 24, 25, 26, 27, 28
    // 31, 32, 33, 34, 35, 36, 37, 38
    // 41, 42, 43, 44, 45, 46, 47, 48
    vector_t vector[] = {
        11, 12, 13, 14, 15, 16, 17, 18, // 8 elements
        21, 22, 23, 24, 25, 26, 27, 28, // 8 elements
        31, 32, 33, 34, 35, 36, 37, 38, // 8 elements
        41, 42, 43, 44, 45, 46, 47, 48 }; // 8 elements

    // Reserve memory in the GPU
    vector_t *vector_gpu;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&vector_gpu), sizeof(vector_t)*32));

    // For instrumentation
    auto start_time = std::chrono::steady_clock::now();

    // Copy data from host to the GPU
    gpuErrchk(cudaMemcpy(vector_gpu, &vector, sizeof(vector_t)*32, cudaMemcpyHostToDevice));
    auto host2device_cpy_duration = since(start_time);

    // Print the values of the input vector
    std::cout << "Accessing the vector in the device:" << std::endl;
    start_time = std::chrono::steady_clock::now();
    print_vector_values<<<grid, block>>>(vector_gpu);

    // Wait for all the kernels to finnish execution
    gpuErrchk(cudaDeviceSynchronize());
    auto execution_duration = since(start_time);

    // Print memory transfer and kernel execution times
    std::cout << "Host to device memcpy time (ns) : " << host2device_cpy_duration.count() << std::endl;
    std::cout << "Kernel execution time (ns) : " << execution_duration.count() << std::endl;

    // Free memory
    gpuErrchk(cudaFree(vector_gpu));

    gpuErrchk(cudaDeviceReset());
    return 0;
}
