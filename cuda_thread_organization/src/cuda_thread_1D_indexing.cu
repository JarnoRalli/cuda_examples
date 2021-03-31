#include <iostream>
#include <cmath>

#include <stdio.h>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

__global__
void print_vector_values(float* vector)
{
    // This is a so called monolithic kernel, which assumes
    // that the grid of threads processes the array in a single pass.
    
    // Thread index
    int tidx = threadIdx.x;

    // Block index
    int bidx = blockIdx.x;

    //Global index
    int gidx = blockDim.x * bidx + tidx;

    printf("blockIdx.x: %d, threadIdx.x: %d, vector[%d] = %f\n", bidx, tidx, gidx, vector[gidx]);
}

int main(int argc, char** argv)
{
    int nx;
    nx = 8; // Total number of threads in x-direction

    dim3 block(4); // Number of threads per block
    dim3 grid(nx / block.x);

    // Create a vector in the host
    using vector_t = float;
    vector_t *vector;
    vector = new vector_t [nx];

    std::cout << "Vector in the host: ";
    for(int i = 0; i < nx; ++i )
    {
        vector[i] = static_cast<float>(i) + 0.1f;
        if( i != nx -1 )
        {
            std::cout << vector[i] << ", ";
        }else{
            std::cout << vector[i] << std::endl;
        }
    }

    // Reserve memory in the GPU
    vector_t *vector_gpu;
    cudaMalloc(reinterpret_cast<void**>(&vector_gpu), sizeof(vector_t)*nx);

    // Copy data from host to the GPU
    cudaMemcpy(vector_gpu, vector, sizeof(vector_t)*nx, cudaMemcpyHostToDevice);

    // Print the values of the input vector
    std::cout << "Accessing the vector in the device:" << std::endl;
    print_vector_values<<<grid, block>>>(vector_gpu);

    // Wait for all the kernels to finnish execution
    cudaDeviceSynchronize();

    // Free memory
    cudaFree(vector_gpu);
    delete vector;

    cudaDeviceReset();
    return 0;
}
