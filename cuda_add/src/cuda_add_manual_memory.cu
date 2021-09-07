#include <iostream>
#include <cmath>

#include <stdio.h>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t error_code, const char *file, int line, bool abort = true)
{
    if(error_code != cudaSuccess)
    {
        fprintf(stderr, "gpuAssert: %s %s %d\n", cudaGetErrorString(error_code), file, line);
        if(abort)
        {
            exit(error_code);
        }
    }
}

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

int main(int argc, char** argv)
{
    int N = 1 << 22;

    float *h_x, *h_y, *h_z, *h_result;
    
    clock_t cpu_execution_start, cpu_execution_end;
    cpu_execution_start = clock();

    // Allocate memory on the host
    if( (h_x = (float*)malloc(N*sizeof(float))) == nullptr )
    {
        fprintf(stderr, "Failed to reserve memory for 'h_x'");
    }

    if( (h_y = (float*)malloc(N*sizeof(float))) == nullptr )
    {
        fprintf(stderr, "Failed to reserve memory for 'h_y'");
    }

    if( (h_z = (float*)malloc(N*sizeof(float))) == nullptr )
    {
        fprintf(stderr, "Failed to reserve memory for 'h_z'");
    }

    if( (h_result = (float*)malloc(N*sizeof(float))) == nullptr )
    {
        fprintf(stderr, "Failed to reserve memory for 'h_result'");
    }

    // Initialize the memory on the host device
    for(int i = 0; i < N; ++i)
    {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
        h_z[i] = 3.0f;
    }

    // Memory on the device
    float *d_x, *d_y, *d_z, *d_result;

    // Allocate space in the device
    gpuErrchk(cudaMalloc((float**)&d_x, N*sizeof(float)));
    gpuErrchk(cudaMalloc((float**)&d_y, N*sizeof(float)));
    gpuErrchk(cudaMalloc((float**)&d_z, N*sizeof(float)));
    gpuErrchk(cudaMalloc((float**)&d_result, N*sizeof(float)));
    cpu_execution_end = clock();

    clock_t host_to_device_start, host_to_device_end;
    host_to_device_start = clock();
    // Transfer the data to the device
    gpuErrchk(cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_z, h_z, N*sizeof(float), cudaMemcpyHostToDevice));
    host_to_device_end = clock();

    // Run the kernel on the elements
    int blockSize = 256; // For efficiency, this needs to be a multiple of 32
    int numBlocks = (N + blockSize - 1) / blockSize;

    clock_t gpu_execution_start, gpu_execution_end;
    gpu_execution_start = clock();
    addKernel<<<numBlocks, blockSize>>>(N, d_result, d_x, d_y, d_z);

    // Wait for all the kernels to finnish execution
    gpuErrchk(cudaDeviceSynchronize());
    gpu_execution_end = clock();

    clock_t device_to_host_start, device_to_host_end;
    device_to_host_start = clock();
    // Transfer the results back to the host
    gpuErrchk(cudaMemcpy(h_result, d_result, N*sizeof(float), cudaMemcpyDeviceToHost));
    device_to_host_end = clock();

    printf("GPU execution time: %4.6f\n", (double)((double)(gpu_execution_end - gpu_execution_start) / CLOCKS_PER_SEC));
    printf("CPU execution time: %4.6f\n", (double)((double)(cpu_execution_end - cpu_execution_start) / CLOCKS_PER_SEC));
    printf("Host to device transfer time: %4.6f\n", (double)((double)(host_to_device_end - host_to_device_start) / CLOCKS_PER_SEC));
    printf("Device to hose transfer time: %4.6f\n", (double)((double)(device_to_host_end - device_to_host_start) / CLOCKS_PER_SEC));

    // Free host memory
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_result);

    // Free device memory
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_y));
    gpuErrchk(cudaFree(d_z));
    gpuErrchk(cudaFree(d_result));

    cudaDeviceReset();
    return 0;
}
