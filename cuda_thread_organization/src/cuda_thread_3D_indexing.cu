#include <iostream>
#include <cmath>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

__global__
void print_matrix_values(int* matrix, int matrix_elems)
{
    // Row stride (with respect to x-axis)
    int row_stride = blockDim.x * gridDim.x;

    // Depth stride (with respect to z-axis)
    int depth_stride = row_stride * blockDim.y * gridDim.y;

    // Block offset is the offset to the beginning of the block in question
    int block_offset = depth_stride * blockDim.z * blockIdx.z + blockDim.x * blockIdx.x  + row_stride * blockDim.y * blockIdx.y;

    // Global index
    int gidx =  block_offset + depth_stride * threadIdx.z + row_stride * threadIdx.y + threadIdx.x;

    if(gidx < matrix_elems)
    {
        printf("blockIdx: (%d, %d, %d), threadIdx: (%d, %d, %d), global index: %d, value: %03d\n",
            blockIdx.x, blockIdx.y, blockIdx.z,
            threadIdx.x, threadIdx.y, threadIdx.z,
            gidx,
            matrix[gidx]
        );
    }
}

int main(int argc, char** argv)
{
    int nx, ny, nz;
    nx = 2; // Total number of threads in x-direction
    ny = 2; // Total number of threads in y-direction
    nz = 4; // Total number of threads in z-direction
    
    dim3 block(2, 2, 2);
    dim3 grid( nx/block.x, ny/block.y, nz/block.z);

    // Create a 3D matrix in the host
    int *h_matrix;
    dim3 matrix_dims(2, 2, 4);
    int matrix_size(matrix_dims.x * matrix_dims.y * matrix_dims.z);
    int matrix_size_bytes(matrix_size * sizeof(h_matrix));
    h_matrix = (int*)malloc(matrix_size_bytes);

    // Fill in the matrix so that the value encodes the position in the matrix.
    // E.g. 211 = (x = 2, y = 1, z = 1)
    int index = 0;
    for(int z = 0; z < matrix_dims.z; ++z)
    {
        for(int y = 0; y < matrix_dims.y; ++y)
        {
            for(int x = 0; x < matrix_dims.x; ++x)
            {
                h_matrix[index++] = x*100 + y*10 + z;
            }
        }
    }
    
    // Create memory for the 3D matrix in the device
    int *d_matrix;
    cudaMalloc((void**)&d_matrix, matrix_size_bytes);

    // Copy data from the host to the device
    cudaMemcpy(d_matrix, h_matrix, matrix_size_bytes, cudaMemcpyHostToDevice);

    // Print out the values in the matrix
    print_matrix_values<<<grid, block>>>(d_matrix, matrix_size);

    // Wait for all the kernels to finnish execution
    cudaDeviceSynchronize();

    // Free both host and device memory
    free(h_matrix);
    cudaFree(d_matrix);

    // Reset the device
    cudaDeviceReset();
    
    return 0;
}
