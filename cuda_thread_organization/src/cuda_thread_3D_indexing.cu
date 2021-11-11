#include <iostream>
#include <cmath>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

// Matrix element type
using matrix_t = int;

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
void print_matrix_values(matrix_t* matrix, int matrix_elems)
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
    dim3 grid(nx/block.x, ny/block.y, nz/block.z);

    // Create a 3D matrix in the host
    matrix_t *h_matrix;
    dim3 matrix_dims(2, 2, 4);
    int matrix_size(matrix_dims.x * matrix_dims.y * matrix_dims.z);
    int matrix_size_bytes(matrix_size * sizeof(matrix_t));
    h_matrix = new matrix_t [matrix_size];

    // Fill in the matrix so that the value encodes the position in the matrix.
    // E.g. 211 = (x = 2, y = 1, z = 1)
    int index = 0;
    for(int z = 0; z < matrix_dims.z; ++z)
    {
        for(int y = 0; y < matrix_dims.y; ++y)
        {
            for(int x = 0; x < matrix_dims.x; ++x)
            {
                h_matrix[index++] = static_cast<matrix_t>(x) * matrix_t(100) + static_cast<matrix_t>(y) * matrix_t(10) + static_cast<matrix_t>(z);
            }
        }
    }

    // Create memory for the 3D matrix in the device
    matrix_t *d_matrix;
    gpuErrchk(cudaMalloc((void**)&d_matrix, matrix_size_bytes));

    // Copy data from the host to the device
    gpuErrchk(cudaMemcpy(d_matrix, h_matrix, matrix_size_bytes, cudaMemcpyHostToDevice));

    // Print out the values in the matrix
    std::cout << "Accessing the matrix in the device:" << std::endl;
    print_matrix_values<<<grid, block>>>(d_matrix, matrix_size);

    // Wait for all the kernels to finnish execution
    gpuErrchk(cudaDeviceSynchronize());

    // Free both host and device memory
    delete [] h_matrix;
    gpuErrchk(cudaFree(d_matrix));

    // Reset the device
    gpuErrchk(cudaDeviceReset());

    return 0;
}
