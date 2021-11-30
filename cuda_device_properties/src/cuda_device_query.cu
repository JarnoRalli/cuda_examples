#include <iostream>
#include <cmath>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include <stdio.h>

void query_device()
{
    int deviceCount = 0;

    cudaGetDeviceCount(&deviceCount);

    if(deviceCount == 0)
    {
        printf("No CUDA device found");
    }

    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);

    printf("Device %d: %s\n", devNo, iProp.name);
    printf("\tnumber of streaming multiprocessors:      %d\n", iProp.multiProcessorCount);
    printf("\tclock rate:                               %d\n", iProp.clockRate);
    printf("\tcompute capability:                       %d.%d\n", iProp.major, iProp.major);
    printf("\ttotal amount of memory:                   %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
    printf("\ttotal amount of shared memory per block:  %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
    printf("\ttotal amount of shared memory per SM:     %4.2f KB\n", iProp.sharedMemPerMultiprocessor/ 1024.0);
    printf("\ttotal number of registers per block:      %d\n", iProp.regsPerBlock);
    printf("\twarp size:                                %d\n", iProp.warpSize);
    printf("\tmaximum number of threads per block:      %d\n", iProp.maxThreadsPerBlock);
    printf("\tmaximum number of threads per SM:         %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("\tmaximum grid size:                        (%d, %d, %d)\n", iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
}

int main(int argc, char** argv)
{
    query_device();
    
    return 0;
}
