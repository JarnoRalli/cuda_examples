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

    std::cout << "Device [" << devNo << "]: " << iProp.name << std::endl;
    std::cout << "\tcompute capability:                       " << iProp.major << "." << iProp.minor << std::endl;
    std::cout << "\tnumber of streaming multiprocessors:      " << iProp.multiProcessorCount << std::endl;
    std::cout << "\tclock rate:                               " << iProp.clockRate << std::endl;
    std::cout << "\ttotal amount of memory:                   " << iProp.totalGlobalMem / 1024.0 << "KB" << std::endl;
    std::cout << "\ttotal amount of shared memory per block:  " << iProp.sharedMemPerBlock / 1024.0 << "KB" << std::endl;
    std::cout << "\ttotal amount of shared memory per SM:     " << iProp.sharedMemPerMultiprocessor/ 1024.0 << "KB" << std::endl;
    std::cout << "\ttotal number of registers per block:      " << iProp.regsPerBlock << "KB" << std::endl;
    std::cout << "\twarp size:                                " << iProp.warpSize << "KB" << std::endl;
    std::cout << "\tmaximum number of threads per block:      " << iProp.maxThreadsPerBlock << "KB" << std::endl;
    std::cout << "\tmaximum number of threads per SM:         " << iProp.maxThreadsPerMultiProcessor << "KB" << std::endl;
    std::cout << "\tmaximum grid size:                        (" << iProp.maxGridSize[0] << ", " << iProp.maxGridSize[1] << ", " << iProp.maxGridSize[2] << ")" << std::endl;
}

int main(int argc, char** argv)
{
    query_device();
    
    return 0;
}
