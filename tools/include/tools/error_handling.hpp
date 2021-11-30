#pragma once

#include "cuda_runtime.h"

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

