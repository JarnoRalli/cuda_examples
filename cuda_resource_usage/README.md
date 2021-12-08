# Resource Usage per Thread

The example `cuda_add_managed_memory.cu` contains a very simple kernel for adding vectors together. The kernel is as follows:

```
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
```

The number of registers, and shared memory, per thread can be queried/visualized by compiling the kernel in question with the flags `--ptxas-options=-v`, for example

```
nvcc --ptxas-options=-v -o test.out cuda_add_managed_memory.cu
cuda_add_managed_memory.cu
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z9addKerneliPfS_' for 'sm_52'
ptxas info    : Function properties for _Z9addKerneliPfS_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, 344 bytes cmem[0]
   Creating library test.lib and object test.exp
```

CMake target called `resource_usage_add_managed_memory` is built with the flags `--ptxas-options=-v`. In CMake this can be achieved, per target, as follows:

```
target_compile_options(profile_add_managed_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
```
