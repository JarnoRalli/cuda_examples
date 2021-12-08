# CUDA Warp

A warp simply refers to the number of threads executed simultaneously in a thread block. Currently the warp size is 32, but this might change in future devices.
While grid and block sizes define logical threads, it's up to the device to assign the threads to the physical cores. Since the warp size is 32, even if a block
contains only a single thread, we still have 32 concurrent threads running, out of which only one is active. Therefore from efficiency point of view, the number of threads
per block should be a multiple of 32, i.e. 32, 64, 96 etc.

## Execution Context

The execution context of a warp consists of the following resources:

* program counters
* registers
* shared memory

The execution context of each warp being processed by a SM is maintained during the complete lifetime of the warp. Therefore, there is no penalty switching
from one execution context to another. Since the execution context is kept during the complete lifetime of the warp, the number of resources, such as registers and shared memory, 
limits the maximum number of active warps.

## Active Warps

A warp is active, if resources have been allocated to it. Active warps can either be in one of the following states:

* `selected` : actively executing warp
* `stalled` : not ready for execution
* `eligible` : ready for execution, but not executing currently

At every instruction, the warp-scheduler selects a warp that has threads ready to execute the next command.

## Warp Divergence

Inside the kernel code we can have have control structures, such as `if-then` clauses. Warp divergence happens when inside the same warp, due to control structures,
different threads execute different code, and this leads to worse branch efficiency. CUDA programming model follows a so called SIMT (Single Instruction Multiple Threads)
paradigm. This is due to the fact that the hardware has been optimized to run the same instructions in the threads.

## Warp Divergence Example

In the following we have two examples of a CUDA-kernel with `if-then` control structure inside the kernel. The first example does not have warp divergence, since the
`if-then` clause causes different code to be executed on a warp boundary, whereas the second example has warp divergece. Compiler tries to optimize the code in order
to avoid warp devergences, and other sub-optimal code. 

```
__global__
void no_warp_divergence()
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gid / 32;
    unsigned int a = 0, b = 0;

    if(warp_id % 2 == 0)
    {
        a++;
    }else{
        b++;
    }
}
```

In the example above, threads `0..31` execute the `if-part`, and threads `32..63` execute the `else-part`.

Following code example leads to warp divergence when built with the `-G` debug option which also turns off all optimizations of the device code.

```
__global__
void warp_divergence()
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int a = 0, b = 0;

    // Even threads execute the if-part and odd threads execute the else-part
    if(gid % 2 == 0)
    {
        a++;
    }else{
        b++;
    }
}
```

In the example above, threads `0, 2, 4, ...` (even) execute the if-part, and threads `1, 3, 5, ...` (odd) execute the else-part.

## How to Detect Warp Divergence

You can use the nvprof-profiler to analyze the CUDA-code for warp divergence, for example:

```
sudo nvprof --metrics branch_efficiency ./cuda_warp_divergence

```

In Linux, you need to execute the profiler with sudo-rights.

The report you should get for the cuda_warp_divergence.cu code (when built with the `-G` option), should look something similar to

```
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1070 (0)"
    Kernel: no_warp_divergence(void)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: warp_divergence(void)
          1                         branch_efficiency                         Branch Efficiency      80.00%      80.00%      80.00%
```

# Warp Divergence Sample Code

* `cuda_warp_divergence`. An example of warp divergence.
