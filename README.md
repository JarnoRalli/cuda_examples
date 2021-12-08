# CUDA Examples

This repository contains an explanation of the CUDA programming model and some examples demonstrating CUDA programming.
CMake is used to build the examples.

# CUDA Programming Model

The CUDA programming model provides an abstraction of the GPU architecture and acts as a bridge between an application
and the implementation on GPU hardware. The model has three key abstractions:

1. a hierarchy of thread groups
2. shared memories
3. synchronization

The programming model consists of two different levels of parallelism: a fine-grained data- and thread parallelism embedded
into a coarse-grained data- and task parallelism. This is achieved by combining arrays of threads, each executing the same code,
into thread blocks, and combining several blocks into grids. Each thread block is executed by one streaming multiprocessor (SM)
and cannot be migrated to other SMs in the GPU. Since a block runs in a single SM, the threads in the block can communicate and
synchronize with each other. One SM can run several concurrent CUDA blocks depending on the resources needed
by the CUDA blocks. The CUDA programming model guides the programmer to partition the problem into sub-problems
that are solved independently, in parallel, by these blocks of threads. Following figure shows the mapping between the
abstract programming model (on the left) and the hardware (on the right).

<p align="center">
<img src="./images/cuda_blocks_and_grids.png" width="500">
</p>

As it can be seen from the above figure, a thread maps to a CUDA core, a block of threads maps to a CUDA streaming multiprocessor (SM), and
CUDA grid maps to a CUDA device.

## Warp Size

CUDA programming model follows a SIMT (Single Instruction, Multiple Threads) paradigm. Warp refers to a number of threads being executed
simultaneously in a thread block. Remember that a thread block is executed by one streaming processor (SM). Currently the warp size is 32.
For more information regarding warps, and warp divergency, take a look at [CUDA warps](./cuda_warp/README.md)

## Execution of a CUDA Program

A CUDA enabled system consists of:

* a host (CPU)
* the device(s) (GPUs)

Execution of the CUDA programming consists of three steps:

1. Copy input data from the host to the device memory (host-to-device transfer)
2. Load the GPU program and execute it
3. Copy the results from the device back to the host memory (device-to-host transfer)

In an NVIDIA GPU, the basic unit of execution is a warp. A warp is a collection of threads that are executed in parallel by an SM. Multiple
warps can be executed on an SM simultaneously.

## CUDA Device Occupancy

Occupancy refers to the ratio of active warps with respect to the maximum number of warps per SM. Since each thread uses registers and shared memory from the SM,
the number active warps is limited by the resources that each SM has. In order to calculate the occupancy, NVidia provides an Excel chart called
`CUDA_Occupancy_Calculator.xls`. This chart allows to analyze device occupancy with respect to different numbers of threads per block, thread register and 
shared memory usage etc. Following information is needed for the chart:

* Compute capability. The chart contains all the information, like the maximum number of warps per SM, based on the compute capability of the device
* Threads per block
* Number of registers used per thread
* Shared memory used per thread

Probably the easiest way to get the exact number of used registers and shared memory per thread is to build the kernel with a flag that outputs the information.
For more information, take a look at [resource usage per thread](./cuda_resource_usage/README.md)

# Data Mapping to a GPU

CUDA defines built-in variables that define the number of threads, blocks and grids. Using these variables data can be mapped
to the threads for execution. These variables are:

* `threadIdx` with x-, y- and z- coordinates. For example, `threadIdx.x`
* `blockDim` and `blockIdx` with x-, y- and z- coordinates
* `gridDim` with x-, y- and z- coordinates

Depending on the problem at hand, there exists many different ways of mapping the data to the threads. In the following example we show an example of one particular way of mapping a 2D matrix to threads.
There is nothing special about this mapping, it could be mapped in any other way.

## Example of Mapping 2D Data to a GPU

Using the built-in CUDA variables, we can map a 2D matrix to the threads for execution. In the following example we have a 6x6 2D
matrix. In this case we have chosen the block dimension to be (3, 3) and the grid dimension to be (2, 2). The matrix is a contiguous array in the memory, so that [0...35] define
the offsets from the beginning of the array, logically, however, we have a 6x6 matrix. Following figure shows how the matrix is mapped to the threads.

<p align="center">
<img src="./images/cuda_2d_element_access.png" width="500">
</p>

We can define the following auxiliary variables

* `row_stride = blockDim.x * gridDim.x`
* `block_offset = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * row_stride`

, where `row_stride` is the interleave between matrix rows and `block_offset` points to the first data element assigned to each block. For example,
in the above case `row_stride` is 6 and the `block_offset` would be the following:

* block(0, 0): `0 * 3 + 0 * 3 * 6 = 0`
* block(1, 0): `1 * 3 + 0 * 3 * 6 = 3`
* block(0, 1): `0 * 3 + 1 * 3 * 6 = 18`
* block(1, 1): `1 * 3 + 1 * 3 * 6 = 21`

# Examples


* [CUDA hello world example](./cuda_hello_world/README.md)
* [CUDA device property examples](./cuda_device_properties/README.md)
* [CUDA thread organization and element access examples](./cuda_thread_organization/README.md)
* [CUDA add examples](./cuda_add/README.md)
* [CUDA warp divergence example](./cuda_warp/README.md)
* [CUDA resource usage example](./cuda_resource_usage/README.md)
