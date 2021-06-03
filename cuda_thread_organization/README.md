# Thread Organization and Vector/Matrix Element Access

Here we have some examples regarding accessing vector/matrix elements from CUDA threads. We use the following
CUDA function call notation for blocks and threads `<grid, block>`, where `grid` defines the size of the processing
grid, and `block` defines the number of actual threads in a `grid`.


* cuda_thread_organization. Prints out `blockIdx` and `threadIdx` ids.
* cuda_thread_1D_indexing. Vector element access example.
* cuda_thread_2D_indexing. 2D Matrix element access example.
* cuda_thread_3D_indexing. 3D Matrix element access example.

