# Synchronization

## Threads Execute in Parallel
- Threads can read a result before another thread writes to that address
    - race condition

## Thread Synchronization via Explcit Barrier
- A point in a program where all threads that are executing a kernel within a block will halt execution when they reach this barrier point
- Barriers can be implemented using `syncthreads()`

## `synchthreads()` Example
- Shift contents of an array to the left by 1 element


```
// Broken code
__global__ void kernel(int *a) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < 3)
        a[i] = a[i+1];
}

// Good code
__global__ void kernel(int *a) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < 3) {
        int temp = a[i+1];
        __syncthreads(); // Ensure all cells are read before we start writing
        a[i] = temp;
        __syncthreads(); // Just to be safe. Ensure that all writes occur before any reads from a future grid
    }
}
```

## Implicit Barriers between Kernels
```
int main(void) { // Host Code
    // Do sequential stuff

    // Launch Kernel
    kernel_0<<<grid_sz0, blk_sz0>>>(...);

    // Force Host to wait on the completion of the Kernel
    cudaDeviceSynchronize();

    // Copy data from Device to Host
    cudaMemcpy(...);

    // Do more sequential stuff

    return 0;    
}
```
- Host and device operate asynchronously, unless the host code is explicitely specified to wait for the device
- Execution of consecutive Kernels **do** operation synchronously (implicit barrier)
    - 2 consecutive Kernel launches will guarantee that the 2nd kernel will not be executed on the device until the first kernel has completed

## Cuda in a nutshell
- Hierarchy of Computations
    - Threads
    - Blocks
    - Grids

- Corresponding Memory Spaces
    - Local
    - Shared
    - Global

- Synchronization Primitives
    - Implicit Barriers
    - Thread synchronization


