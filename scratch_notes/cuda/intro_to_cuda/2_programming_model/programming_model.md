# Programming Models

## Program Flow
- CUDA code runs serially on CPU
- When it reaches a portion of parallelizable code, it launches the kernel
- The kernel runs in parallel on GPU
- Host code doesn't wait on kernel's completion unless explicitely told to do so

```
int main(void) { // Host Code
    // Do sequential stuff

    // Launch Kernel
    kernel_0<<<grid_sz0, blk_sz0>>>(...);

    // Do more sequential stuff

    // Launch Kernel
    kernel_1<<<grid_sz1, blk_sz1>>>(...);

    return 0;
}
```

## Kernel Launch Syntax
```
// BLock and Grid dimensions
dim3 grid_size(x,y,z);
dim3 block_size(x,y,z);

// Launch Kernel
kernelName<<< grid_size, block_size >>>(...);
```
- `dim3` is a CUDA data structure
    - default values are `(1,1,1)`


## Memory allocation
- Host and Device are different componenets and have separate memory
- To launch a kernel, we need to:
    - allocate memory on the device
    - copy data from host -> device
    - launch kernel
    - copy data from device to host

- In C:
    - Allocate memory: `malloc(...);`
    - Deallocate memory: `free(...);`
- In CUDA:
    - `cudaMalloc( LOCATION, SIZE );`
    - 1st argument: Memory address in Device to allocate memory
    - 2nd argument: Number of bytes to allocate

- To copy data to and from Host and device
    - `cudaMemcpy(dst, src, numBytes, direction);`
    - `dst` is a ptr to the address we are copying into
    - `src` is a ptr to the address we are copying from
    - `numBytes = N*sizeof(type)`
    - `direction` = `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`

## Example program

```
int main(void) {
    // Declare variables
    // variables that start with h_ live on host. d_ lives on device
    int *h_c, *d_c;

    // Allocate memory on the device
    cudaMalloc((void**)&d_c, sizeof(int));

    cudaMemcpy(d_c, h_c, sizeof(int), cudaMemcpyHostToDevice);

    // Configuration Parameters
    dim3 grid_size(1);
    dim3 block_size(1);

    // Launch the Kernel
    kernel<<<grid_size, block_size>>>(...);

    // Copy data back to host
    cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // De-allocate memory
    cudaFree(d_c);
}
```

