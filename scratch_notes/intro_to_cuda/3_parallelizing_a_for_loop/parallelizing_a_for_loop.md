# Parallelizing a For Loop

## Kernel Definition
```
__global__ void kernel(int *d_out, int *d_in) {
    // Perform this operation for every thread
    d_out[0] = d_in[0]
}
```
- `__global__` is a "Declaration Specifier" that tells the compiler that the function is supposed to be called from the Host and run on the device
- Kernels that are called from the host never return a value (void return type). 
- Variables operated on in the kernel need to be passed by reference
- When a kernel is launched, the operations in the kernel body are executed for every thread that executes that kernel

## Thread Index
- Each Thread needs to know which position they are in within a block
- Accessible via the `threadIdx` variable
- Thread blocks can have as many as 3-dimensions, hence `threadIdx.x`, `threadIdx.y`, `threadIdx.z`

## Parallelize for loop
CPU program
```
void increment_cpu(int *a, int N) {
    for (i=0; i<N; i++)
    a[i] = a[i] + 1;
}

int main(void){
    int a[N] = // ...

    // Call Function
    increment_cpu(a, N);
    // ...
    return 0;
}
```

CUDA program
```
// Kernel Definition
__global__ void increment_gpu(int *a, int N) {
    int i = threadIdx.x;
    if (i < N)
        a[i] = a[i] + 1;
}

int main(void) {
    int h_a[N] = // ... 

    // Allocate arrays in Device memory
    int* d_a; cudaMalloc( (void**)&d_a, N * sizeof(int) );

    // Copy memory from Host to Device
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    // Block and Grid dimensions
    dim3 grid_size(1); dim3 block_size(N);

    // Launch Kernel
    increment_gpu<<<grid_size, block_size>>>(d_a, N);

    // Copy memory from Device to Host
    cudaMemcpy(h_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);

    // De-allocate memory
    cudaFree(d_a);

    // ...
    return 0;
}
```
