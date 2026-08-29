# Tiles

- a tile is a higher-level way of programming the GPU where you operate on a small multidimensional chunk of data as a whole instead of explicitely telling each CUDA thread what to do
- the compiler figures out how the threads in the block perform on that tile operation
- the block in the tile follows a single control flow, so there is no concept of warp divergence
- traditional CUDA might look like:
```
int i = blockIdx.x * blockDim.x + threadIdx.x;

C[i] = A[i] + B[i];
```
- where you explicitely reason about
    - threads
    - threadIdx
    - which element each thread owns
    - synchronization
    - warp behaviour
- with the tile abstraction, it looks more like
```
a = load(A, tile=(16,16))
b = load(B, tile=(16,16))

c = a + b

store(C, c)
```

## Tiles vs Thread Blocks
- A thread block is an execution concept
- A tile is a data concept
- One block can create and operate on many tiles, and those tiles can have different shapes and types
