# Indexing Threads within Grids and Blocks

- to retrieve the index of a thread within a block, use `threadIdx`
- to retrieve the dimensions of a block, use `blockDim`

- to retrieve the index of a block within a grid, use `blockIdx`
- to retrieve the dimensions of a grid, use `gridDim`

- each one of those variables has a corresponding `x,y,z` dimension
    - i.e. `gridDim.x`, `gridDim.y`, `gridDim.z`


## Indexing Within a Grid
- `threadIdx` is only unique within its own Thread Block
- to determine the unique Grid index of a Thread:
    - `i = threadIdx.x + blockIdx.x * blockDim.x`

