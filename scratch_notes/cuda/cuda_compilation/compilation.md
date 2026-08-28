# CUDA Compilation

## nvcc (Nvidia CUDA Compiler)
- nvcc is a compiler driver
- a `.cu` file mixes 2 languages into 1 file:
    - ordinary C++ that runs on the CPU
    - `__global__/__device__` kernel code that runs on GPU
- nvcc is able to compile the 2 parts separately:
    - ordinary C++ using a C++ compiler like g++
    - compiles the CUDA code into PTX using cicc (CUDA Internal Compiler for CUDA)
- it then links the host object code together with the compiled device code, embedding both into one executable - a "fat binary"
- Upon execution, the Host will launch the kernels on the Device

## PTX (Parallel Thread Execution)
- PTX is the output of cicc
- A textual, architecture-independent virtual instruction set for GPUs
- Stable acrosss GPU generations by design - meant to be a portable target
- Similar in spirit to Java bytecode or LLVM IR
    - a fixed instruction set that any future NVIDIA GPU's driver can JIT-compile at runtime if it doesn't find matching precompiled SASS

## SASS (Streaming ASSembler)
- SASS is the actual native instruction set that the GPU's hardware executes
- Every architecture generation (Ampere, Ada, Hopper, ...) has its own SASS instruction set
- ptxas takes in PTX code and outputs SASS for a specific `--arch=sm_XX` target

Flow:
```
.cu device code ----> PTX -----> SASS
                cicc       ptxas
```


