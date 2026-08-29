# Warps and SIMT

## Warps

- Within a thread block, threads are organized into groups of 32 called **warps**
- Suppose you launch a block with 128 threads:
```
kernel<<<1,128>>>();
```
- Then you have 128 CUDA threads and the GPU will group them into 4 warps
```
Warp 0: threads 0-31
Warp 1: threads 32-63
Warp 2: threads 64-95
Warp 3: threads 96-127
```

- Each position inside the warp is called a lane, numbered 0-31

## SIMT
- SIMT = Single Instruction, Multiple Threads
- A warp generally advances through the kernel together
    - For a vector add, all 32 threads in the warp do the same add operation

### Branching Example 1
```
if (threadIdx.x % 2 == 0) {
    doSomething();
}
```
- The even threads will execute the function, but odd threads wont
- CUDA executes the `if` body while masking off the threads that shouldn't participate
```
Lane:    0 1 2 3 4 5 6 7 ...
         ✓ X ✓ X ✓ X ✓ X
```
- The odd threads sit idle while `doSomething()` runs
- This is known as warp divergence

### Branching Example 2
```
if (threadIdx.x < 16) {
    functionA();
} else {
    functionB();
}
```
- Then
```
threads  0–15 → functionA()
threads 16–31 → functionB()
```
- The execution becomes
```
Step 1:
threads 0–15:   execute functionA
threads 16–31:  inactive

Step 2:
threads 0–15:   inactive
threads 16–31:  execute functionB
```
- Instead of all 32 lanes doing useful work together, only half are active at a time
- GPU utilization is highest when threads ina. warp follow the same control-flow path

### Branching Example 3
```
if (blockIdx.x == 0) {
    ...
}
```
- This example is fine
- No warp divergence because every thread in the warp will either run or not run together

## Block sizing
- blocks should generally be multiples of 32
- Suppose you launch `kernel<<<1,40>>>();`
- Then CUDA needs:
```
Warp 0: threads 0–31      → 32 active lanes
Warp 1: threads 32–39     → 8 active lanes
                              24 unused lanes
```

## SIMT vs SIMD
### SIMD
- A SIMD might look like
```
ADD  [x0 x1 x2 x3]
     [y0 y1 y2 y3]
```
- Each instruction operates on a fixed-width vector

### SIMT
- CUDA SIMT exposes
```
Thread 0
Thread 1
Thread 2
...
Thread 31
```
- Each thread is an independent program that can have its own control flow

