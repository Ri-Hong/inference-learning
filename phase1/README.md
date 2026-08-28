# Phase 1 — CUDA Fundamentals

**Milestone:** explain why two kernels doing the same arithmetic can have very
different performance.

Every exercise is built around that one question. Three of the five contain a
pair of kernels that compute *byte-for-byte identical results* and differ only
in which thread touches which address.

## What this is

A scaffold, not a tutorial. Each exercise gives you:

- a complete harness — allocation, transfers, CUDA-event timing with warmup,
  correctness checking against a reference, and a table reporting GB/s or
  GFLOP/s against the device's own peak
- kernel stubs with a spec and the things worth getting wrong called out
- a README with what to predict before running and what to answer after

It compiles and runs from the first `make`. Every correctness column reads `NO`
until you write the kernels.

## Setup

You need a remote NVIDIA GPU. See [`SETUP.md`](SETUP.md) for RunPod specifics
and the toolchain gotchas.

```bash
cd /workspace/inference-learning/phase1
./setup.sh              # checks nvcc/driver/torch, then builds
```

Day to day:

```bash
make                    # -arch=native
make ARCH=sm_89         # or name the target explicitly
make vector_add         # one target
make run                # build and run all five in order
```

## The five exercises

Do them in order — each assumes the measurement discipline of the last.

| | Exercise | What it isolates |
|---|---|---|
| 1 | [`01_vector_add`](01_vector_add/) | transfers, event timing, block-size sweep, occupancy, coalescing |
| 2 | [`02_transpose`](02_transpose/) | coalescing and shared-memory bank conflicts, with zero arithmetic |
| 3 | [`03_reduction`](03_reduction/) | synchronization, warp divergence, warp shuffles |
| 4 | [`04_matmul`](04_matmul/) | data reuse, arithmetic intensity, and why cuBLAS wins |
| 5 | [`05_streams`](05_streams/) | launch overhead and overlapping transfer with compute |

Exercises 1–4 are the ones the lesson plan asks for. Exercise 5 covers the
"streams" and "kernel launch overhead" topics, which the plan lists but gives no
exercise for, and which matter in Phase 6 when you meet CUDA graphs.

## How to work through one

1. Read the exercise README. **Write your predictions down first**, in
   `results/`, before you have any numbers. A prediction you got wrong is the
   only reliable signal about which part of your model is broken; a prediction
   you make after seeing the answer is worth nothing.
2. Implement the TODOs. Get `correct: yes` before you look at a single timing.
3. Run it. Record the output and the GPU it came from.
4. Explain every gap between prediction and result. If you cannot explain one,
   that is the interesting part — write down the question rather than moving on.

Copy [`results/RESULTS_TEMPLATE.md`](results/RESULTS_TEMPLATE.md) per session.

## What the harness gives you

`common/cuda_utils.cuh`:

| | |
|---|---|
| `CUDA_CHECK(call)` | aborts with file:line on any error |
| `CUDA_CHECK_KERNEL()` | `cudaGetLastError()` + `cudaDeviceSynchronize()`; kernel launches report errors asynchronously, so both are needed |
| `GpuTimer` | CUDA events |
| `timeGpuMs(f, warmup, iters)` | warms up, then averages — use this, not wall clock |
| `allClose` / `allCloseQuiet` | verification; the loud one prints the first mismatch |
| `peakBandwidthGBs(prop)` | the denominator for every bandwidth number |
| `printDeviceInfo()` | SMs, shared memory, bus width, peak bandwidth |

Two things it is doing on your behalf that are worth understanding rather than
trusting:

**Warmup.** The first launch in a process pays for CUDA context creation, module
loading, and JIT — sometimes hundreds of milliseconds. Same reason the first
PyTorch op of a session is slow, which Phase 2 asks about directly.

**Events, not wall clock.** Kernel launches are asynchronous; the call returns
long before the GPU has done anything. Timing a launch with `std::chrono`
measures how fast you can *ask* for work. `04_matmul/torch_matmul_bench.py`
demonstrates the mistake side by side with the truth.

## Reading

Read these *after* you have your own numbers to argue with, not before.

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
  — chapters 2 (model), 5.3 (memory throughput), and the appendix on compute
  capabilities
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  — the memory optimization chapter is the one that matters here
- Mark Harris, *Optimizing Parallel Reduction in CUDA* — exercise 3 is this
  paper, one version at a time
- NVIDIA devblog, *An Efficient Matrix Transpose in CUDA C/C++* — exercise 2
- Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance
  Model* — the frame for the whole curriculum

## Concepts the exercises assume

Names to look up, not explanations. If you cannot say what one means by the end
of the phase, that is the gap.

- host / device, PCIe vs on-card DRAM
- grid, block, thread; why blocks must be independent
- warp, SIMT, warp divergence
- coalescing, memory transactions
- registers / shared memory / L2 / global, and their latencies
- shared memory banks, bank conflicts, padding
- `__syncthreads()`, `__syncwarp()`, `__shfl_down_sync()`
- occupancy, and what limits it
- effective bandwidth vs peak bandwidth
- arithmetic intensity, roofline, memory-bound vs compute-bound, ridge point
- kernel launch overhead, streams, pinned memory

## Milestone check

You are done with Phase 1 when you can answer these from your own numbers,
without looking anything up:

1. Two kernels perform the same additions on the same arrays and one is 8x
   slower. What is the most likely cause, and how would you confirm it?
2. Why does the padded shared-memory transpose beat the unpadded one, when both
   move identical bytes through identical instructions?
3. Why does the sum reduction need `__syncthreads()` inside the loop, and why
   does the warp-shuffle version not?
4. Your kernel hits 30 GB/s on a GPU with 320 GB/s peak. Name three plausible
   causes, ordered by likelihood.
5. Why does a hand-written tiled GEMM lose to cuBLAS, given that both are tiled?
6. When does raising occupancy help, and when does it do nothing at all?
