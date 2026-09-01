# Phase 2 -- PyTorch on CUDA

**Goal:** connect the GPU execution model from Phase 1 to high-level PyTorch
ops. Same hardware, same concepts (asynchronous execution, coalescing,
arithmetic intensity, launch overhead) -- now observed through the framework
you'll actually use for the rest of this curriculum.

## What this is

Two exercises, same spirit as Phase 1: a scaffold, not a tutorial.

- `common/bench.py` -- the CUDA-event timing utility, given complete. It's
  identical to `benchmark()`/`naive_timing_trap()` from
  `phase1/04_matmul/torch_matmul_bench.py`, which you already worked through.
  Nothing new to build here; every exercise imports it so every measurement in
  this phase is done the same, correct way.
- Exercise stubs with `raise NotImplementedError("TODO N")` in place of the
  logic, a spec in the comment above each, and a README with what to predict
  before running and what to answer after.

## Setup

Same GPU as Phase 1. Just needs `torch` with CUDA support:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## The two exercises

| | Exercise | What it isolates |
|---|---|---|
| 1 | [`01_gemm_shapes`](01_gemm_shapes/) | GEMM shape (not just size) determines compute- vs. memory-bound; prefill vs. decode one phase early |
| 2 | [`02_normalization_ops`](02_normalization_ops/) | memory-bound ops, achieved vs. peak bandwidth, kernel fusion |

Do them in order -- Exercise 2's fusion question leans on Exercise 5's
launch-overhead lesson from Phase 1, and Exercise 1 leans directly on
`04_matmul`.

## How to work through one

Same discipline as Phase 1:

1. Read the exercise README. Write your predictions down before running
   anything.
2. Implement the TODOs.
3. Run it, record the numbers.
4. Explain every gap between prediction and result.

## Concepts the exercises assume

- everything from Phase 1's list, plus:
- asynchronous execution in a framework (not just raw CUDA calls)
- GEMM / cuBLAS dispatch from `torch.matmul`
- Tensor Cores, and which dtypes/matmul shapes actually use them
- FP32 / TF32 / FP16 / BF16 -- what changes numerically vs. what changes for
  throughput
- kernel fusion, and what it does and doesn't eliminate

## Milestone check

You are done with Phase 2 when you can answer these from your own numbers:

1. Why is the first invocation of a PyTorch op on the GPU often much slower
   than the rest?
2. Why do larger matmuls often utilize the GPU better than smaller ones?
3. Why do FP16/BF16 often beat FP32 -- and is the reason the same at every
   matmul shape you tested?
4. Why can small GPU ops be inefficient, and what does fusing them actually
   save?
