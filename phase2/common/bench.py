"""Shared GPU-timing utility for Phase 2.

Identical to the `benchmark()` / `naive_timing_trap()` pair already given to you
complete in `phase1/04_matmul/torch_matmul_bench.py`. You already worked out
*why* this is the correct way to time a GPU op back in Phase 1 (CUDA launches
are asynchronous; wall clock around an unsynchronized call measures how fast
you can enqueue work, not how long the GPU took). Nothing new to implement
here -- it's given so every exercise in this phase measures the same way.
"""

import time

import torch


def benchmark(fn, warmup=5, iters=20):
    """Time a GPU op correctly: warm up, then average over synchronized runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def naive_timing_trap(fn, iters=3):
    """The wrong way. Keep few iterations -- more and the queue backs up and
    the host blocks anyway, which accidentally makes the wrong method look
    right."""
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000 / iters
