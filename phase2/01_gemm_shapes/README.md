# Exercise 1 -- GEMM across shapes and dtypes

```bash
python 01_gemm_shapes/gemm_bench.py
```

## What to implement

Two TODOs in `gemm_bench.py`:

- `arithmetic_intensity(m, k, n, dtype)` -- FLOP/byte for one shape, same
  definition `matmul.cu` printed for you in Phase 1.
- `bench_gemm(m, k, n, dtype)` -- allocate `x @ w` at that shape/dtype and time
  it with the shared `benchmark()` util from `phase2/common/bench.py`. That
  util is given, complete, unchanged from Phase 1 -- do not re-derive
  CUDA-event timing.

## Predict first

`SHAPES` sweeps `M` from 4096 down to 1, holding `K = N = 4096` fixed --
prefill-like down to decode-like, exactly the comparison the `matmul.cu`
README asked you to reason about by hand for `x: [1, 4096]`.

1. Which shape reaches the highest GFLOP/s? Which reaches the lowest?
2. At `M=1`, does fp16/bf16 beat fp32 by roughly the same factor as at
   `M=4096`? More? Less? Commit to a guess before running.
3. Compute arithmetic intensity by hand for `M=1` and `M=4096`. Which one is
   memory-bound by the roofline model, and which is compute-bound?

## Then answer

- Why does a GEMM's *shape*, not just its total FLOP count, determine whether
  it's compute- or memory-bound? (`M=1` and `M=4096` can have wildly different
  GFLOP/s from the same hardware doing the same kind of arithmetic.)
- At `M=1`, the weight matrix `w` (`[4096, 4096]`, up to 64 MB) has to be read
  from DRAM once to compute one row of output. At `M=4096`, that same weight
  read is amortized across 4096 rows. Tie this back to why reduced precision
  (fp16/bf16) helps a *memory-bound* GEMM -- is it Tensor Cores doing the work
  faster, or something else?
- `torch.matmul` on fp32 CUDA tensors dispatches straight to cuBLAS -- the
  exact library you benchmarked against in `phase1/04_matmul`. Does the
  `M=4096` fp32 row here roughly match the cuBLAS GFLOP/s you measured there,
  given it's the same GPU?

## Where this shows up later

This *is* the prefill/decode distinction from Phase 3, stated one phase early.
Prefill runs a large-`M` GEMM against the weights (compute-bound, good GPU
utilization). Decode runs `M = batch_size` (often 1, for a single interactive
request) against the exact same weights (memory-bound, poor utilization no
matter how fast the math units are) -- which is why decode throughput scales
with *batch size*, not with how powerful the GPU's compute is, and why
continuous batching exists.
