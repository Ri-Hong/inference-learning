# Exercise 2 -- softmax, layer_norm, and fusion

```bash
python 02_normalization_ops/ops_bench.py
```

## What to implement

Three TODOs in `ops_bench.py`:

- `achieved_gbs(numel, dtype, ms)` -- bytes moved / time. One read, one write,
  no reuse possible -- these ops are memory-bound by construction, unlike the
  GEMM in Exercise 1.
- `bench_ops(batch, seq, hidden, dtype)` -- time `torch.softmax` and
  `F.layer_norm` at one shape/dtype using the shared `benchmark()` util.
- `bench_fusion(batch, seq, hidden, dtype)` (stretch) -- compare calling
  softmax then layer_norm as two ops vs. a `torch.compile`'d fused version.

## Predict first

1. Given these are memory-bound, does fp16 roughly halve the ms compared to
   fp32 at the same shape? Why would that be true here specifically, when it
   *wasn't* uniformly true for the compute-bound GEMM shapes in Exercise 1?
2. Compare the smallest shape `(1, 512, 4096)` against the largest
   `(32, 4096, 4096)` -- both process the same *kind* of data, just more of
   it. Do you expect GB/s to be roughly constant across shapes, or to change?
   What would make it change?
3. For the fusion TODO: at which shape do you expect fusing softmax +
   layer_norm into one compiled kernel to save more time, the small shape or
   the large one?

## Then answer

- Look up your GPU's peak DRAM bandwidth (printed by `phase1`'s
  `printDeviceInfo()`, or `nvidia-smi -q -d MEMORY`, or the spec sheet). What
  fraction of peak are you actually achieving? If it's well under 100%, is
  that a bug, or expected? (Hint: what did Exercise 1 of Phase 1,
  `01_vector_add`, teach you about achievable vs. peak bandwidth at different
  transfer sizes?)
- Two separate kernel launches (softmax, then layer_norm) means the
  intermediate result gets written to DRAM and immediately read back by the
  next kernel. What does fusing them into one kernel eliminate, precisely?
  Is it the launch overhead from Phase 1 Exercise 5, the redundant DRAM
  round-trip, or both?
- Your prediction in question 3, above, was probably "the small shape
  benefits more." Explain why in terms of what's *fixed* (launch overhead)
  vs. what *scales* (bytes moved) as the shape grows -- this is the same
  fixed-cost-vs-real-work tension from Phase 1 Exercise 5's launch-count
  table, applied to fusion instead of chunking.

## Where this shows up later

Real transformer inference chains dozens of these small, memory-bound ops per
layer (norms, activations, residual adds) between the large GEMMs. Fusing them
is exactly what makes libraries like FlashAttention and fused-kernel
implementations of RMSNorm/SwiGLU matter in practice -- Phase 9 (Triton) has
you write some of these fused kernels yourself.
