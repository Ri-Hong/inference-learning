# Exercise 4 — Matrix multiplication

```bash
make matmul && ./bin/matmul                  # 2048 x 2048 x 2048
./bin/matmul 1024
python 04_matmul/torch_matmul_bench.py       # run right after, same session
```

The first **compute-bound** kernel in Phase 1. Exercises 1–3 were all limited by
DRAM bandwidth. A GEMM does O(N³) arithmetic on O(N²) data, so with enough reuse
it can actually saturate the math units.

## What to implement

Three TODOs in `matmul.cu`. cuBLAS is wired up as both the correctness reference
and the ceiling.

| kernel | reuse strategy | global reads of A |
|---|---|---|
| `matmulNaive` | none (L2 only) | N times |
| `matmulNaiveSwapped` | none, and **one line different** | N times |
| `matmulTiled` | 32x32 shared-memory tiles | N/32 times |
| cuBLAS | register blocking, vectorized loads, tuned tiles, Tensor Cores | — |

`matmulNaive` and `matmulNaiveSwapped` differ only in whether `threadIdx.x`
indexes the row or the column. Identical arithmetic, identical output. This is
the milestone question compressed into a single diff — spend time on it.

## Predict first

1. What fraction of the GPU's peak fp32 throughput will `naive` reach?
2. How much does tiling buy over naive?
3. How much faster is cuBLAS than your tiled kernel? Commit to a factor.
4. Which naive variant is faster, and by how much?

## Then answer

- For a fixed `k`, write down the addresses the 32 threads of a warp read from A
  and from B, for each naive variant. Derive the ratio you measured.
- The tiled kernel has two `__syncthreads()`. Delete each one separately and
  describe precisely what goes wrong — not just "it's wrong", but which read
  races which write.
- Why does tiling's advantage over naive grow with N?
- At N=128 the naive kernel may beat the tiled one. Predict, then check, then
  explain.
- **cuBLAS will still beat your tiled kernel, probably by several times.** Work
  out where the gap comes from before reading anything. Hints, in the order you
  should consider them: how many output elements does each of your threads
  compute? how many bytes does one load instruction move? what are the math units
  doing while a tile is being staged?
- Arithmetic intensity for a 2048³ GEMM is printed at startup. Compute the same
  number by hand for `x @ W` where x is `[1, 4096]` and W is `[4096, 4096]` — one
  token of decode. Which side of the roofline does it fall on?

## The PyTorch comparison

`torch_matmul_bench.py` is provided complete — it is a reference, not an
exercise. `torch.matmul` on fp32 CUDA tensors *is* cuBLAS, so it should land on
essentially the same GFLOP/s as the cuBLAS row. If it does not, you have a
measurement problem, not a math problem.

It also sweeps tf32, fp16 and bf16 (Tensor Cores — Phase 2's subject), and
demonstrates the async-timing trap by timing the same matmul with a naive wall
clock and with CUDA events. Read `benchmark()` and `naive_timing_trap()` and make
sure you can say why they disagree.

## Where this shows up later

The last question is the foundation of Phase 3. Prefill multiplies a large
activation matrix by the weights and is compute-bound. Decode multiplies a
*single row* by those same weights, so the weights are read from DRAM once per
token and barely used. Same operation, same hardware, completely different
bottleneck — which is why continuous batching exists, why decode benefits from
batching far more than prefill does, and why prefill/decode disaggregation is a
live research area.
