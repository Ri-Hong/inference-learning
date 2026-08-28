# Exercise 2 — Matrix transpose

```bash
make transpose && ./bin/transpose            # 4096 x 4096
./bin/transpose 8192
```

The cleanest experiment in Phase 1, because transpose does **no arithmetic at
all**. Every kernel moves exactly the same bytes and produces exactly the same
output. Any difference in runtime is purely the memory access pattern.

## What to implement

Five TODOs in `transpose.cu`:

| kernel | reads | writes | shared memory |
|---|---|---|---|
| `copyKernel` | coalesced | coalesced | none — this is the ceiling |
| `naiveTranspose` | one side is strided | | none |
| `tiledTranspose` | coalesced | coalesced | `tile[32][32]` |
| `tiledPaddedTranspose` | coalesced | coalesced | `tile[32][33]` |

plus `transposeCpu` as the correctness reference.

`copyKernel` is not a transpose. It exists to establish the bandwidth ceiling:
identical volume with a perfect access pattern. Judge every other row against
it, not against each other.

## Predict first

1. How far below `copy` will `naive` land?
2. Will `tiled` beat `naive`? By how much?
3. How much does one padding column buy you — 0%, 5%, 50%?
4. Will any kernel match `copy`?

## Then answer

- Why is `naive` slow on one side but not the other, when both are strided from
  *some* point of view?
- `BLOCK_ROWS` is 8, not 32, so each thread handles 4 elements. Why is a 32x8
  block with a loop better than a 32x32 block without one? Change it and see.
- Work out, from the banking rules, exactly which bank `tile[r][c]` lands in for
  both the padded and unpadded versions. Then explain your measured difference.
- Would padding by 2 work? By 32? Predict, then try.
- Why does no kernel match `copy`? Name the specific costs.
- Shrink to `./bin/transpose 512`. What happens to the gap between `naive` and
  `tiled`, and why?

## Where this shows up later

The transposed-access problem is everywhere in attention. Q, K and V get
reshaped and permuted between `[batch, seq, heads, dim]` layouts, and a naive
permute is exactly this kernel. It is one reason fused attention kernels
(FlashAttention, and the backends in Phase 6) avoid materializing transposed
intermediates at all.
