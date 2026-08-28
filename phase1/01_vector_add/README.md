# Exercise 1 — Vector addition

```bash
make vector_add && ./bin/vector_add          # 16M elements
./bin/vector_add 1000000                     # other sizes
```

## What to implement

Four TODOs in `vector_add.cu`:

| | | |
|---|---|---|
| 1 | `vectorAddCpu` | baseline and correctness reference |
| 2 | `vectorAddKernel` | one thread per element |
| 3 | `vectorAddGridStride` | grid sized to the GPU, not to the data |
| 4 | `vectorAddScattered` | **the same additions on the same operands**, with consecutive threads assigned elements far apart |

The harness times H2D and D2H separately from compute, sweeps block sizes
32…1024, and prints occupancy per block size.

## Predict first

Write these in `results/` before you run anything:

1. Which block size will be fastest, and by how much?
2. What fraction of peak DRAM bandwidth will the best version reach?
3. How much slower will the scattered version be? Commit to a number.
4. Which is larger: total transfer time, or kernel time?

## Then answer

- Why is the multiplier in `effectiveBandwidthGBs` 3 and not 2? Would you count
  it differently if `c` were write-only in a way the hardware knew about?
- Why is the CPU version's effective bandwidth so much lower than the GPU's,
  given that CPU DRAM bandwidth is only about 10x worse, not 100x?
- Occupancy at block size 1024 may be *lower* than at 256. Work out why from the
  numbers `printDeviceInfo()` gives you.
- The grid-stride version launches far fewer blocks. Is it slower? Why not?
- Take your coalesced-vs-scattered ratio. Derive it from first principles —
  memory transaction size, warp width, and the stride. Does your arithmetic
  match the measurement? If not, what else is going on?
- If transfers cost 25x more than the add, when is it ever worth doing a vector
  add on the GPU at all?

## Variations worth running

```bash
./bin/vector_add 1000000        # does the coalescing penalty scale with size?
./bin/vector_add 67108864
```

Change `SCATTER` from 32 to 4, 8, and 128. Plot slowdown against scatter
distance and explain the shape — particularly where it stops getting worse.

## Where this shows up later

The last prediction question is the whole reason serving engines exist. A single
small operation is never worth a round trip; value comes from keeping data
resident and doing hundreds of operations on it. That is what a forward pass is,
and why vLLM keeps the KV cache in VRAM rather than paging it to host memory.
