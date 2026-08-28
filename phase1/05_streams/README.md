# Exercise 5 — Launch overhead and streams

```bash
make streams && ./bin/streams
```

Not in the original exercise list, but "kernel launch overhead" and "streams" are
both Phase 1 topics, and both matter by Phase 6.

## What to implement

Two TODOs in `streams.cu`. Parts 1 and 2 are given complete — they are
measurements, not exercises, and they set up the number Part 3 depends on.

**Part 1 (given)** times an empty kernel — no memory traffic, no arithmetic — so
whatever it measures is pure overhead. Then the same launch with a
`cudaDeviceSynchronize()` after each, to price the synchronize separately.

**Part 2 (given)** does a fixed amount of work split into 1, 16, 256, 4096, and
65536 kernel launches. Same total elements, same total arithmetic.

**Part 3 (yours)** — `busyKernel`, plus the multi-stream pipeline that overlaps
H2D copy, compute and D2H copy across 2, 4 and 8 streams against a serial
baseline.

## Predict first

1. How many microseconds does an empty kernel launch cost?
2. At what chunk count does launch overhead start to dominate Part 2?
3. What is the theoretical maximum speedup from perfect 3-way overlap?
4. Will 8 streams beat 4? Beat 2?

## Then answer

- Why does a synchronize cost several times more than a launch?
- Read the `us / launch` column in Part 2 downward. Where does it stop being a
  measurement of the kernel, and what is it measuring after that point?
- Why must the H2D, kernel and D2H for a given chunk all go in the *same* stream?
- `cudaMallocHost` is not optional here. What happens if you use ordinary
  `malloc`'d memory with `cudaMemcpyAsync`, and why is that a nasty bug to find?
- The default stream (stream 0) has special synchronizing behaviour with respect
  to other streams. What is it, and how would it break Part 3 if one operation
  accidentally landed there?
- Two streams, each with a kernel that alone occupies the whole GPU. Do they
  overlap? What actually determines whether concurrent kernels run concurrently?

## The experiment that makes the point

Once Part 3 works with `rounds = 200`, set `rounds = 5` and rerun. The speedup
should change character. Explain what changed and what caps the speedup in each
regime.

## Where this shows up later

This is the setup for **CUDA graphs**, which you meet inside vLLM in Phase 6.

A single decode step of a 32-layer model launches on the order of hundreds of
small kernels — per-layer QKV projection, attention, output projection, two or
three MLP kernels, two norms, plus elementwise work. At batch size 1 each may
take only a few microseconds, which puts them squarely in the regime Part 2
measures.

Once you have your Part 1 number, multiply it by the number of kernels in a
decode step and compare against a realistic per-token latency. That product is
why CUDA graphs exist.
