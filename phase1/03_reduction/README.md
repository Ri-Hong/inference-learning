# Exercise 3 — Sum reduction

```bash
make reduction && ./bin/reduction            # 16M elements
```

The first exercise where threads must **cooperate**. Vector add and transpose let
every thread work alone; a sum requires combining partial results, which requires
communication, which requires synchronization.

## What to implement

Six TODOs in `reduction.cu`. Write them in order and measure after each — the
deltas are the lesson, not the final number.

| | change | problem it is meant to fix |
|---|---|---|
| v1 | `if (tid % (2*stride) == 0)` | — (the naive starting point) |
| v2 | active threads are contiguous | warp divergence |
| v3 | stride halves instead of doubling | shared-memory bank conflicts |
| v4 | each thread loads and adds 2 elements | half the threads idle after step 1 |
| v5 | grid-stride load + `__shfl_down_sync` | shared memory and barriers entirely |

The host driver — including the multi-pass plumbing that reduces an array down to
a single value — is written for you. Note that it does no host synchronization
inside the timed region, on purpose.

## Predict first

1. Rank the five by speed.
2. Which single change gives the biggest jump? By how much?
3. What fraction of peak bandwidth can a good reduction reach? (What does a
   reduction actually read and write?)
4. Will all five print the same sum?

## Then answer

- For v1, at stride = 128, which lanes of warp 0 are active? At stride = 1?
  Now explain the cost of a divergent branch in terms of what the warp actually
  issues.
- v2 fixes divergence but your speedup may be less than that argument predicts.
  What did it break? Compute the bank each thread touches to confirm.
- Why is it illegal to put `__syncthreads()` inside `if (tid < stride)`?
- v5 uses `__syncthreads()` exactly once. Why is that one still necessary, and
  why are the others not?
- Why does `__shfl_down_sync` take a mask (`0xffffffff`)? What breaks if some
  lanes have already returned?
- Why does the last tree step of v3 still waste 31 of 32 lanes, and what would
  you do about it?
- Compare your best reduction's GB/s against the `copy` ceiling you measured in
  exercise 2. Should they match? Do they?

## The sum column

Look at it before you decide something is broken. Then explain it — the
explanation is short, and it has consequences that will come back in Phase 6 when
you try to reproduce an LLM's output bit-exactly across batch sizes.

## Variations worth running

```bash
./bin/reduction 1024        # which version wins on a small array? why a different one?
./bin/reduction 268435456
```

Change `BLOCK` from 256 to 64 and 1024 and explain the shape of the result.

## Where this shows up later

Softmax is a reduction (max, then sum of exponentials). LayerNorm and RMSNorm are
reductions (mean, variance). Attention scores are reductions. You will
reimplement softmax and RMSNorm in Triton in Phase 9 — and the reason Triton is
pleasant to use is that it hides exactly the machinery you are about to write by
hand five times.
