# Phase 1 results — <date>

## Environment

```text
Date:
Host:                    (RunPod pod type / region, or wherever this ran)
GPU:                     (nvidia-smi --query-gpu=name --format=csv)
Driver:
CUDA (nvcc --version):
Compute capability:
SMs:
Peak DRAM bandwidth:     (printed by every exercise)
```

Record the GPU every time. Numbers from a 4090 and an A100 are not comparable —
different memory technology, different bandwidth, different fp32/fp16 ratios —
so a results file without the device on it is worth nothing later.

---

## 1. Vector add

**Predictions**

| question | my guess |
|---|---|
| fastest block size | smallest block |
| best % of peak bandwidth | 100% |
| scattered slowdown factor | ? |
| transfer vs kernel time | transfer time |

**Output**

```text
(paste)
```

**What surprised me**

**Explanation**

---

## 2. Transpose

**Predictions**

| kernel | predicted GB/s |
|---|---|
| copy (ceiling) | |
| naive | |
| tiled | |
| tiled + padded | |

**Output**

```text
(paste)
```

**Explanation**

Why padding by one float changed the number it did:

---

## 3. Reduction

**Predicted ranking**

1.
2.
3.
4.
5.

**Output**

```text
(paste)
```

**Biggest single jump was** v__ → v__, because:

**On the differing sums**

---

## 4. Matmul

**Output — CUDA**

```text
(paste)
```

**Output — PyTorch**

```text
(paste)
```

| kernel | GFLOP/s | % of cuBLAS |
|---|---|---|
| naive | | |
| naive swapped | | |
| tiled | | |
| cuBLAS | | 100% |
| torch fp32 | | |
| torch fp16 | | |

**Does torch fp32 match cuBLAS?** If not, why not?

**Arithmetic intensity of a single decode step** (`[1, 4096] @ [4096, 4096]`):

---

## 5. Streams and launch overhead

```text
(paste)
```

| measurement | value |
|---|---|
| empty kernel launch | µs |
| launch + synchronize | µs |
| chunk count where overhead dominates | |
| best stream speedup | x |

---

## Milestone

> Explain why two kernels doing the same arithmetic can have very different
> performance.

Answer it in your own words, citing your own numbers:

**Open questions to carry into Phase 2**

1.
2.
3.
