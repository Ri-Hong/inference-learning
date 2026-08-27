# LLM Inference Systems — Lesson Plan

## Objective

Build an end-to-end understanding of modern LLM inference systems, beginning with GPU programming and ending with multi-GPU serving.

By the end, you should be able to write and reason about CUDA kernels, understand PyTorch GPU execution, explain prefill/decode and KV cache, benchmark inference correctly, operate and inspect vLLM and SGLang, write simple Triton kernels, profile workloads, and reason about distributed inference.

---

# Phase 1 — CUDA Fundamentals

## Goal

Understand the GPU programming model well enough to reason about kernel performance.

## Topics

- host vs device
- kernels
- threads, blocks, grids
- warps and SIMT
- global/shared/register memory
- synchronization
- coalesced access
- kernel launch overhead
- streams
- CUDA events
- occupancy

## Exercises

### Vector addition
Implement CPU and CUDA versions. Benchmark kernel time and transfer time.

Vary threads per block:

```text
32
64
128
256
512
```

### Matrix transpose
Implement naive and shared-memory tiled versions.

### Reduction
Implement a sum reduction and study synchronization.

### Naive matrix multiplication
Implement basic GEMM and compare with:

```python
torch.matmul()
```

## Milestone

Explain why two kernels doing the same arithmetic can have very different performance.

---

# Phase 2 — PyTorch on CUDA

## Goal

Connect high-level ML operations to GPU execution.

## Topics

- asynchronous execution
- `torch.cuda.synchronize()`
- CUDA events
- GEMM / cuBLAS
- Tensor Cores
- FP32 / FP16 / BF16
- arithmetic intensity
- memory-bound vs compute-bound
- kernel fusion

## Exercises

Benchmark:

```python
x @ W
torch.softmax(x)
torch.nn.functional.layer_norm(...)
```

across tensor sizes, dtypes, and batch sizes.

## Questions

- Why is the first invocation often slower?
- Why do larger matmuls often utilize the GPU better?
- Why do FP16/BF16 often beat FP32?
- Why can small GPU ops be inefficient?

---

# Phase 3 — Transformer Inference Fundamentals

## Goal

Understand autoregressive inference before introducing a serving engine.

## Topics

- transformer forward pass
- attention
- MLP
- prefill
- decode
- KV cache
- weights vs activations
- context length
- memory bandwidth

## Exercises

Estimate model-weight memory:

```text
8B parameters × 2 bytes ≈ 16 GB
```

Repeat for FP32, FP16/BF16, INT8, and 4-bit.

Investigate KV-cache memory separately.

## Milestone

Explain why prefill and decode are fundamentally different workloads.

---

# Phase 4 — Hugging Face Baseline

## Goal

Experience basic inference without an optimized serving engine.

Choose a manageable 3B–8B model.

Measure:

- model load time
- VRAM usage
- TTFT
- tokens/sec
- GPU utilization

Test batches 1, 2, 4, and 8.

Use:

```bash
watch -n 1 nvidia-smi
```

Important question:

> Why might one interactive generation request fail to fully utilize a powerful GPU?

---

# Phase 5 — vLLM Basics

## Goal

Understand what a serving engine adds.

Start with:

```bash
vllm serve <model>
```

## Concepts

- continuous batching
- request scheduling
- KV-cache management
- PagedAttention
- attention backends
- GPU memory utilization

## Experiment 1 — Concurrency

Test:

```text
1
2
4
8
16
32
```

Record TTFT, ITL, output tok/s, total tok/s, requests/s, utilization, and VRAM.

## Experiment 2 — Prompt length

Test approximately:

```text
128
512
1K
4K
8K
```

Hold output length constant.

## Experiment 3 — Output length

Hold prompt length constant and generate:

```text
64
256
1024
```

tokens.

## Milestone

Understand the throughput/latency tradeoff as concurrency rises.

---

# Phase 6 — vLLM Internals

## Goal

Trace how one request becomes GPU work.

Do not read the repo top-to-bottom.

Trace:

```text
HTTP request
    ↓
API server
    ↓
scheduler
    ↓
batch construction
    ↓
model executor
    ↓
attention layer
    ↓
backend
    ↓
CUDA / Triton kernel
```

Investigate:

- scheduler
- request state
- batching
- model executor
- KV-cache allocation
- attention backend selection
- CUDA graphs

Add logging or breakpoints and answer:

> What happens between receiving a request and launching GPU kernels?

---

# Phase 7 — KV Cache and PagedAttention

## Goal

Understand why KV-cache management is central to serving.

Study:

- cache growth
- block/page allocation
- fragmentation
- maximum concurrent sequences
- context length
- prefix caching

Experiment with context length, concurrency, and GPU memory utilization.

## Milestone

Explain why KV-cache management directly affects serving throughput and concurrency.

---

# Phase 8 — SGLang

## Goal

Compare modern inference engines under identical workloads.

Use the same model, hardware, input lengths, output lengths, and concurrency levels.

Suggested table:

| Workload | Hugging Face | vLLM | SGLang |
|---|---:|---:|---:|
| Single request | | | |
| Concurrency 8 | | | |
| Concurrency 32 | | | |
| 4K prompt | | | |
| Shared prefix | | | |

Investigate:

- scheduler design
- RadixAttention
- prefix caching
- structured generation
- speculative decoding

Later:
- prefill/decode disaggregation
- distributed serving

Focus on *why* results differ, not only who wins.

---

# Phase 9 — Triton

## Goal

Learn a modern ML-kernel abstraction.

Reimplement:

- vector add
- softmax
- RMSNorm

Compare:

```text
PyTorch
vs
Triton
vs
custom CUDA
```

across tensor sizes.

## Milestone

Be comfortable reading Triton code in inference repositories.

---

# Phase 10 — Profiling

## Goal

Observe what the GPU actually does.

### Nsight Systems

Inspect:

- CPU threads
- CUDA API calls
- kernel launches
- streams
- GPU kernels
- memory transfers
- synchronization
- cuBLAS activity

Profile a short inference request and identify prefill vs decode.

### Nsight Compute

Use it on individual kernels to inspect:

- memory throughput
- compute utilization
- occupancy
- cache behavior
- instruction mix

## Milestone

Look at a profile and form a plausible bottleneck hypothesis.

---

# Phase 11 — Performance Experiments

For each experiment:

1. predict the result
2. run it
3. record metrics
4. explain deviations

Change one variable at a time:

- batch size
- concurrency
- input length
- output length
- precision
- quantization
- attention backend
- CUDA graphs
- model size
- KV-cache allocation

Classify workloads as primarily compute-bound, memory-bandwidth-bound, latency-bound, or communication-bound.

---

# Phase 12 — Multi-GPU Inference

## Goal

Understand distributed inference after mastering the single-GPU case.

Progress:

```text
1 GPU → 2 GPUs → 4 GPUs
```

Topics:

- tensor parallelism
- NCCL
- all-reduce
- PCIe
- NVLink
- topology
- scaling efficiency

Measure 1-, 2-, and 4-GPU throughput.

Ask:

> Why doesn't four GPUs automatically provide four times the performance?

---

# Suggested Timeline

## Week 1
CUDA fundamentals.

Deliverables:
- vector add
- transpose
- reduction
- naive matmul
- benchmark notes

## Week 2
PyTorch + transformer inference.

Deliverables:
- CUDA timing experiments
- dtype benchmarks
- Hugging Face baseline
- notes on prefill/decode/KV cache

## Week 3
vLLM.

Deliverables:
- server
- concurrency benchmark
- input-length benchmark
- output-length benchmark

## Week 4
vLLM internals.

Deliverables:
- request execution trace
- scheduler notes
- KV-cache/PagedAttention explanation

## Week 5
SGLang.

Deliverables:
- server
- vLLM/SGLang comparison
- shared-prefix experiment

## Week 6
Triton.

Deliverables:
- vector add
- softmax
- RMSNorm
- performance comparison

## Week 7
Profiling.

Deliverables:
- Nsight Systems trace
- prefill/decode analysis
- one Nsight Compute kernel analysis

## Week 8+
Multi-GPU and advanced serving.

Deliverables:
- tensor-parallel benchmarks
- NCCL observations
- scaling analysis

---

# Benchmark Template

```text
Date:
GPU:
GPU count:
Driver:
CUDA:
Framework:
Framework version/commit:
Model:
Precision:
Quantization:
Input tokens:
Output tokens:
Concurrency:
Batching configuration:

TTFT:
ITL:
Output tok/s:
Total tok/s:
Requests/s:
Peak VRAM:
Average GPU utilization:
Power:

Hypothesis:
Result:
Explanation:
Open questions:
```

---

# Final Project

Write a report:

## What Determines LLM Serving Performance?

Use your own experiments to explain:

- GPU compute
- memory bandwidth
- model size
- precision
- prefill
- decode
- KV cache
- continuous batching
- scheduling
- attention kernels
- quantization
- CUDA graphs
- prefix caching
- tensor parallelism
- NCCL
- interconnects
- communication overhead

---

# Immediate Next Steps

1. Confirm GPU access:

```bash
nvidia-smi
nvcc --version
```

2. Create `cuda/vector_add/`.
3. Implement CPU vector addition.
4. Implement CUDA vector addition.
5. Benchmark 32/64/128/256/512 threads per block.
6. Record the results.
7. Implement naive matrix multiplication.
8. Compare with PyTorch `torch.matmul`.
9. Write down why the performance differs.

Only then move into transformer serving.
