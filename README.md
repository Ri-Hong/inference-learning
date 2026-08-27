# LLM Inference Learning

A hands-on learning repo for building practical intuition around **CUDA, PyTorch GPU execution, vLLM, SGLang, Triton, profiling, and multi-GPU inference**.

The goal is not just to learn how to launch an LLM server. The goal is to understand **why inference systems perform the way they do**, from CUDA kernels all the way up to request schedulers and distributed serving.

## Learning path

1. CUDA fundamentals
2. PyTorch on CUDA
3. Transformer inference fundamentals
4. Hugging Face baseline inference
5. vLLM
6. vLLM internals and KV-cache management
7. SGLang
8. Triton kernels
9. Profiling with Nsight
10. Performance experiments
11. Multi-GPU inference

See [`LESSON_PLAN.md`](LESSON_PLAN.md) for the full curriculum.

## Suggested repo structure

```text
llm-inference-learning/
├── README.md
├── LESSON_PLAN.md
├── cuda/
│   ├── vector_add/
│   ├── reduction/
│   ├── matmul/
│   ├── softmax/
│   └── rmsnorm/
├── pytorch/
├── vllm/
├── sglang/
├── triton/
├── profiling/
├── multi_gpu/
├── results/
└── notes/
```

## Core question

By the end of this project, you should be able to answer:

> What determines LLM inference and serving performance?

That includes GPU compute throughput, memory bandwidth, model size, KV-cache behavior, prefill vs decode, batching, scheduling, attention kernels, quantization, CUDA graphs, tensor parallelism, communication overhead, NCCL, and hardware topology.

## Working style

For every experiment:

1. Form a hypothesis.
2. Change one variable.
3. Measure.
4. Record the result.
5. Explain why it happened.

## Metrics to record

- TTFT — time to first token
- ITL — inter-token latency
- output tokens/sec
- total tokens/sec
- requests/sec
- GPU utilization
- VRAM usage
- power draw
- batch size
- concurrency
- input length
- output length

## First session

```bash
nvidia-smi
nvcc --version
```

Then:

1. Write a CUDA vector-add kernel.
2. Benchmark it.
3. Change threads-per-block.
4. Write a simple matrix multiplication kernel.
5. Compare against PyTorch.
6. Observe GPU utilization and kernel timings.

Do not begin by reading all of vLLM's source code. Build the GPU-performance mental model first.
