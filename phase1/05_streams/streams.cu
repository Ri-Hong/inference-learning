// Exercise 5 — Launch overhead and streams.
//
// The lesson plan lists "kernel launch overhead" and "streams" as topics with
// no exercise attached. They matter later: LLM decode launches hundreds of tiny
// kernels per token, and when each kernel takes less time than its own launch,
// the GPU sits idle waiting on the CPU. That is the problem CUDA graphs exist
// to solve (Phase 6).
//
// Part 1 measures the floor: what does launching a kernel that does nothing
//        actually cost?                                    [given]
// Part 2 shows the consequence: fixed work split into many tiny kernels vs few
//        large ones.                                        [given]
// Part 3 overlaps transfer with compute using streams.      [TODO]
//
// Build:  make streams
// Run:    ./bin/streams

#include <vector>

#include "../common/cuda_utils.cuh"

__global__ void emptyKernel() {}

__global__ void scaleKernel(float *data, int n, float factor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] * factor + 1.0f;
}

// ============================================================================
// TODO 1 — a kernel with a tunable amount of arithmetic
//
// Each thread loads data[i], applies `rounds` iterations of some cheap
// dependent operation (e.g. v = v * 1.0001f + 0.0001f), and stores it back.
//
// The dependency chain is the point: it makes the kernel take a controllable
// amount of time so Part 3 can compare kernel time against transfer time.
// You will change `rounds` later to flip which one dominates.
// ============================================================================

__global__ void busyKernel(float *data, int n, int rounds) {
  // TODO
  (void)data; (void)n; (void)rounds;
}

int main() {
  printDeviceInfo();

  // ---------------------------------------------------------------- Part 1
  // An empty kernel does no memory traffic and no arithmetic. Whatever time it
  // takes is pure overhead: the driver packaging the launch, the command
  // reaching the GPU, the block scheduler spinning it up.
  std::printf("--- Part 1: what does a launch cost when the kernel does "
              "nothing? ---\n");

  float asyncMs = timeGpuMs([&] { emptyKernel<<<1, 1>>>(); }, 100, 1000);
  std::printf("  empty kernel, back to back      : %8.2f us\n",
              asyncMs * 1000.0f);

  // Now force the host to wait for each one. This is what happens whenever you
  // call .item(), .cpu(), print a tensor, or synchronize in a loop.
  float syncMs = timeGpuMs(
      [&] {
        emptyKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
      },
      20, 200);
  std::printf("  empty kernel + synchronize each : %8.2f us\n",
              syncMs * 1000.0f);
  std::printf("  cost of the synchronize         : %8.2f us\n\n",
              (syncMs - asyncMs) * 1000.0f);

  // ---------------------------------------------------------------- Part 2
  // Same total elements, same total arithmetic, chopped into different numbers
  // of kernel launches.
  std::printf("--- Part 2: same work, different number of launches ---\n");

  const int total = 1 << 22;  // 4M elements
  size_t bytes = static_cast<size_t>(total) * sizeof(float);
  float *dData;
  CUDA_CHECK(cudaMalloc(&dData, bytes));
  CUDA_CHECK(cudaMemset(dData, 0, bytes));

  std::printf("%12s %12s %14s %14s\n", "launches", "elems each", "total ms",
              "us / launch");

  for (int chunks : {1, 16, 256, 4096, 65536}) {
    int per = total / chunks;
    if (per < 1) continue;
    int block = 256;
    int grid = (per + block - 1) / block;

    float ms = timeGpuMs(
        [&] {
          for (int c = 0; c < chunks; ++c) {
            scaleKernel<<<grid, block>>>(dData + (size_t)c * per, per, 1.0f);
          }
        },
        2, 5);
    CUDA_CHECK_KERNEL();

    std::printf("%12d %12d %14.3f %14.2f\n", chunks, per, ms,
                ms * 1000.0f / chunks);
  }
  std::printf("\n  Where does this table stop measuring the kernel?\n\n");

  // ---------------------------------------------------------------- Part 3
  // Streams. Work in different streams may overlap; work in the same stream is
  // ordered. Overlapping H2D, compute, and D2H turns three serial phases into a
  // pipeline.
  std::printf("--- Part 3: overlapping transfer and compute with streams ---\n");

  const int n = 1 << 22;
  size_t nbytes = static_cast<size_t>(n) * sizeof(float);
  const int rounds = 200;  // change this later and rerun

  // Pinned (page-locked) host memory. cudaMemcpyAsync from ordinary pageable
  // memory silently falls back to a synchronous copy, so without this the whole
  // experiment quietly does nothing.
  float *hPinned;
  CUDA_CHECK(cudaMallocHost(&hPinned, nbytes));
  for (int i = 0; i < n; ++i) hPinned[i] = 1.0f;

  float *dBuf;
  CUDA_CHECK(cudaMalloc(&dBuf, nbytes));

  // Baseline: copy everything in, compute, copy everything out. Three serial
  // phases on the default stream.
  float serialMs = timeGpuMs(
      [&] {
        cudaMemcpy(dBuf, hPinned, nbytes, cudaMemcpyHostToDevice);
        busyKernel<<<(n + 255) / 256, 256>>>(dBuf, n, rounds);
        cudaMemcpy(hPinned, dBuf, nbytes, cudaMemcpyDeviceToHost);
      },
      2, 10);
  CUDA_CHECK_KERNEL();
  std::printf("  serial (1 stream, blocking copies) : %8.3f ms\n", serialMs);

  for (int nStreams : {2, 4, 8}) {
    std::vector<cudaStream_t> streams(nStreams);
    for (auto &s : streams) CUDA_CHECK(cudaStreamCreate(&s));

    int chunk = n / nStreams;
    size_t chunkBytes = static_cast<size_t>(chunk) * sizeof(float);

    // ========================================================================
    // TODO 2 — pipeline the same work across nStreams streams
    //
    // Split the array into nStreams chunks. For chunk i, issue all three
    // operations into streams[i] so they stay ordered relative to each other
    // but are free to overlap with the other chunks' work:
    //
    //   cudaMemcpyAsync(..., cudaMemcpyHostToDevice, streams[i]);
    //   busyKernel<<<blocks, 256, 0, streams[i]>>>(...);
    //   cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, streams[i]);
    //
    // The offset for chunk i is i * chunk elements into both buffers. Finish
    // with a cudaDeviceSynchronize() inside the timed lambda so the measurement
    // covers the whole pipeline.
    //
    // Predict the speedup for 2, 4, and 8 streams before you run it. What caps
    // it?
    // ========================================================================
    float ms = timeGpuMs(
        [&] {
          for (int i = 0; i < nStreams; ++i) {
            size_t off = static_cast<size_t>(i) * chunk;
            (void)off;
            (void)chunkBytes;
            // TODO
          }
          cudaDeviceSynchronize();
        },
        2, 10);
    CUDA_CHECK_KERNEL();

    std::printf("  %d streams, async chunked          : %8.3f ms  (%.2fx)\n",
                nStreams, ms, serialMs / ms);

    for (auto &s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  }

  std::printf("\n  Once it works: set `rounds` to 5 and rerun. The speedup\n"
              "  should change character. Explain why.\n");

  CUDA_CHECK(cudaFreeHost(hPinned));
  CUDA_CHECK(cudaFree(dBuf));
  CUDA_CHECK(cudaFree(dData));
  return 0;
}
