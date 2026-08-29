// Exercise 1 — Vector addition.
//
// The harness below is complete: it allocates, transfers, times with CUDA
// events, verifies against the CPU result, and reports GB/s against the
// device's peak. What is missing is every kernel. Fill in the TODOs.
//
// It compiles and runs as-is; every correctness check will fail until you
// implement the kernels.
//
// Build:  make vector_add
// Run:    ./bin/vector_add [N]

#include <functional>

#include "../common/cuda_utils.cuh"

#define SCATTER 32

// ============================================================================
// TODO 1 — CPU baseline
//
//   c[i] = a[i] + b[i]  for i in [0, n)
// ============================================================================

void vectorAddCpu(const float *a, const float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// ============================================================================
// TODO 2 — one thread per element
//
// Each thread computes exactly one output element. Derive the global index
// from blockIdx, blockDim, and threadIdx.
//
// n is not a multiple of the block size, so the grid covers slightly more
// elements than exist. Guard accordingly.
// ============================================================================

__global__ void vectorAddKernel(const float *a, const float *b, float *c,
                                int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// ============================================================================
// TODO 3 — grid-stride loop
//
// Same result, but the grid is sized to the GPU rather than to the data, so
// each thread may handle several elements. The driver launches this with a
// fixed grid of (SMs * 32) blocks regardless of n, so a thread that only
// handles one element will produce a wrong answer for large n.
//
// The stride is the total number of threads in the grid.
// ============================================================================

__global__ void vectorAddGridStride(const float *a, const float *b, float *c,
                                    int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = gridDim.x * blockDim.x;
  while (i < n) {
    c[i] = a[i] + b[i];
    i += stride;
  }
}

// ============================================================================
// TODO 4 — the same additions, deliberately uncoalesced
//
// This must produce output identical to TODO 2. The only thing that changes is
// which thread is responsible for which element.
//
// Given a thread id t, work on element:
//
//     i = (t % SCATTER) * (n / SCATTER) + (t / SCATTER)
//
// With n a multiple of SCATTER this is a bijection over [0, n), so every
// element is still written exactly once. Adjacent threads in a warp now land
// n/SCATTER floats apart instead of adjacent.
//
// Guard against t >= n as before.
// ============================================================================

__global__ void vectorAddScattered(const float *a, const float *b, float *c,
                                   int n) {
  int t = threadIdx.x + blockDim.x * blockIdx.x; // t = thread id
  int i = (t % SCATTER) * (n / SCATTER) + (t / SCATTER);
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// ------------------------------------------------------------------ reporting

// Effective bandwidth: bytes the kernel genuinely had to move, over time.
// See the README for why the multiplier is 3.
static double effectiveBandwidthGBs(int n, float ms) {
  return (3.0 * n * sizeof(float)) / (ms * 1e-3) / 1e9;
}

static void reportOccupancy(int blockSize, const cudaDeviceProp &p) {
  int maxBlocksPerSm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocksPerSm, vectorAddKernel, blockSize, 0));

  int activeWarps = maxBlocksPerSm * (blockSize / p.warpSize);
  int maxWarps = p.maxThreadsPerMultiProcessor / p.warpSize;
  std::printf("             occupancy: %2d blocks/SM, %2d/%d warps = %.0f%%\n",
              maxBlocksPerSm, activeWarps, maxWarps,
              100.0 * activeWarps / maxWarps);
}

// ----------------------------------------------------------------------- main

int main(int argc, char **argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);  // 16M elements
  n = (n / SCATTER) * SCATTER;  // keep the scattered mapping a clean bijection
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  cudaDeviceProp prop = printDeviceInfo();
  double peak = peakBandwidthGBs(prop);
  std::printf("N = %d elements (%.1f MB per array)\n\n", n, bytes / 1e6);

  float *hA = static_cast<float *>(std::malloc(bytes));
  float *hB = static_cast<float *>(std::malloc(bytes));
  float *hC = static_cast<float *>(std::malloc(bytes));
  float *hRef = static_cast<float *>(std::malloc(bytes));
  fillRandom(hA, n, 1);
  fillRandom(hB, n, 2);

  // --- CPU baseline -------------------------------------------------------
  CpuTimer cpu;
  cpu.start();
  vectorAddCpu(hA, hB, hRef, n);
  cpu.stop();
  std::printf("CPU  : %8.3f ms   (%.1f GB/s effective)\n", cpu.elapsedMs(),
              effectiveBandwidthGBs(n, static_cast<float>(cpu.elapsedMs())));

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));

  // --- Transfer cost, measured on its own ---------------------------------
  float h2dMs = timeGpuMs(
      [&] {
        cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);
      },
      2, 10);
  float d2hMs = timeGpuMs(
      [&] { cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost); }, 2, 10);

  std::printf("H2D  : %8.3f ms   (%.1f GB/s, 2 arrays)\n", h2dMs,
              (2.0 * bytes) / (h2dMs * 1e-3) / 1e9);
  std::printf("D2H  : %8.3f ms   (%.1f GB/s, 1 array)\n\n", d2hMs,
              static_cast<double>(bytes) / (d2hMs * 1e-3) / 1e9);

  CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  const int blockSizes[] = {32, 64, 128, 256, 512, 1024};
  float bestMs = 1e30f;
  float scatteredMs = 0.0f;

  // Runs one kernel variant across the block-size sweep and prints a table.
  // The <<<>>> syntax needs a literal kernel name, so each variant passes a
  // lambda that captures the launch rather than a function pointer.
  using Launch = std::function<void(int grid, int block)>;
  auto sweep = [&](const char *title, bool gridStride, const Launch &launch,
                   bool showOccupancy, float *recordAt256) {
    std::printf("--- %s ---\n", title);
    std::printf("%12s %12s %12s %11s %9s\n", "block size", "kernel ms", "GB/s",
                "% of peak", "correct");

    for (int bs : blockSizes) {
      int grid = gridStride ? prop.multiProcessorCount * 32 : (n + bs - 1) / bs;

      CUDA_CHECK(cudaMemset(dC, 0, bytes));
      float ms = timeGpuMs([&] { launch(grid, bs); });
      CUDA_CHECK_KERNEL();

      CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
      bool ok = allCloseQuiet(hC, hRef, n);

      double gbs = effectiveBandwidthGBs(n, ms);
      if (ok && ms < bestMs) bestMs = ms;
      if (recordAt256 && bs == 256) *recordAt256 = ms;

      std::printf("%12d %12.4f %12.1f %10.0f%% %9s\n", bs, ms, gbs,
                  100.0 * gbs / peak, ok ? "yes" : "NO");
      if (showOccupancy) reportOccupancy(bs, prop);
    }
    std::printf("\n");
  };

  sweep("one thread per element", false,
        [&](int g, int b) { vectorAddKernel<<<g, b>>>(dA, dB, dC, n); }, true,
        nullptr);
  sweep("grid-stride, grid sized to the GPU", true,
        [&](int g, int b) { vectorAddGridStride<<<g, b>>>(dA, dB, dC, n); },
        false, nullptr);
  sweep("uncoalesced, same arithmetic", false,
        [&](int g, int b) { vectorAddScattered<<<g, b>>>(dA, dB, dC, n); },
        false, &scatteredMs);

  // --- The two headline numbers -------------------------------------------
  if (bestMs < 1e29f && scatteredMs > 0.0f) {
    std::printf("Coalesced vs uncoalesced: %.4f ms vs %.4f ms  ->  %.1fx\n",
                bestMs, scatteredMs, scatteredMs / bestMs);
    std::printf("One round trip: %.3f ms of transfer for %.4f ms of compute "
                "(%.0fx).\n",
                h2dMs + d2hMs, bestMs, (h2dMs + d2hMs) / bestMs);
  } else {
    std::printf("(Implement the kernels to get the summary numbers.)\n");
  }

  std::free(hA);
  std::free(hB);
  std::free(hC);
  std::free(hRef);
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
