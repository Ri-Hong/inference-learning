// Exercise 2 — Matrix transpose.
//
// Transpose does ZERO arithmetic. Every kernel below must move exactly the same
// bytes and produce exactly the same output, so any difference in runtime is
// purely the memory access pattern.
//
// The driver is complete. Implement the four kernels.
//
// Build:  make transpose
// Run:    ./bin/transpose [N]     (square N x N matrix, default 4096)

#include <functional>
#include <vector>

#include "../common/cuda_utils.cuh"

#define TILE 32
#define BLOCK_ROWS 8  // each thread handles TILE/BLOCK_ROWS = 4 elements

// Blocks are launched as dim3(TILE, BLOCK_ROWS) = 32 x 8 threads over a grid of
// (n/TILE) x (n/TILE). So each block owns a TILE x TILE patch of the matrix and
// each thread handles TILE/BLOCK_ROWS elements of it, one per loop iteration:
//
//     for (int j = 0; j < TILE; j += BLOCK_ROWS) { ... row (y + j) ... }
//
// All four kernels share that skeleton.

// ============================================================================
// TODO 1 — copy (the bandwidth ceiling, not a transpose)
//
//   out[r][c] = in[r][c]
//
// Same read+write volume as a transpose with a perfect access pattern on both
// sides. No transpose kernel can beat this, so it is the number every other row
// in the table should be judged against.
//
// Guard both x and (y + j) against n.
// ============================================================================

__global__ void copyKernel(float *out, const float *in, int n) {
  // TODO
  (void)out; (void)in; (void)n;
}

// ============================================================================
// TODO 2 — naive transpose
//
//   out[c][r] = in[r][c]
//
// The obvious one-liner. Read along a row, write down a column (or the other
// way round — try both and see whether it matters).
// ============================================================================

__global__ void naiveTranspose(float *out, const float *in, int n) {
  // TODO
  (void)out; (void)in; (void)n;
}

// ============================================================================
// TODO 3 — tiled transpose via shared memory
//
// Stage a TILE x TILE patch in shared memory so that BOTH the global read and
// the global write are coalesced, and the awkward strided access happens in
// shared memory instead of DRAM.
//
// Shape:
//   __shared__ float tile[TILE][TILE];
//   phase 1: read a patch from `in`, write it into `tile`
//   __syncthreads();
//   phase 2: read `tile` transposed, write it to `out`
//
// Two things to get right:
//   * the barrier — the tile is written by some threads and read by others
//   * in phase 2 the block must write to the TRANSPOSED block position, which
//     means recomputing x and y with blockIdx.x and blockIdx.y swapped.
//     Otherwise you have transposed within each tile but not between tiles.
// ============================================================================

__global__ void tiledTranspose(float *out, const float *in, int n) {
  // TODO
  (void)out; (void)in; (void)n;
}

// ============================================================================
// TODO 4 — tiled transpose, padded
//
// Identical to TODO 3 with one character changed:
//
//   __shared__ float tile[TILE][TILE + 1];
//
// Everything else is the same. Predict the effect before you measure it, then
// work out from the shared-memory banking rules why it happens.
// ============================================================================

__global__ void tiledPaddedTranspose(float *out, const float *in, int n) {
  // TODO
  (void)out; (void)in; (void)n;
}

// ============================================================================
// TODO 5 — CPU reference, used to verify the kernels
// ============================================================================

static void transposeCpu(float *out, const float *in, int n) {
  // TODO
  (void)out; (void)in; (void)n;
}

// ----------------------------------------------------------------------- main

int main(int argc, char **argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : 4096;
  n = ((n + TILE - 1) / TILE) * TILE;  // round up to a whole number of tiles
  size_t count = static_cast<size_t>(n) * n;
  size_t bytes = count * sizeof(float);

  cudaDeviceProp prop = printDeviceInfo();
  double peak = peakBandwidthGBs(prop);
  std::printf("Matrix: %d x %d floats (%.1f MB)\n", n, n, bytes / 1e6);
  std::printf("Grid:   %d x %d blocks of %d x %d threads\n\n", n / TILE,
              n / TILE, TILE, BLOCK_ROWS);

  float *hIn = static_cast<float *>(std::malloc(bytes));
  float *hOut = static_cast<float *>(std::malloc(bytes));
  float *hRef = static_cast<float *>(std::malloc(bytes));
  fillRandom(hIn, count);
  transposeCpu(hRef, hIn, n);

  float *dIn, *dOut;
  CUDA_CHECK(cudaMalloc(&dIn, bytes));
  CUDA_CHECK(cudaMalloc(&dOut, bytes));
  CUDA_CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

  dim3 grid(n / TILE, n / TILE);
  dim3 block(TILE, BLOCK_ROWS);

  // A transpose reads every element once and writes it once.
  auto gbs = [&](float ms) { return (2.0 * bytes) / (ms * 1e-3) / 1e9; };

  struct Variant {
    const char *name;
    std::function<void()> launch;
    bool isTranspose;
  };
  const std::vector<Variant> variants = {
      {"copy (ceiling)", [&] { copyKernel<<<grid, block>>>(dOut, dIn, n); },
       false},
      {"naive", [&] { naiveTranspose<<<grid, block>>>(dOut, dIn, n); }, true},
      {"tiled shared", [&] { tiledTranspose<<<grid, block>>>(dOut, dIn, n); },
       true},
      {"tiled + padded",
       [&] { tiledPaddedTranspose<<<grid, block>>>(dOut, dIn, n); }, true},
  };

  std::printf("%18s %12s %12s %11s %10s\n", "kernel", "ms", "GB/s", "% of peak",
              "correct");

  float copyMs = 0.0f;
  for (const Variant &v : variants) {
    CUDA_CHECK(cudaMemset(dOut, 0, bytes));
    float ms = timeGpuMs(v.launch, 3, 20);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    bool ok = v.isTranspose ? allCloseQuiet(hOut, hRef, count)
                            : allCloseQuiet(hOut, hIn, count);

    std::printf("%18s %12.4f %12.1f %10.0f%% %10s\n", v.name, ms, gbs(ms),
                100.0 * gbs(ms) / peak, ok ? "yes" : "NO");

    if (!v.isTranspose) copyMs = ms;
  }

  std::printf("\nCompare each row against the copy ceiling (%.4f ms), not "
              "against\neach other.\n", copyMs);

  std::free(hIn);
  std::free(hOut);
  std::free(hRef);
  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dOut));
  return 0;
}
