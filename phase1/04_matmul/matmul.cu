// Exercise 4 — Matrix multiplication (GEMM).
//
// The first compute-bound kernel in Phase 1. Vector add and transpose were
// limited by DRAM bandwidth; a large GEMM does O(N^3) arithmetic on O(N^2)
// data, so with enough reuse it can actually saturate the math units.
//
// cuBLAS is wired up as the reference and the ceiling. Implement the three
// kernels and see how close you get.
//
// Build:  make matmul       (links -lcublas)
// Run:    ./bin/matmul [N]  (square N x N x N, default 2048)

#include <cublas_v2.h>

#include <functional>
#include <vector>

#include "../common/cuda_utils.cuh"

#define TILE 32

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st_ = (call);                                               \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,     \
                   static_cast<int>(st_));                                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// All three kernels compute C = A * B with row-major storage:
//
//   A is M x K,  B is K x N,  C is M x N
//   A[i][j] == A[i * K + j],  B[i][j] == B[i * N + j],  C[i][j] == C[i * N + j]
//
// Blocks are dim3(TILE, TILE) = 32 x 32 threads.

// ============================================================================
// TODO 1 — naive GEMM, one thread per output element
//
//   col comes from blockIdx.x / threadIdx.x
//   row comes from blockIdx.y / threadIdx.y
//
// Each thread walks the full K dimension accumulating A[row][k] * B[k][col]
// into a local float, then writes C[row][col] once.
//
// Guard row < M and col < N.
//
// Before running: for a fixed k, what addresses do the 32 threads of a warp
// read from A? From B?
// ============================================================================

__global__ void matmulNaive(const float *A, const float *B, float *C, int M,
                            int N, int K) {
  // TODO
  (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
}

// ============================================================================
// TODO 2 — the same kernel with row and col swapped
//
// Copy TODO 1 verbatim and change exactly one thing: take `row` from
// blockIdx.x / threadIdx.x and `col` from blockIdx.y / threadIdx.y.
//
//   int row = blockIdx.x * blockDim.x + threadIdx.x;
//   int col = blockIdx.y * blockDim.y + threadIdx.y;
//
// Everything else — the loop, the accumulation, the store — is identical, and
// the output must be identical too. (The driver launches this one with the grid
// dimensions swapped to match; see gridRowMajorX below.)
//
// This pair is the Phase 1 milestone in a single diff. Predict the ratio before
// you measure it.
// ============================================================================

__global__ void matmulNaiveSwapped(const float *A, const float *B, float *C,
                                   int M, int N, int K) {
  // TODO
  (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
}

// ============================================================================
// TODO 3 — tiled GEMM via shared memory
//
// Naive GEMM reads each element of A and B N times from DRAM. Tiling drops that
// to N/TILE by staging tiles in shared memory and reusing them.
//
//   __shared__ float As[TILE][TILE];
//   __shared__ float Bs[TILE][TILE];
//
//   for each tile t along K:
//       cooperatively load A's tile and B's tile into As / Bs
//       __syncthreads();
//       for (k = 0; k < TILE; ++k)  acc += As[ty][k] * Bs[k][tx];
//       __syncthreads();
//   write C[row][col] = acc
//
// Both barriers are mandatory and they guard different things — the first that
// the tiles are fully loaded before anyone reads them, the second that everyone
// has finished reading before the next iteration overwrites them. Try removing
// each one separately and see what breaks.
//
// K is not necessarily a multiple of TILE, so out-of-range loads should write
// 0.0f into the tile rather than being skipped.
// ============================================================================

__global__ void matmulTiled(const float *A, const float *B, float *C, int M,
                            int N, int K) {
  // TODO
  (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
}

// ----------------------------------------------------------------------- main

int main(int argc, char **argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : 2048;
  int M = n, N = n, K = n;
  size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

  printDeviceInfo();
  std::printf("GEMM: (%d x %d) x (%d x %d)\n", M, K, K, N);

  // 2 flops (one multiply, one add) per (i,j,k) triple.
  double gflop = 2.0 * M * N * K / 1e9;
  std::printf("Work: %.2f GFLOP\n", gflop);

  // Arithmetic intensity: FLOPs per byte of DRAM traffic, in the ideal case
  // where each matrix is read exactly once.
  double idealBytes = static_cast<double>(bytesA + bytesB + bytesC);
  std::printf("Ideal arithmetic intensity: %.1f FLOP/byte\n\n",
              (gflop * 1e9) / idealBytes);

  float *hA = static_cast<float *>(std::malloc(bytesA));
  float *hB = static_cast<float *>(std::malloc(bytesB));
  float *hC = static_cast<float *>(std::malloc(bytesC));
  float *hRef = static_cast<float *>(std::malloc(bytesC));
  fillRandom(hA, static_cast<size_t>(M) * K, 1);
  fillRandom(hB, static_cast<size_t>(K) * N, 2);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.0f, beta = 0.0f;

  // cuBLAS is column-major. C = A*B in row-major has the same memory layout as
  // C^T = B^T * A^T in column-major, so we hand it B and A in that order and
  // read the result back as row-major with no transposes anywhere. This is a
  // layout footgun, not part of the exercise — it is written out for you.
  auto launchCublas = [&] {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K,
                &beta, dC, N);
  };

  // cuBLAS is the ground truth for correctness. A CPU GEMM at N=2048 would take
  // minutes.
  launchCublas();
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(hRef, dC, bytesC, cudaMemcpyDeviceToHost));

  std::printf("%22s %12s %12s %14s %10s\n", "kernel", "ms", "GFLOP/s",
              "vs cuBLAS", "correct");

  auto gflops = [&](float ms) { return gflop / (ms * 1e-3); };

  float cublasMs = timeGpuMs(launchCublas, 5, 20);
  CUDA_CHECK_KERNEL();

  dim3 block(TILE, TILE);
  // Grid dimensions follow whichever axis each kernel maps to the row.
  dim3 gridColMajorX((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  dim3 gridRowMajorX((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  struct Variant {
    const char *name;
    std::function<void()> launch;
  };
  const std::vector<Variant> variants = {
      {"naive (row=blockIdx.y)",
       [&] { matmulNaive<<<gridColMajorX, block>>>(dA, dB, dC, M, N, K); }},
      {"naive (row=blockIdx.x)",
       [&] {
         matmulNaiveSwapped<<<gridRowMajorX, block>>>(dA, dB, dC, M, N, K);
       }},
      {"tiled shared memory",
       [&] { matmulTiled<<<gridColMajorX, block>>>(dA, dB, dC, M, N, K); }},
  };

  for (const Variant &v : variants) {
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    v.launch();
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    // Loose tolerance: cuBLAS accumulates in a different order, and with
    // K=2048 random terms the fp32 rounding difference is real but small.
    bool ok = allCloseQuiet(hC, hRef, static_cast<size_t>(M) * N, 1e-2f, 1e-3f);

    float ms = timeGpuMs(v.launch, 3, 10);
    CUDA_CHECK_KERNEL();

    std::printf("%22s %12.3f %12.1f %13.2fx %10s\n", v.name, ms, gflops(ms),
                ms / cublasMs, ok ? "yes" : "NO");
  }

  std::printf("%22s %12.3f %12.1f %13.2fx %10s\n", "cuBLAS", cublasMs,
              gflops(cublasMs), 1.0, "ref");

  CUBLAS_CHECK(cublasDestroy(handle));
  std::free(hA);
  std::free(hB);
  std::free(hC);
  std::free(hRef);
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
