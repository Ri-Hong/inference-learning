// Exercise 3 — Sum reduction, and what synchronization actually costs.
//
// Reduction is the "hello world" of cooperative GPU programming: threads must
// combine results, so they must communicate, so they must synchronize.
//
// Five versions, each fixing one specific inefficiency in the previous one.
// Write them in order and measure after each — the point is the deltas, not
// the final number.
//
// The host driver (including the multi-pass plumbing that reduces an array to a
// single value) is complete. Implement the kernels.
//
// Build:  make reduction
// Run:    ./bin/reduction [N]
//
// Reference worth reading once you have your own numbers: Mark Harris,
// "Optimizing Parallel Reduction in CUDA" (NVIDIA). Try to get there yourself
// first.

#include <functional>
#include <vector>

#include "../common/cuda_utils.cuh"

#define BLOCK 256

// All of v1-v4 have the same contract:
//
//   in   : `n` floats
//   out  : one float per block — out[blockIdx.x] = sum of this block's slice
//
// and the same skeleton: stage the block's slice into shared memory, tree-
// reduce it, have thread 0 write the result. The driver calls each kernel
// repeatedly (feeding its output back in) until one value remains.
//
// Shared memory is allocated dynamically at launch, so declare it as:
//
//   extern __shared__ float s[];

// ============================================================================
// TODO v1 — interleaved addressing, divergent branch
//
//   for (stride = 1; stride < blockDim.x; stride *= 2)
//       threads where (tid % (2 * stride) == 0) do  s[tid] += s[tid + stride]
//
// Write it exactly this way, warts and all. It is the baseline the other four
// are measured against. Note where the barrier has to go.
//
// Before you run it: which lanes of a warp are active at each step?
// ============================================================================

__global__ void reduce_v1(const float *in, float *out, int n) {
  // TODO
  (void)in; (void)out; (void)n;
}

// ============================================================================
// TODO v2 — interleaved addressing, non-divergent
//
// Same additions in the same order, but reindexed so the ACTIVE threads are the
// low-numbered ones, letting whole warps retire instead of every warp running
// half idle:
//
//   index = 2 * stride * tid;
//   if (index + stride < blockDim.x)  s[index] += s[index + stride];
//
// This fixes one problem and introduces another. Work out what, and predict
// whether v2 beats v1 by more or less than you would guess from the divergence
// argument alone.
// ============================================================================

__global__ void reduce_v2(const float *in, float *out, int n) {
  // TODO
  (void)in; (void)out; (void)n;
}

// ============================================================================
// TODO v3 — sequential addressing
//
// Halve the stride each step instead of doubling it:
//
//   for (stride = blockDim.x / 2; stride > 0; stride >>= 1)
//       if (tid < stride)  s[tid] += s[tid + stride];
//
// Active threads stay contiguous AND their shared-memory addresses stay
// contiguous. This is the version most people should write first.
// ============================================================================

__global__ void reduce_v3(const float *in, float *out, int n) {
  // TODO
  (void)in; (void)out; (void)n;
}

// ============================================================================
// TODO v4 — first add during the global load
//
// In v3, half the threads do nothing after the very first iteration. Give each
// thread two elements to load and add them on the way into shared memory.
//
// This means each block now consumes 2 * blockDim.x elements, so the driver
// launches half as many blocks (see the `multiplier` field in the variant
// table). Index accordingly:
//
//   i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
//
// and guard both i and i + blockDim.x.
// ============================================================================

__global__ void reduce_v4(const float *in, float *out, int n) {
  // TODO
  (void)in; (void)out; (void)n;
}

// ============================================================================
// TODO v5a — warp-level reduction with shuffles
//
// Sum a value across the 32 lanes of one warp, leaving the total in lane 0,
// with no shared memory and no __syncthreads():
//
//   for (offset = warpSize / 2; offset > 0; offset >>= 1)
//       v += __shfl_down_sync(0xffffffffu, v, offset);
//
// __shfl_down_sync lets a lane read another lane's register directly. The first
// argument is the mask of participating lanes.
// ============================================================================

__inline__ __device__ float warpReduceSum(float v) {
  // TODO
  return v;
}

// ============================================================================
// TODO v5b — grid-stride load + warp shuffle
//
// Different shape from v1-v4. The driver launches a FIXED grid sized to the GPU
// and calls the kernel twice (see launchShuffleReduction below), so:
//
//   1. Each thread accumulates over a grid-stride loop across the whole array.
//   2. Reduce within each warp with warpReduceSum — no barrier needed.
//   3. Lane 0 of each warp writes its total to __shared__ float warpSums[32].
//   4. __syncthreads(), then warp 0 reduces those partials the same way.
//   5. Thread 0 writes out[blockIdx.x].
//
// Step 4 is the only place shared memory and a barrier appear at all. Note that
// blockDim.x / warpSize warps wrote to warpSums, so lanes beyond that must
// contribute zero.
// ============================================================================

__global__ void reduce_v5(const float *in, float *out, int n) {
  // TODO
  (void)in; (void)out; (void)n;
}

// ----------------------------------------------------------------- host driver

// The <<<>>> syntax needs a literal kernel name, so each variant supplies a
// small lambda rather than a function pointer.
using LaunchPass =
    std::function<void(const float *src, float *dst, int n, int blocks)>;

// v1-v4 reduce one array to one partial per block, so we run them repeatedly
// until a single value is left. Returns the buffer holding the final sum.
//
// Launches only -- no synchronization. Anything that blocks the host (a
// cudaMemcpy, a cudaDeviceSynchronize) inside a timing loop measures the sync,
// not the kernel.
static const float *launchTreeReduction(const LaunchPass &pass, int multiplier,
                                        const float *dIn, float *dTmpA,
                                        float *dTmpB, int n) {
  const float *src = dIn;
  float *dst = dTmpA;
  int remaining = n;

  while (remaining > 1) {
    int elemsPerBlock = BLOCK * multiplier;
    int blocks = (remaining + elemsPerBlock - 1) / elemsPerBlock;
    pass(src, dst, remaining, blocks);
    remaining = blocks;
    src = dst;
    dst = (dst == dTmpA) ? dTmpB : dTmpA;
  }
  return src;
}

static void launchShuffleReduction(const float *dIn, float *dTmpA, int n,
                                   int gridSize) {
  reduce_v5<<<gridSize, BLOCK>>>(dIn, dTmpA, n);
  reduce_v5<<<1, BLOCK>>>(dTmpA, dTmpA, gridSize);
}

static float readBack(const float *dPtr) {
  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, dPtr, sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

int main(int argc, char **argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
  size_t bytes = static_cast<size_t>(n) * sizeof(float);

  cudaDeviceProp prop = printDeviceInfo();
  double peak = peakBandwidthGBs(prop);
  int gridV5 = prop.multiProcessorCount * 16;
  std::printf("N = %d elements (%.1f MB), block size %d\n\n", n, bytes / 1e6,
              BLOCK);

  float *hIn = static_cast<float *>(std::malloc(bytes));
  for (int i = 0; i < n; ++i) hIn[i] = 1.0f;  // exact sum = n, easy to verify

  // Summed in double on the host so float rounding in the kernels shows up as a
  // small relative error rather than being hidden by an equally-wrong reference.
  double refSum = 0.0;
  for (int i = 0; i < n; ++i) refSum += hIn[i];

  float *dIn, *dTmpA, *dTmpB;
  CUDA_CHECK(cudaMalloc(&dIn, bytes));
  CUDA_CHECK(cudaMalloc(&dTmpA, bytes));
  CUDA_CHECK(cudaMalloc(&dTmpB, bytes));
  CUDA_CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

  // A reduction reads n floats and writes almost nothing, so effective
  // bandwidth is (n * 4 bytes) / time. It is purely memory bound, so a good
  // reduction should land close to peak.
  auto gbs = [&](float ms) {
    return (static_cast<double>(bytes)) / (ms * 1e-3) / 1e9;
  };

  const size_t shmem = BLOCK * sizeof(float);
  struct Variant {
    const char *name;
    LaunchPass pass;
    int multiplier;  // elements consumed per block, in units of BLOCK
  };
  const std::vector<Variant> variants = {
      {"v1 divergent",
       [=](const float *s, float *d, int m, int b) {
         reduce_v1<<<b, BLOCK, shmem>>>(s, d, m);
       },
       1},
      {"v2 non-divergent",
       [=](const float *s, float *d, int m, int b) {
         reduce_v2<<<b, BLOCK, shmem>>>(s, d, m);
       },
       1},
      {"v3 sequential",
       [=](const float *s, float *d, int m, int b) {
         reduce_v3<<<b, BLOCK, shmem>>>(s, d, m);
       },
       1},
      {"v4 load+add",
       [=](const float *s, float *d, int m, int b) {
         reduce_v4<<<b, BLOCK, shmem>>>(s, d, m);
       },
       2},
  };

  std::printf("%20s %12s %12s %11s %16s %9s\n", "kernel", "ms", "GB/s",
              "% of peak", "sum", "correct");

  float baseline = 0.0f;
  for (const Variant &v : variants) {
    const float *dResult =
        launchTreeReduction(v.pass, v.multiplier, dIn, dTmpA, dTmpB, n);
    CUDA_CHECK_KERNEL();
    float sum = readBack(dResult);

    float ms = timeGpuMs(
        [&] {
          launchTreeReduction(v.pass, v.multiplier, dIn, dTmpA, dTmpB, n);
        },
        3, 20);

    if (baseline == 0.0f) baseline = ms;
    bool ok = std::fabs(sum - refSum) / refSum < 1e-4;
    std::printf("%20s %12.4f %12.1f %10.0f%% %16.1f %9s\n", v.name, ms, gbs(ms),
                100.0 * gbs(ms) / peak, sum, ok ? "yes" : "NO");
  }

  {
    launchShuffleReduction(dIn, dTmpA, n, gridV5);
    CUDA_CHECK_KERNEL();
    float sum = readBack(dTmpA);
    float ms =
        timeGpuMs([&] { launchShuffleReduction(dIn, dTmpA, n, gridV5); }, 3, 20);
    bool ok = std::fabs(sum - refSum) / refSum < 1e-4;
    std::printf("%20s %12.4f %12.1f %10.0f%% %16.1f %9s\n", "v5 shuffle", ms,
                gbs(ms), 100.0 * gbs(ms) / peak, sum, ok ? "yes" : "NO");
    if (baseline > 0.0f && ms > 0.0f) {
      std::printf("\nv5 vs v1, doing identical arithmetic: %.1fx\n",
                  baseline / ms);
    }
  }

  std::printf("\nThe exact answer is %.1f. Watch the sum column across "
              "versions.\n", refSum);

  std::free(hIn);
  CUDA_CHECK(cudaFree(dIn));
  CUDA_CHECK(cudaFree(dTmpA));
  CUDA_CHECK(cudaFree(dTmpB));
  return 0;
}
